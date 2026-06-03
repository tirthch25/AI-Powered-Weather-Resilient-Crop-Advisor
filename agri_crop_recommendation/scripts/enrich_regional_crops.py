"""
Enrich regional crop database using Gemini AI.

Queries Gemini for each district to determine which crops can be grown there,
then saves results to data/reference/regional_crops.json.

The recommendation engine loads this file at startup to provide accurate,
district-specific crop suitability scores for all 640 districts.

Run from agri_crop_recommendation/:
    python scripts/enrich_regional_crops.py
    python scripts/enrich_regional_crops.py --only-missing   # skip completed districts
    python scripts/enrich_regional_crops.py --state MH       # single state only
"""

import sys, os, json, time, argparse, logging
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

def _enable_debug():
    logging.getLogger().setLevel(logging.DEBUG)

# ── Paths ─────────────────────────────────────────────────────────────────────
# Anchor all paths to the agri_crop_recommendation/ package root (parent of scripts/)
_PACKAGE_ROOT  = Path(__file__).resolve().parent.parent
REGIONS_JSON   = _PACKAGE_ROOT / "data/reference/regions.json"
OUTPUT_JSON    = _PACKAGE_ROOT / "data/reference/regional_crops.json"
ENV_FILE       = _PACKAGE_ROOT / ".env"

# ── Load environment ───────────────────────────────────────────────────────────
def _load_env():
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

_load_env()

def _collect_api_keys() -> list:
    """Auto-discover all GEMINI_API_KEY, GEMINI_API_KEY_2, GEMINI_API_KEY_3 ... from env."""
    keys = []
    # Always check the base key first
    base = os.environ.get("GEMINI_API_KEY", "").strip()
    if base:
        keys.append(("key1", base))
    # Then check numbered variants: GEMINI_API_KEY_2, _3, _4 ... up to _20
    for i in range(2, 21):
        k = os.environ.get(f"GEMINI_API_KEY_{i}", "").strip()
        if k:
            keys.append((f"key{i}", k))
        # Stop scanning once we hit two consecutive missing keys
        elif not os.environ.get(f"GEMINI_API_KEY_{i+1}", "").strip():
            break
    return keys

# ── All crops from the database ────────────────────────────────────────────────
def _get_all_crops():
    from src.crops.database import CROPS_DATA
    # CROPS_DATA is a list of CropInfo objects
    return [(crop.crop_id, crop.common_name) for crop in CROPS_DATA]

# ── Zone helper ───────────────────────────────────────────────────────────────
def _get_zone(region_id: str) -> str:
    from src.weather.history import get_zone_for_region
    return get_zone_for_region(region_id)

# ── Build compact crop list string ────────────────────────────────────────────
def _crop_list_str(crops):
    return ", ".join(f"{cid}={name}" for cid, name in crops)

# ── Call Gemini for a single district (with retry) ────────────────────────────
def _ask_gemini(district_name: str, state_name: str, zone: str,
                region_id: str, crop_list_str: str, client,
                model: str = "gemini-2.0-flash",
                max_retries: int = 1, key_label: str = "key1") -> dict:
    prompt = (
        f"You are an Indian agricultural expert with deep knowledge of district-level farming.\n"
        f"District: {district_name}, {state_name}. Agro-climatic zone: {zone}.\n\n"
        f"From the following crop list, identify ALL crops that CAN be cultivated in this district "
        f"(include crops grown with basic irrigation, not just rainfed traditional crops).\n"
        f"Include vegetables, pulses, oilseeds and short-duration crops if they are feasible.\n"
        f"ONLY exclude crops that are climatically IMPOSSIBLE (e.g. rice in extreme desert, "
        f"apple in tropical plains, tea in arid regions).\n\n"
        f"Crop list: {crop_list_str}\n\n"
        f"For each approved crop, also give a suitability score (0.50 to 1.00) where:\n"
        f"  1.00 = this district is famous for growing it\n"
        f"  0.80 = commonly grown, suitable conditions\n"
        f"  0.65 = can be grown, moderate suitability\n"
        f"  0.50 = possible with effort/irrigation\n\n"
        f"Reply ONLY with valid JSON (no markdown):\n"
        f"{{\"approved\":{{\"CROP_ID\": score, ...}}, \"excluded\":[\"CROP_ID\", ...]}}"
    )

    for attempt in range(max_retries + 1):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
            )
            raw = resp.text.strip()
            # Strip markdown code block if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
        except Exception as e:
            err_str = str(e)
            # Always log the raw error so we can diagnose quota issues
            logger.debug(f"[{key_label}] Raw error (attempt {attempt}): {err_str[:600]}")

            # ── 429 / RESOURCE_EXHAUSTED (quota) ──────────────────────────────
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                import re as _re

                err_lower = err_str.lower()

                # Detect DAILY quota exhaustion.
                # The Gemini API error often contains BOTH PerMinute AND PerDay quotaIds
                # in its details section — we must check for daily signals first.
                # Priority signals (treat as daily even if PerMinute also appears):
                #   1. "PerDay" in a quotaId (GenerateRequestsPerDayPerProjectPerModel-FreeTier)
                #   2. "limit: 0" in the message body (daily quota = 0 available)
                #   3. "You exceeded your current quota" (Gemini's daily exhausted message)
                #   4. Any "generate_content_free_tier" metric (these are daily-scoped)
                perday_signal = (
                    "PerDay" in err_str or
                    "GenerateRequestsPerDay" in err_str or
                    "daily" in err_lower
                )
                freetier_signal = "generate_content_free_tier" in err_str
                limit_zero_signal = "limit: 0" in err_lower
                exceeded_signal = "you exceeded your current quota" in err_lower

                # Pure RPM keywords (only present when it's truly a per-minute limit,
                # not mixed with a daily quota violation)
                pure_rpm = (
                    ("PerMinute" in err_str or "per_minute" in err_lower or "GenerateRequestsPerMinute" in err_str)
                    and not perday_signal        # PerMinute alone, without PerDay
                    and not limit_zero_signal    # and limit isn't actually 0
                    and not exceeded_signal      # and it's not the "exceeded quota" message
                )

                daily_quota_exhausted = (
                    perday_signal or
                    exceeded_signal or
                    (freetier_signal and limit_zero_signal)   # free_tier metric + limit=0
                )

                if daily_quota_exhausted:
                    logger.warning(
                        f"[{key_label}] Detected daily quota exhaustion. "
                        f"Error snippet: {err_str[:400]}"
                    )
                    raise RuntimeError(f"DAILY_QUOTA_EXHAUSTED:{key_label}") from e

                if pure_rpm:
                    # Per-minute rate limit — extract retry delay from error message
                    retry_after = 30
                    m = _re.search(r"retry[\s_]?(?:after|in)[\s:]+([\d]+)", err_str, _re.IGNORECASE)
                    if m:
                        retry_after = int(m.group(1)) + 2
                    if attempt < max_retries:
                        wait = min(retry_after * (2 ** attempt), 60)  # cap at 60s
                        logger.warning(
                            f"[{key_label}] Rate limited (RPM). Waiting {wait}s "
                            f"(attempt {attempt+1}/{max_retries}). "
                            f"Tip: use --delay 5 or higher to avoid this."
                        )
                        time.sleep(wait)
                        continue

                # Unknown 429 or exhausted RPM retries — treat as daily quota exhausted
                logger.warning(
                    f"[{key_label}] 429 unclassified or retries exhausted — "
                    f"treating as daily quota exhausted."
                )
                raise RuntimeError(f"DAILY_QUOTA_EXHAUSTED:{key_label}") from e

            # ── 503 / UNAVAILABLE (server overloaded — always retry) ──────────
            elif "503" in err_str or "UNAVAILABLE" in err_str:
                if attempt < max_retries:
                    wait = 10 * (2 ** attempt)  # 10s, 20s, 40s
                    logger.warning(f"Service unavailable (503). Retrying in {wait}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait)
                    continue
            raise  # Non-retryable error or exhausted retries
    raise RuntimeError(f"Failed after {max_retries} retries")


# ── Main enrichment loop ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Enrich regional crop DB using Gemini")
    parser.add_argument("--only-missing", action="store_true",
                        help="Skip districts already in the output JSON")
    parser.add_argument("--state", type=str, default=None,
                        help="Process only a specific state prefix (e.g. MH, UP)")
    parser.add_argument("--delay", type=float, default=5.0,
                        help="Seconds between API calls (default: 5.0). "
                             "Free tier allows 15 RPM → minimum 4s recommended.")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash",
                        help="Gemini model to use (default: gemini-2.0-flash).")
    parser.add_argument("--debug", action="store_true",
                        help="Enable DEBUG logging to print raw API errors for diagnosis.")
    args = parser.parse_args()

    if args.debug:
        _enable_debug()

    _key_pool = _collect_api_keys()
    if not _key_pool:
        print("[ERROR] No GEMINI_API_KEY found in .env")
        sys.exit(1)
    logger.info(f"Loaded {len(_key_pool)} API key(s) for rotation: {[lbl for lbl,_ in _key_pool]}")
    from google import genai as google_genai
    key_clients = [(lbl, google_genai.Client(api_key=k)) for lbl, k in _key_pool]
    exhausted_keys: set = set()

    def _active_client():
        """Return the first non-exhausted (label, client) pair, or None."""
        for lbl, c in key_clients:
            if lbl not in exhausted_keys:
                return lbl, c
        return None, None

    # Load regions
    regions = json.loads(REGIONS_JSON.read_text(encoding="utf-8"))
    all_regions = regions if isinstance(regions, list) else regions.get("regions", [])
    logger.info(f"Loaded {len(all_regions)} regions")

    # Filter by state if requested
    if args.state:
        all_regions = [r for r in all_regions
                       if r.get("region_id", "").startswith(args.state.upper() + "_")]
        logger.info(f"Filtered to {len(all_regions)} regions for state {args.state}")

    # Load existing output
    existing = {}
    if OUTPUT_JSON.exists():
        existing = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
        logger.info(f"Loaded {len(existing)} existing enrichments from {OUTPUT_JSON}")

    # Get all crops
    all_crops = _get_all_crops()
    crop_str  = _crop_list_str(all_crops)
    logger.info(f"Database has {len(all_crops)} crops to evaluate")

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    done = 0
    skipped = 0
    errors  = 0

    for region in all_regions:
        region_id = region.get("region_id", "")
        name      = region.get("name", region_id)
        state     = region.get("state", "")

        if not region_id:
            continue

        # Skip already enriched
        if args.only_missing and region_id in existing:
            skipped += 1
            continue

        zone = _get_zone(region_id)

        # Get active client (may have switched due to quota)
        key_label, client = _active_client()
        if client is None:
            logger.error(
                "All API keys exhausted. Saving progress and stopping.\n"
                f"  Completed so far: {done} districts\n"
                "  Re-run with --only-missing tomorrow to continue."
            )
            OUTPUT_JSON.write_text(
                json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"\n{'='*60}")
            print(f"  Stopped: All API keys quota exhausted.")
            print(f"  Done   : {done}")
            print(f"  Re-run tomorrow: python scripts/enrich_regional_crops.py --only-missing")
            print(f"{'='*60}")
            return

        try:
            result = _ask_gemini(name, state, zone, region_id, crop_str, client,
                                 model=args.model, key_label=key_label)

            approved   = result.get("approved", {})
            # Normalise: approved could be dict {id:score} or list [id]
            if isinstance(approved, list):
                approved = {cid: 0.75 for cid in approved}

            existing[region_id] = {
                "name":         name,
                "state":        state,
                "zone":         zone,
                "approved":     approved,
                "excluded":     result.get("excluded", []),
                "source":       args.model,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            done += 1
            logger.info(
                f"[{done:4d}] {region_id:30s} -> {len(approved):2d} crops approved"
            )

        except RuntimeError as e:
            if "DAILY_QUOTA_EXHAUSTED" in str(e):
                exhausted_label = str(e).split(":", 1)[-1] if ":" in str(e) else key_label
                exhausted_keys.add(exhausted_label)
                remaining = [lbl for lbl, _ in key_clients if lbl not in exhausted_keys]
                if remaining:
                    logger.warning(
                        f"Key '{exhausted_label}' daily quota exhausted. "
                        f"Switching to next key ({remaining[0]}) and retrying {region_id} in 5s..."
                    )
                    # Brief pause before switching — avoid hammering the next key immediately
                    time.sleep(5)
                    continue
                else:
                    logger.error(
                        "All API keys exhausted. Saving progress and stopping.\n"
                        f"  Completed so far: {done} districts\n"
                        "  Re-run with --only-missing tomorrow to continue."
                    )
                    OUTPUT_JSON.write_text(
                        json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8"
                    )
                    print(f"\n{'='*60}")
                    print(f"  Stopped: All API keys quota exhausted.")
                    print(f"  Done   : {done}")
                    print(f"  Re-run tomorrow: python scripts/enrich_regional_crops.py --only-missing")
                    print(f"{'='*60}")
                    return
            errors += 1
            logger.warning(f"[SKIP] {region_id}: {e}")

        except Exception as e:
            errors += 1
            logger.warning(f"[SKIP] {region_id}: {e}")

        # Save incrementally every 10 districts
        if done % 10 == 0 and done > 0:
            OUTPUT_JSON.write_text(
                json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        time.sleep(args.delay)


    # Final save
    OUTPUT_JSON.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n{'='*60}")
    print(f"  Enrichment complete!")
    print(f"  Done   : {done}")
    print(f"  Skipped: {skipped} (already present)")
    print(f"  Errors : {errors}")
    print(f"  Output : {OUTPUT_JSON}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
