"""
Static Regional Crops Generator
================================
Generates data/reference/regional_crops.json for all 640 districts
using agro-climatic zone rules — NO API calls, NO quota issues.

Run from agri_crop_recommendation/:
    python scripts/generate_regional_crops_static.py
    python scripts/generate_regional_crops_static.py --only-missing
"""

import sys, os, json, argparse, logging
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
REGIONS_JSON  = _PACKAGE_ROOT / "data/reference/regions.json"
OUTPUT_JSON   = _PACKAGE_ROOT / "data/reference/regional_crops.json"

# ── Zone helper ──────────────────────────────────────────────────────────────
def _get_zone(region_id: str) -> str:
    from src.weather.history import get_zone_for_region
    return get_zone_for_region(region_id)

# ── Crop suitability rules ────────────────────────────────────────────────────
# Each rule: (crop_id, base_score, conditions)
# conditions is a dict with optional keys:
#   climate_zones  : list of climate_zone strings that ALLOW this crop
#   exclude_climates: list of climate_zone strings that BLOCK this crop
#   min_lat, max_lat: latitude range
#   min_elev, max_elev: elevation range (m)
#   regions        : list of region zone strings (North/South/East/West/Central/Northeast)
#   states         : list of state names that are famous for it (higher score)
#   states_block   : list of state names where crop is impossible
#   seasons        : crop requires at least one of these supported_seasons

ALL_CROPS = [
    "BAJRA_01","JOWAR_01","RAGI_01","FOXTAIL_01",
    "MOONG_01","URAD_01","COWPEA_01","GUAR_01",
    "SESAME_01","SUNFLOWER_01","SOYBEAN_01",
    "TOMATO_01","BRINJAL_01","OKRA_01","BOTTLEGOURD_01",
    "SPINACH_01","FENUGREEK_01","CORIANDER_01","AMARANTH_01",
    "MUSTARD_GREENS_01","LETTUCE_01","RADISH_01","SPRING_ONION_01",
    "CARROT_01","TURNIP_01","BEETROOT_01",
    "CUCUMBER_01","RIDGE_GOURD_01","BITTER_GOURD_01",
    "FRENCH_BEANS_01","CLUSTER_BEANS_01","CAPSICUM_01","GREEN_CHILLI_01",
    "SPONGE_GOURD_01","PUMPKIN_01",
    "MOONG_DAL_01","URAD_DAL_01","COWPEA_VEG_01","MASOOR_01",
    "MINT_01","DILL_01","CURRY_LEAVES_01",
    "BABY_POTATO_01","BABY_CORN_01","MICROGREENS_01","DRUMSTICK_LEAVES_01",
]

# Crops that grow EVERYWHERE in India (base vegetables / short-duration)
UNIVERSAL_CROPS = {
    "TOMATO_01": 0.80,
    "BRINJAL_01": 0.80,
    "OKRA_01": 0.80,
    "BOTTLEGOURD_01": 0.80,
    "SPINACH_01": 0.75,
    "CORIANDER_01": 0.80,
    "RADISH_01": 0.75,
    "SPRING_ONION_01": 0.75,
    "CUCUMBER_01": 0.78,
    "RIDGE_GOURD_01": 0.75,
    "BITTER_GOURD_01": 0.75,
    "SPONGE_GOURD_01": 0.75,
    "PUMPKIN_01": 0.78,
    "GREEN_CHILLI_01": 0.80,
    "MICROGREENS_01": 0.85,
    "MINT_01": 0.78,
    "DILL_01": 0.72,
    "AMARANTH_01": 0.75,
    "MOONG_DAL_01": 0.75,
    "URAD_DAL_01": 0.73,
    "BABY_CORN_01": 0.72,
}

def _score(region: dict, zone: str) -> dict:
    """Return {crop_id: score} for a given region using agro-climatic rules."""
    state      = region.get("state", "")
    climate    = region.get("climate_zone", "")
    lat        = region.get("latitude", 20.0)
    elev       = region.get("elevation", 100)
    seasons    = region.get("supported_seasons", [])
    soil_types = region.get("typical_soil_types", [])

    approved = dict(UNIVERSAL_CROPS)  # start with universals

    # ── helpers ──────────────────────────────────────────────────────────────
    def _is_arid():
        return any(x in climate for x in ["Arid", "Semi-Arid", "Desert"])

    def _is_humid():
        return any(x in climate for x in ["Humid", "Tropical Humid", "Sub-Humid"])

    def _is_tropical():
        return "Tropical" in climate

    def _is_highland():
        return elev > 1000

    def _is_cold():
        return elev > 1500 or lat > 30

    def _has_rabi():
        return "Rabi" in seasons

    def _has_kharif():
        return "Kharif" in seasons

    # ── Millets ──────────────────────────────────────────────────────────────
    # Bajra: hot dry climates, all India except extreme NE highlands
    if not (elev > 2000):
        score = 0.70
        if _is_arid(): score = 0.90
        if state in ["Rajasthan", "Gujarat", "Haryana", "Uttar Pradesh"]: score = 0.92
        if elev > 1200: score = max(0.50, score - 0.20)
        approved["BAJRA_01"] = round(score, 2)

    # Jowar: warm semi-arid
    if not _is_cold():
        score = 0.72
        if _is_arid() or "Semi-Arid" in climate: score = 0.88
        if state in ["Maharashtra", "Karnataka", "Madhya Pradesh", "Telangana", "Andhra Pradesh"]: score = 0.92
        if elev > 1500: score = max(0.50, score - 0.20)
        approved["JOWAR_01"] = round(score, 2)

    # Ragi: semi-arid / tropical, peninsular India + Karnataka famous
    if lat < 28 and not _is_cold():
        score = 0.70
        if state in ["Karnataka", "Tamil Nadu", "Andhra Pradesh", "Odisha", "Jharkhand"]: score = 0.92
        if "Tropical" in climate: score = max(score, 0.78)
        approved["RAGI_01"] = round(score, 2)

    # Foxtail Millet: warm dry areas
    if not _is_cold() and not (elev > 1800):
        score = 0.65
        if state in ["Andhra Pradesh", "Telangana", "Karnataka", "Tamil Nadu"]: score = 0.82
        approved["FOXTAIL_01"] = round(score, 2)

    # ── Pulses ───────────────────────────────────────────────────────────────
    # Moong: warm kharif
    if _has_kharif() and not _is_cold():
        score = 0.78
        if state in ["Rajasthan", "Maharashtra", "Andhra Pradesh", "Telangana", "Madhya Pradesh"]: score = 0.90
        approved["MOONG_01"] = round(score, 2)

    # Urad: warm humid kharif
    if _has_kharif() and not _is_cold() and not _is_arid():
        score = 0.75
        if state in ["Andhra Pradesh", "Telangana", "Uttar Pradesh", "Tamil Nadu", "Maharashtra"]: score = 0.88
        approved["URAD_01"] = round(score, 2)

    # Cowpea: warm tropical
    if not _is_cold() and not (elev > 1500):
        score = 0.72
        if _is_humid() or _is_tropical(): score = 0.82
        approved["COWPEA_01"] = round(score, 2)

    # Guar: arid/semi-arid
    score = 0.65
    if _is_arid(): score = 0.90
    if state in ["Rajasthan", "Haryana", "Gujarat"]: score = 0.92
    if _is_humid(): score = max(0.50, score - 0.15)
    if not _is_cold():
        approved["GUAR_01"] = round(score, 2)

    # Masoor: rabi, cool areas
    if _has_rabi():
        score = 0.72
        if state in ["Uttar Pradesh", "Madhya Pradesh", "Bihar", "West Bengal", "Rajasthan"]: score = 0.88
        if _is_cold(): score = max(score, 0.80)
        approved["MASOOR_01"] = round(score, 2)

    # ── Oilseeds ─────────────────────────────────────────────────────────────
    # Sesame: warm dry kharif
    if _has_kharif() and not _is_cold():
        score = 0.70
        if _is_arid() or "Semi-Arid" in climate: score = 0.85
        if state in ["Gujarat", "Rajasthan", "West Bengal", "Madhya Pradesh", "Uttar Pradesh"]: score = 0.88
        approved["SESAME_01"] = round(score, 2)

    # Sunflower: widely adaptable but not extreme cold/humid
    if not (elev > 1800):
        score = 0.72
        if state in ["Karnataka", "Andhra Pradesh", "Telangana", "Maharashtra", "Odisha"]: score = 0.88
        if _is_cold(): score = max(0.55, score - 0.15)
        approved["SUNFLOWER_01"] = round(score, 2)

    # Soybean: humid/sub-humid kharif
    if _has_kharif() and not _is_arid() and not _is_cold():
        score = 0.75
        if state in ["Madhya Pradesh", "Maharashtra", "Rajasthan", "Karnataka"]: score = 0.92
        approved["SOYBEAN_01"] = round(score, 2)

    # ── Cool-season vegetables ────────────────────────────────────────────────
    # Fenugreek: dry/semi-arid rabi
    if _has_rabi():
        score = 0.75
        if _is_arid() or "Semi-Arid" in climate: score = 0.88
        if state in ["Rajasthan", "Gujarat", "Madhya Pradesh"]: score = 0.92
        approved["FENUGREEK_01"] = round(score, 2)

    # Mustard Greens: cool season, north India
    if _has_rabi() and (lat > 20 or _is_cold()):
        score = 0.75
        if state in ["Punjab", "Haryana", "Uttar Pradesh", "Bihar", "Rajasthan"]: score = 0.92
        if _is_cold(): score = max(score, 0.85)
        approved["MUSTARD_GREENS_01"] = round(score, 2)

    # Lettuce: cool climate
    if _is_cold() or _is_highland() or lat > 28:
        score = 0.70
        if elev > 1500: score = 0.85
        approved["LETTUCE_01"] = round(score, 2)
    elif _has_rabi() and not _is_tropical():
        approved["LETTUCE_01"] = 0.60

    # Carrot: cool rabi
    if _has_rabi():
        score = 0.75
        if _is_cold() or elev > 800: score = 0.88
        if state in ["Punjab", "Haryana", "Uttar Pradesh", "Himachal Pradesh", "Uttarakhand"]: score = 0.90
        approved["CARROT_01"] = round(score, 2)

    # Turnip: cold highland
    if _is_cold() or elev > 800:
        score = 0.72
        if elev > 1500: score = 0.88
        approved["TURNIP_01"] = round(score, 2)
    elif _has_rabi():
        approved["TURNIP_01"] = 0.60

    # Beetroot: cool rabi
    if _has_rabi():
        score = 0.72
        if _is_cold() or elev > 1000: score = 0.85
        approved["BEETROOT_01"] = round(score, 2)

    # ── Legume vegetables ─────────────────────────────────────────────────────
    # French Beans: cool & highland tropical
    if _is_cold() or elev > 800 or (_is_humid() and lat < 20):
        score = 0.75
        if elev > 1200: score = 0.88
        if state in ["Himachal Pradesh", "Uttarakhand", "Jammu & Kashmir", "Karnataka", "Tamil Nadu"]: score = 0.90
        approved["FRENCH_BEANS_01"] = round(score, 2)
    elif _has_rabi():
        approved["FRENCH_BEANS_01"] = 0.62

    # Cluster Beans (vegetable): warm dry
    if not _is_cold() and not (elev > 1500):
        score = 0.70
        if _is_arid(): score = 0.85
        approved["CLUSTER_BEANS_01"] = round(score, 2)

    # Cowpea vegetable: warm tropical
    if not _is_cold():
        score = 0.72
        if _is_humid(): score = 0.82
        approved["COWPEA_VEG_01"] = round(score, 2)

    # Capsicum: moderate climate, not extreme arid/cold
    if not (elev > 2000) and not state in ["Ladakh"]:
        score = 0.72
        if elev > 800: score = 0.82
        if state in ["Himachal Pradesh", "Uttarakhand", "Karnataka", "Maharashtra"]: score = 0.88
        approved["CAPSICUM_01"] = round(score, 2)

    # ── Herbs & specialty ─────────────────────────────────────────────────────
    # Curry Leaves: tropical south India
    if _is_tropical() or lat < 20:
        score = 0.80
        if state in ["Tamil Nadu", "Karnataka", "Andhra Pradesh", "Telangana", "Kerala", "Maharashtra"]: score = 0.92
        approved["CURRY_LEAVES_01"] = round(score, 2)
    elif not _is_cold():
        approved["CURRY_LEAVES_01"] = 0.60

    # Drumstick (Moringa): tropical/semi-arid warm
    if not _is_cold() and not (elev > 1200):
        score = 0.72
        if _is_tropical() or lat < 20: score = 0.88
        if state in ["Tamil Nadu", "Andhra Pradesh", "Karnataka", "Telangana", "Gujarat"]: score = 0.92
        approved["DRUMSTICK_LEAVES_01"] = round(score, 2)

    # Baby Potato: cool rabi / highland
    if _has_rabi() or _is_cold():
        score = 0.75
        if _is_cold() or elev > 1000: score = 0.88
        if state in ["Uttar Pradesh", "Bihar", "West Bengal", "Punjab", "Himachal Pradesh"]: score = 0.92
        approved["BABY_POTATO_01"] = round(score, 2)

    return approved


def main():
    parser = argparse.ArgumentParser(description="Generate regional_crops.json statically")
    parser.add_argument("--only-missing", action="store_true",
                        help="Skip districts already enriched in the output file")
    parser.add_argument("--state", type=str, default=None,
                        help="Process only a specific state prefix (e.g. MH)")
    args = parser.parse_args()

    regions = json.loads(REGIONS_JSON.read_text(encoding="utf-8"))
    all_regions = regions if isinstance(regions, list) else regions.get("regions", [])
    logger.info(f"Loaded {len(all_regions)} regions")

    if args.state:
        all_regions = [r for r in all_regions
                       if r.get("region_id", "").startswith(args.state.upper() + "_")]
        logger.info(f"Filtered to {len(all_regions)} regions for state {args.state}")

    existing = {}
    if OUTPUT_JSON.exists():
        existing = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
        logger.info(f"Loaded {len(existing)} existing enrichments")

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    done = skipped = 0
    for region in all_regions:
        region_id = region.get("region_id", "")
        if not region_id:
            continue

        if args.only_missing and region_id in existing:
            skipped += 1
            continue

        zone     = _get_zone(region_id)
        approved = _score(region, zone)
        excluded = [c for c in ALL_CROPS if c not in approved]

        existing[region_id] = {
            "name":         region.get("name", region_id),
            "state":        region.get("state", ""),
            "zone":         zone,
            "approved":     approved,
            "excluded":     excluded,
            "source":       "static_rules_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        done += 1
        logger.info(f"[{done:4d}] {region_id:35s} -> {len(approved):2d} crops approved")

    OUTPUT_JSON.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"  Static enrichment complete!")
    print(f"  Done   : {done}")
    print(f"  Skipped: {skipped} (already present)")
    print(f"  Total  : {len(existing)} districts in file")
    print(f"  Output : {OUTPUT_JSON}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
