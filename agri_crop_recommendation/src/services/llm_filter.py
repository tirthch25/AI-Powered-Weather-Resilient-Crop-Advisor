"""
LLM Regional Crop Filter
========================
Uses an LLM to filter crops that are actually cultivated in a specific
Indian district/region, solving the 552-region coverage gap in the static
crop database.

Primary provider : Ollama  (llama3.2 / gemma3 — runs locally, free, private)
Fallback provider: Gemini  (used automatically if Ollama is not running)

Set LLM_PROVIDER=ollama (default) or LLM_PROVIDER=gemini in .env.
Falls back to the unfiltered list gracefully if both providers fail.
"""

import os
import json
import logging
import re
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Provider configuration ────────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER",    "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL",    "llama3.2")
OLLAMA_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GEMINI_MODEL = "gemini-2.0-flash-lite"

_ollama_client = None
_gemini_client = None


# ── Ollama client ─────────────────────────────────────────────────────────────

def _get_ollama_client():
    global _ollama_client
    if _ollama_client is not None:
        return _ollama_client
    try:
        import ollama
        client = ollama.Client(host=OLLAMA_URL)
        client.list()  # ping
        _ollama_client = client
        logger.info(f"Ollama filter client connected ({OLLAMA_MODEL})")
        return _ollama_client
    except ImportError:
        logger.warning("ollama package not installed — run: pip install ollama")
        return None
    except Exception as e:
        logger.warning(f"Ollama not available for filter ({e}) — trying Gemini")
        return None


# ── Gemini fallback client ────────────────────────────────────────────────────

def _get_gemini_client():
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client
    try:
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None
        _gemini_client = genai.Client(api_key=api_key)
        logger.info(f"Gemini filter fallback client loaded ({GEMINI_MODEL})")
        return _gemini_client
    except Exception as e:
        logger.warning(f"Gemini filter client failed: {e}")
        return None


def _resolve_client():
    """Return (provider, client) — tries Ollama first, then Gemini."""
    if LLM_PROVIDER == "gemini":
        client = _get_gemini_client()
        if client:
            return ("gemini", client)
        logger.warning("LLM_PROVIDER=gemini but unavailable — trying Ollama")

    client = _get_ollama_client()
    if client:
        return ("ollama", client)

    client = _get_gemini_client()
    if client:
        return ("gemini", client)

    return ("none", None)


# ── Compact state-to-zone mapping for prompt context (keeps tokens low) ───────
STATE_ZONE = {
    "MH": "Maharashtra/Marathwada: black cotton soil, semi-arid, crops: jowar, tur, cotton, soybean, bajra",
    "PB": "Punjab: alluvial soil, irrigated, crops: wheat, rice, maize, potato",
    "RJ": "Rajasthan: arid/sandy, crops: bajra, jowar, cluster bean, mustard",
    "UP": "Uttar Pradesh: fertile plain, crops: wheat, rice, sugarcane, potato, pulses",
    "GJ": "Gujarat: semi-arid coastal, crops: cotton, groundnut, castor, cumin, wheat",
    "MP": "Madhya Pradesh: black soil, crops: soybean, wheat, gram, jowar, maize",
    "KA": "Karnataka: diverse zones, crops: ragi, jowar, sugarcane, maize, vegetables",
    "TN": "Tamil Nadu: tropical, crops: rice, sugarcane, banana, groundnut, vegetables",
    "AP": "Andhra Pradesh: crops: rice, chilli, cotton, groundnut, tobacco",
    "TS": "Telangana: crops: cotton, paddy, maize, soybean, red gram",
    "WB": "West Bengal: humid, crops: rice, jute, potato, vegetables",
    "BR": "Bihar: alluvial, crops: rice, wheat, maize, lentil, vegetables",
    "HR": "Haryana: irrigated, crops: wheat, rice, sugarcane, oilseeds",
    "CG": "Chhattisgarh: tribal belt, crops: rice, maize, pulses",
    "OD": "Odisha: crops: rice, pulses, oilseeds, vegetables",
    "AS": "Assam: humid, crops: rice, jute, tea, mustard",
    "HP": "Himachal Pradesh: cool hills, crops: wheat, potato, vegetables, apple",
    "UK": "Uttarakhand: hills, crops: wheat, rice, potato, vegetables",
    "KL": "Kerala: tropical humid, crops: coconut, rubber, rice, banana, spices",
    "JH": "Jharkhand: tribal, crops: rice, maize, pulses",
}


# ── Core filter ───────────────────────────────────────────────────────────────

def llm_filter_crops(
    crop_ids: List[str],
    crop_names: List[str],
    region_id: str,
    region_name: str,
    season: str,
    state: str = "",
) -> Optional[List[str]]:
    """
    Ask the LLM which crops from the list are actually cultivated in this region.

    Returns the approved crop_id list, or None if LLM unavailable (caller falls back
    to the unfiltered rule-based list).
    """
    provider, client = _resolve_client()
    if provider == "none":
        return None

    try:
        state_code = region_id.split("_")[0].upper() if region_id else ""
        zone_hint  = STATE_ZONE.get(state_code, "")

        # Compact crop list: "BAJRA_01=Bajra, JOWAR_01=Jowar, ..."
        crop_list_str = ", ".join(
            f"{cid}={name}" for cid, name in zip(crop_ids, crop_names)
        )

        # Inclusive prompt — only reject crops that are CLIMATICALLY IMPOSSIBLE,
        # not just uncommon. Short-duration vegetables and pulses are always acceptable.
        prompt = (
            f"You are an Indian agriculture expert.\n"
            f"District: {region_name} ({state_code}). Zone: {zone_hint}. Season: {season}.\n"
            f"Crops to evaluate: {crop_list_str}\n\n"
            f"Task: Identify crops that are CLIMATICALLY IMPOSSIBLE here "
            f"(e.g. rice in arid desert, apple in tropics, cold-requiring crops in 40°C heat).\n"
            f"Keep ALL crops that CAN be grown, including ones that are less common.\n"
            f"Short-duration vegetables (okra, brinjal, gourd, tomato, spinach, methi) are "
            f"acceptable in almost any Indian district with irrigation — keep them.\n"
            f"Approve at least 70% of the crops unless there is a strong climatic reason to remove.\n"
            f'Reply ONLY valid JSON: {{"ok":[approved crop_ids],"no":[removed crop_ids]}}'
        )

        if provider == "ollama":
            response = client.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                format="json",   # Ollama enforces valid JSON — eliminates parse errors
            )
            text = response["message"]["content"].strip()

        else:  # gemini
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            text = response.text.strip()

        # Strip any stray markdown fences
        text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()

        data     = json.loads(text)
        approved = data.get("ok", [])
        removed  = data.get("no", [])

        if removed:
            logger.info(f"LLM removed for {region_name}: {removed}")

        valid_ids      = set(crop_ids)
        approved_valid = [c for c in approved if c in valid_ids]

        # Safety net: if LLM is too aggressive (< 5 crops OR < 40% of input),
        # treat the response as unreliable and fall back to rule-based list.
        min_count = max(5, int(len(crop_ids) * 0.40))
        if len(approved_valid) < min_count:
            logger.warning(
                f"LLM returned only {len(approved_valid)}/{len(crop_ids)} crops "
                f"for {region_name} — below safety threshold ({min_count}). "
                f"Falling back to rule-based list."
            )
            return None

        if not approved_valid:
            logger.warning("LLM returned empty list — falling back to all crops")
            return None

        logger.info(
            f"LLM gate ({provider}): {len(crop_ids)} → {len(approved_valid)} "
            f"crops for {region_name}"
        )
        return approved_valid

    except json.JSONDecodeError as e:
        logger.warning(f"LLM JSON parse error ({provider}): {e}")
        return None
    except Exception as e:
        logger.warning(f"LLM filter failed — fallback active ({provider}): {e}")
        return None
