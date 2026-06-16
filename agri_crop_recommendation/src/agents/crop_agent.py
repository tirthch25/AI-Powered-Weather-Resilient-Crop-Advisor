"""
Crop Recommendation Agent — v4 (100% Dynamic, No Static Fallback Tables)
=========================================================================
Strategy (priority order):

1. In-memory cache (instant) — same location+season+zone returns cached result
2. Gemini with Google Search Grounding — real-time crop advisories + live prices
3. Gemini plain (4-key rotation, model fallback list) — knowledge-based
4. Ollama with web search tool-calling — local LLM + DuckDuckGo
5. Ollama plain — local LLM
6. Gemini simple fallback prompt — stripped-down request for basic recommendations

NO STATIC FALLBACK TABLES: The old _fallback_crops() static tables (600+ lines of
hardcoded crop data) have been removed. All crop data now comes from LLMs.
If every LLM provider fails, the agent returns an empty list instead of
serving fake pre-baked crop data.

Key improvements vs v3:
  - Removed _COUNTRY_CROP_HINTS (60+ countries of static hint text)
  - Removed _fallback_crops() (massive static zone-keyed tables)
  - Removed _HINDI_CROP_NAMES detection (too restrictive, wrong for multilingual outputs)
  - Removed _SH_COUNTRIES / _SOUTH_ASIAN_COUNTRIES / _SOUTH_ASIA hardcoded sets
  - Prompt now instructs LLM to search for actual crops grown (not use hint lists)
  - Added _llm_simple_fallback() — minimal prompt for when full prompt fails
  - Validation simplified: just requires ≥ 3 entries with real crop names
"""

import os
import json
import logging
import re
import time
from typing import Optional, Dict, List

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# All 4 Gemini keys — rotated on 429
GEMINI_KEYS: list = [k for k in [
    os.getenv("GEMINI_API_KEY", ""),
    os.getenv("GEMINI_API_KEY_2", ""),
    os.getenv("GEMINI_API_KEY_3", ""),
    os.getenv("GEMINI_API_KEY_4", ""),
] if k.strip()]
GEMINI_KEY = GEMINI_KEYS[0] if GEMINI_KEYS else ""  # backward compat

# Model fallback list — lite models first to conserve quota
_GEMINI_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite-001",
    "gemini-flash-lite-latest",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
]

# Models that support Google Search Grounding
_SEARCH_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash-001",
]

# ── In-memory result cache (1 hour TTL) ──────────────────────────────────────
_CROP_CACHE: Dict[tuple, tuple] = {}
_CROP_CACHE_TTL = 3600


# ── JSON extraction ───────────────────────────────────────────────────────────

def _extract_json(text: str) -> Optional[list]:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    cleaned = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    match = re.search(r'\[[\s\S]*\]', cleaned)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        except Exception:
            pass
    return None


# ── Crop validation ───────────────────────────────────────────────────────────

def _validate_crops(crops: Optional[list], country: str) -> Optional[list]:
    """
    Validate that LLM returned a real crop list.
    Simplified: only checks that we have ≥ 3 entries with actual crop names.
    The strict Hindi-name geography check was removed — it was too restrictive
    and blocked valid multilingual responses from non-Indian countries.
    """
    if not crops or not isinstance(crops, list):
        return None
    valid = [
        c for c in crops
        if isinstance(c, dict) and c.get("crop_name") and c["crop_name"] != "Unknown Crop"
    ]
    return crops if len(valid) >= 3 else None


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(
    country, state, district, season, climate, planning_days,
    irrigation, temp, humidity, current, forecast, soil, market, summary,
    include_search_context: bool = False,
) -> str:
    """
    Build a detailed, geography-aware crop recommendation prompt.
    No static country hint lists — the LLM uses its own knowledge and search.
    """
    forecast_str = ""
    for f in forecast[:3]:
        forecast_str += (
            f"  {f.get('month', '')}: {f.get('temp_avg', '?')}°C avg, "
            f"{f.get('rainfall_mm', '?')}mm rain\n"
        )

    market_str = (
        ", ".join(f"{k}: {v}" for k, v in list(market.items())[:5])
        if market else "No pre-loaded prices — use current local market rates"
    )

    soil_str = (
        f"{soil.get('type', 'Unknown')} pH {soil.get('ph', '?')}, "
        f"organic matter: {soil.get('organic_matter', 'Unknown')}, "
        f"drainage: {soil.get('drainage', 'Unknown')}"
        if soil.get("type") not in (None, "Unknown") else
        "Soil type to be determined from local extension data"
    )

    search_instruction = (
        f"Search for current crop advisories, pest alerts, and market prices "
        f"in {district}, {state}, {country} before answering. "
        if include_search_context else ""
    )

    # Hemisphere flag (used in season framing, not crop selection)
    hemisphere = (
        "Southern" if lat_is_southern(country) else "Northern"
    )

    temp_str     = f"{temp}°C" if temp is not None else "unknown"
    humidity_str = f"{humidity}%" if humidity is not None else "unknown"

    return (
        f"SYSTEM: You are a senior agricultural expert specialising ONLY in "
        f"{district}, {state}, {country}. "
        f"Your recommendations MUST reflect actual farming practices in {country}. "
        f"Only recommend crops that are genuinely grown in {state}, {country}. "
        f"Include a DIVERSE mix: field crops, vegetables, fruits, and specialty crops. "
        f"Do NOT default to generic crops — research what farmers in {district} actually grow.\n"
        f"{search_instruction}"
        f"Season: {season}, Climate zone: {climate}, Hemisphere: {hemisphere}. "
        f"Irrigation: {irrigation}, Planning horizon: {planning_days} days. "
        f"Current conditions: {temp_str} temperature, humidity {humidity_str}, "
        f"rainfall last 7 days: {current.get('rainfall_7d_mm', 'unknown')} mm. "
        f"Soil: {soil_str}. "
        f"3-month forecast:\n{forecast_str.strip()}\n"
        f"Local market prices: {market_str}. "
        f"District context: {summary[:200] if summary else 'Agricultural region'}.\n\n"
        f"Return EXACTLY 8 crops for {district}, {state}, {country} ({season} season). "
        f"Include at least 2 vegetables, 1 fruit crop, and the rest as field/grain crops. "
        f"Sort by suitability_score descending. "
        f"Return ONLY a valid JSON array — no markdown, no explanation:\n"
        f'[{{"crop_name":"<official {country} crop name>",'
        f'"local_name":"<name in local language of {country}>",'
        f'"suitability_score":<0-100>,'
        f'"season_fit":"<Excellent/Good/Fair>",'
        f'"risk_level":"<Low/Medium/High>",'
        f'"duration_days":<int>,'
        f'"water_need":"<Low/Medium/High>",'
        f'"estimated_yield":"<X-Y tons/ha>",'
        f'"planting_window":"<e.g. Oct 1 - Nov 15 in {state} context>",'
        f'"market_demand":"<High/Medium/Low>",'
        f'"reasons":["<reason specific to {district}, {state}>","<reason 2>"],'
        f'"warnings":["<real risk for {district}, {state}>"],'
        f'"growing_tip":"<tip from {state} agriculture extension>"}},...]\n'
    )


def lat_is_southern(country: str) -> bool:
    """True if country is predominantly in the Southern hemisphere."""
    c = country.lower()
    return any(s in c for s in [
        "australia", "new zealand", "south africa", "brazil", "argentina",
        "chile", "peru", "uruguay", "paraguay", "zambia", "zimbabwe",
        "mozambique", "namibia", "botswana", "madagascar", "bolivia",
    ])


# ── Simple fallback prompt (stripped down for when full prompt fails) ─────────

def _build_simple_prompt(country: str, state: str, district: str, season: str,
                          temp: Optional[float], climate: str) -> str:
    """
    Minimal prompt for when the full prompt fails.
    Returns 5 basic crops with required fields.
    """
    temp_str = f"{temp}°C" if temp is not None else "typical"
    return (
        f"Name 5 crops actually grown by farmers in {district}, {state}, {country} "
        f"during {season} season. Temperature: {temp_str}. Climate: {climate}. "
        f"Include at least 1 vegetable and 1 fruit. "
        f"Return ONLY JSON array (no markdown):\n"
        f'[{{"crop_name":"<crop>","local_name":"<local name>","suitability_score":<0-100>,'
        f'"season_fit":"<Excellent/Good/Fair>","risk_level":"<Low/Medium/High>",'
        f'"duration_days":<int>,"water_need":"<Low/Medium/High>",'
        f'"estimated_yield":"<X-Y tons/ha>","planting_window":"<dates>",'
        f'"market_demand":"<High/Medium/Low>",'
        f'"reasons":["<why this crop suits {district}>"],'
        f'"warnings":[],"growing_tip":"<1 practical tip>"}}]'
    )


# ── Gemini callers ────────────────────────────────────────────────────────────

def _call_gemini_with_search(prompt: str) -> Optional[list]:
    """Gemini + Google Search Grounding for real-time crop data."""
    if not GEMINI_KEYS:
        return None
    try:
        from google import genai as _g
        from google.genai import types as _gt

        for api_key in GEMINI_KEYS:
            client = _g.Client(api_key=api_key)
            for model in _SEARCH_MODELS:
                try:
                    resp = client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=_gt.GenerateContentConfig(
                            tools=[_gt.Tool(google_search=_gt.GoogleSearch())],
                        ),
                    )
                    text   = resp.text.strip() if resp.text else None
                    result = _extract_json(text) if text else None
                    if result:
                        logger.info(
                            "[CropAgent] Gemini search-grounded OK (%s, key ...%s) → %d crops",
                            model, api_key[-6:], len(result)
                        )
                        return result
                except Exception as e:
                    err = str(e)
                    if "429" in err or "RESOURCE_EXHAUSTED" in err:
                        continue
                    if "not supported" in err.lower() or "404" in err or "NOT_FOUND" in err:
                        continue
                    logger.debug("[CropAgent] Search model %s: %s", model, err[:80])
                    continue
    except ImportError:
        pass
    except Exception as e:
        logger.debug("[CropAgent] Gemini search grounding unavailable: %s", e)
    return None


def _call_gemini(prompt: str) -> Optional[list]:
    """Plain Gemini call — 4-key rotation, model fallback."""
    if not GEMINI_KEYS:
        return None

    try:
        from google import genai as _g
        for api_key in GEMINI_KEYS:
            client = _g.Client(api_key=api_key)
            for model in _GEMINI_MODELS:
                try:
                    resp   = client.models.generate_content(model=model, contents=prompt)
                    text   = resp.text.strip() if resp.text else None
                    result = _extract_json(text) if text else None
                    if result:
                        logger.info(
                            "[CropAgent] Gemini OK (%s, key ...%s) → %d crops",
                            model, api_key[-6:], len(result)
                        )
                        return result
                except Exception as e:
                    err = str(e)
                    if "429" in err or "RESOURCE_EXHAUSTED" in err:
                        logger.debug("[CropAgent] %s quota on key ...%s", model, api_key[-6:])
                        continue
                    if "404" in err or "NOT_FOUND" in err:
                        continue
                    logger.debug("[CropAgent] %s error: %s", model, err[:80])
                    continue
    except ImportError:
        pass
    except Exception as e:
        logger.warning("[CropAgent] Gemini failed: %s", e)

    # Legacy SDK fallback
    try:
        import google.generativeai as genai  # type: ignore
        for api_key in GEMINI_KEYS:
            genai.configure(api_key=api_key)
            for model in ["gemini-2.5-flash-lite", "gemini-2.0-flash-lite"]:
                try:
                    resp   = genai.GenerativeModel(model).generate_content(prompt)
                    result = _extract_json(resp.text.strip()) if resp.text else None
                    if result:
                        return result
                except Exception:
                    continue
    except ImportError:
        pass
    return None


# ── Ollama callers ────────────────────────────────────────────────────────────

def _call_ollama_with_search(prompt: str, location: str) -> Optional[list]:
    """Ollama + DuckDuckGo web search."""
    try:
        from src.agents.web_search_agent import call_ollama_with_search
        text = call_ollama_with_search(prompt, location=location, timeout=45)
        return _extract_json(text) if text else None
    except Exception as e:
        logger.debug("[CropAgent] Ollama web search failed: %s", e)
        return None


def _call_ollama(prompt: str) -> Optional[list]:
    """Plain Ollama call (no search)."""
    try:
        import ollama
        client   = ollama.Client(host=OLLAMA_URL)
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": "Return valid JSON array only. No markdown or explanation."},
                {"role": "user",   "content": prompt},
            ],
            options={"temperature": 0.1, "num_ctx": 6144},
        )
        return _extract_json(response["message"]["content"].strip())
    except Exception as e:
        logger.debug("[CropAgent] Ollama failed: %s", e)
        return None


def _llm_simple_fallback(
    country: str, state: str, district: str, season: str,
    temp: Optional[float], climate: str,
) -> Optional[list]:
    """
    Last-chance LLM attempt with a stripped-down prompt.
    Returns crops or None (no static tables used).
    """
    simple_prompt = _build_simple_prompt(country, state, district, season, temp, climate)

    # Try Gemini first (fastest)
    if GEMINI_KEYS:
        result = _call_gemini(simple_prompt)
        if result and len([c for c in result if isinstance(c, dict) and c.get("crop_name")]) >= 3:
            logger.info("[CropAgent] Simple fallback prompt succeeded via Gemini")
            return result

    # Try Ollama
    result = _call_ollama(simple_prompt)
    if result and len([c for c in result if isinstance(c, dict) and c.get("crop_name")]) >= 3:
        logger.info("[CropAgent] Simple fallback prompt succeeded via Ollama")
        return result

    logger.error("[CropAgent] ALL LLM providers exhausted for %s/%s", district, country)
    return None


# ── Main entry point ──────────────────────────────────────────────────────────

def recommend_crops_agent(
    country: str,
    state: str,
    district: str,
    gathered_data: dict,
    irrigation: str = "Limited",
    planning_days: int = 90,
    soil_override: Optional[dict] = None,
) -> list:
    """
    Generate AI crop recommendations for any global location.

    Pipeline (all LLM-driven — no static tables):
      cache → Gemini+Search → Gemini plain → Ollama+Search → Ollama → Gemini simple → []
    """
    current  = gathered_data.get("current", {})
    forecast = gathered_data.get("forecast_6month", [])
    soil     = soil_override or gathered_data.get("soil", {})
    season   = gathered_data.get("season", "")
    climate  = gathered_data.get("climate_zone", "")
    market   = gathered_data.get("market_prices", {})
    summary  = gathered_data.get("district_summary", "")

    temp     = current.get("temperature_c")
    humidity = current.get("humidity_pct")

    # ── 1. Cache ──────────────────────────────────────────────────────────────
    cache_key = (country, state, district, season, climate, irrigation)
    cached_ts, cached_val = _CROP_CACHE.get(cache_key, (0, None))
    if cached_val is not None and (time.time() - cached_ts) < _CROP_CACHE_TTL:
        logger.info("[CropAgent] Cache hit for %s", district)
        return cached_val

    location_str = f"{district}, {state}, {country}"

    # ── 2. Build prompts ──────────────────────────────────────────────────────
    prompt_with_search = _build_prompt(
        country, state, district, season, climate, planning_days,
        irrigation, temp, humidity, current, forecast, soil, market, summary,
        include_search_context=True,
    )
    prompt_plain = _build_prompt(
        country, state, district, season, climate, planning_days,
        irrigation, temp, humidity, current, forecast, soil, market, summary,
        include_search_context=False,
    )

    crops = None

    # ── 3. Gemini + Google Search Grounding ───────────────────────────────────
    if GEMINI_KEYS:
        raw   = _call_gemini_with_search(prompt_with_search)
        crops = _validate_crops(raw, country)
        if crops:
            logger.info("[CropAgent] Using search-grounded Gemini result")

    # ── 4. Gemini plain ───────────────────────────────────────────────────────
    if not crops and GEMINI_KEYS:
        raw   = _call_gemini(prompt_plain)
        crops = _validate_crops(raw, country)
        if crops:
            logger.info("[CropAgent] Using plain Gemini result")

    # ── 5. Ollama with web search ─────────────────────────────────────────────
    if not crops:
        raw   = _call_ollama_with_search(prompt_with_search, location=location_str)
        crops = _validate_crops(raw, country)
        if crops:
            logger.info("[CropAgent] Using Ollama+search result")

    # ── 6. Ollama plain ───────────────────────────────────────────────────────
    if not crops:
        raw   = _call_ollama(prompt_plain)
        crops = _validate_crops(raw, country)
        if crops:
            logger.info("[CropAgent] Using plain Ollama result")

    # ── 7. LLM simple fallback (stripped-down prompt) ─────────────────────────
    if not crops:
        logger.warning("[CropAgent] Main prompts failed — trying simple fallback for %s", district)
        raw   = _llm_simple_fallback(country, state, district, season, temp, climate)
        crops = _validate_crops(raw, country) if raw else None
        if crops:
            logger.info("[CropAgent] Simple fallback returned %d crops", len(crops))

    # ── 8. Absolute last resort — return empty list (NO static tables) ────────
    if not crops or not isinstance(crops, list):
        logger.error(
            "[CropAgent] ALL providers exhausted for %s — returning empty list "
            "(no static fallback data served)", location_str
        )
        return []

    # ── 9. Sanitize & sort ────────────────────────────────────────────────────
    sanitized = []
    for crop in crops[:8]:
        if not isinstance(crop, dict):
            continue
        crop.setdefault("crop_name",         "Unknown Crop")
        crop.setdefault("local_name",        crop["crop_name"])
        crop.setdefault("suitability_score", 60)
        crop.setdefault("season_fit",        "Good")
        crop.setdefault("risk_level",        "Medium")
        crop.setdefault("duration_days",     90)
        crop.setdefault("water_need",        "Medium")
        crop.setdefault("estimated_yield",   "2-4 tons/hectare")
        crop.setdefault("planting_window",   "Current season")
        crop.setdefault("market_demand",     "Medium")
        crop.setdefault("reasons",           ["Suitable for local climate and season"])
        crop.setdefault("warnings",          [])
        crop.setdefault("growing_tip",       "Follow local agricultural extension guidelines.")
        sanitized.append(crop)

    sanitized.sort(key=lambda x: x.get("suitability_score", 0), reverse=True)
    _CROP_CACHE[cache_key] = (time.time(), sanitized)
    return sanitized
