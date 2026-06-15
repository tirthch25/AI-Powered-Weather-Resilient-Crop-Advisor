"""
Crop Recommendation Agent — v3 (Dynamic, Search-Grounded)
==========================================================
Strategy (priority order):
  1. In-memory cache (instant) — same location+season+zone returns cached result
  2. Gemini with Google Search Grounding (real-time advisories, live prices)
  3. Gemini without search (4-key rotation, model fallback list)
  4. Ollama with web search tool-calling (uses DuckDuckGo via tool API)
  5. Ollama plain (no search)
  6. Geography-aware zone fallback (no API) — final fallback

Key fixes vs v2:
  - Gemini called directly (NOT inside ThreadPoolExecutor) — fixes
    "Cannot send a request, as the client has been closed" errors
  - All 4 API keys rotated on 429 quota errors
  - Model fallback list: gemini-2.5-flash-lite → gemini-2.0-flash-lite → ...
  - Prompt is geography-aware: country, hemisphere, agro-climate, local names
  - Fallback crops keyed by climate zone + hemisphere (not India-only Hindi names)
  - Web search grounding for real-time crop advisories and market prices
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

# Model fallback list — same as llm_location_agent
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

# ── In-memory result cache ────────────────────────────────────────────────────
_CROP_CACHE: Dict[tuple, tuple] = {}
_CROP_CACHE_TTL = 3600  # 1 hour


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


# Country-specific crop hint lists — shown to LLM as few-shot context
# so it understands what crops are actually grown in each region
_COUNTRY_CROP_HINTS: Dict[str, str] = {
    # Europe
    "germany":       "Winterweizen (Winter Wheat), Winterraps (Canola/Rapeseed), Zuckerrübe (Sugar Beet), Mais (Maize/Corn), Kartoffel (Potato), Wintergerste (Winter Barley), Roggen (Rye), Triticale, Hafer (Oat), Ackerbohne (Field Bean), Sonnenblume (Sunflower), Rüben (Turnips)",
    "france":        "Blé tendre (Soft Wheat), Colza (Canola), Maïs (Maize), Orge (Barley), Betterave sucrière (Sugar Beet), Tournesol (Sunflower), Pomme de terre (Potato), Pois protéagineux (Protein Pea), Soja, Vin (Grapevine), Lin oléagineux (Linseed)",
    "united kingdom":"Winter Wheat, Oilseed Rape (Canola), Spring Barley, Winter Barley, Potatoes, Sugar Beet, Oats, Field Beans, Winter Oats, Linseed, Peas",
    "poland":        "Żyto (Rye), Pszenica (Wheat), Rzepak (Canola), Ziemniaki (Potato), Buraki cukrowe (Sugar Beet), Kukurydza (Maize), Owies (Oat), Groch (Pea), Słonecznik (Sunflower)",
    "ukraine":       "Пшениця (Wheat), Соняшник (Sunflower), Кукурудза (Maize), Соя (Soybean), Ріпак (Canola), Ячмінь (Barley), Буряк цукровий (Sugar Beet), Просо (Millet)",
    "netherlands":   "Aardappel (Potato), Suikerbiet (Sugar Beet), Tarwe (Wheat), Gerst (Barley), Ui (Onion), Bloembollen (Flower bulbs), Koolzaad (Canola)",
    "italy":         "Grano tenero (Wheat), Mais (Maize), Pomodoro (Tomato), Olive (Olive), Uva (Grape), Riso (Rice), Girasole (Sunflower), Soja (Soybean)",
    "spain":         "Trigo (Wheat), Cebada (Barley), Oliva (Olive), Vid (Grape/Vine), Girasol (Sunflower), Maíz (Maize), Naranja (Orange), Almendro (Almond)",
    "czech":         "Pšenice (Wheat), Ječmen (Barley), Řepka (Canola), Kukuřice (Maize), Cukrová řepa (Sugar Beet), Slunečnice (Sunflower), Brambory (Potato)",
    "romania":       "Grâu (Wheat), Porumb (Maize), Floarea-soarelui (Sunflower), Rapița (Canola), Soia (Soybean), Cartofi (Potato), Sfeclă de zahăr (Sugar Beet)",
    "sweden":        "Höstvete (Winter Wheat), Korn (Barley), Havre (Oat), Höstraps (Winter Canola), Potatis (Potato), Sockerbetor (Sugar Beet), Råg (Rye)",
    "denmark":       "Vinterbyg (Winter Barley), Vinterhvede (Winter Wheat), Vinterraps (Winter Canola), Havre (Oat), Kartofler (Potato), Sukkerroer (Sugar Beet)",
    # North America
    "united states": "Corn, Soybean, Winter Wheat, Cotton, Rice, Canola, Sorghum, Sunflower, Alfalfa, Sugarbeet",
    "canada":        "Spring Wheat, Canola, Barley, Corn, Soybean, Oats, Flaxseed, Durum Wheat, Peas, Lentils",
    # South America
    "brazil":        "Soja (Soybean), Milho (Maize), Cana-de-açúcar (Sugarcane), Café (Coffee), Arroz (Rice), Algodão (Cotton), Feijão (Bean), Mandioca (Cassava), Laranja (Orange)",
    "argentina":     "Soja (Soybean), Maíz (Maize), Trigo (Wheat), Girasol (Sunflower), Cebada (Barley), Algodón (Cotton), Caña de azúcar (Sugarcane)",
    # Asia
    "japan":         "Kome (Rice), Mugi (Wheat/Barley), Daizu (Soybean), Satsumaimo (Sweet Potato), Jagaimo (Potato), Natane (Canola), Tendō (Sugar Beet), Nira, Kyabeji (Cabbage)",
    "south korea":   "Sssal (Rice), Baechu (Cabbage), Gochutgaru (Pepper), Baekkimchi (Radish), Ginseng, Maize, Barley, Garlic",
    "china":         "Xiǎomài (Wheat), Shuǐdào (Rice), Yùmǐ (Maize), Dàdòu (Soybean), Miánhuā (Cotton), Táng liú (Sugarcane), Huāshēng (Peanut), Yóucài (Canola)",
    "thailand":      "Khao (Rice), Mân samphalang (Cassava), Sugarcane, Maize, Rubber, Oil Palm, Mango, Durian",
    "vietnam":       "Lúa (Rice), Cà phê (Coffee), Cao su (Rubber), Mía (Sugarcane), Cacao, Ngô (Maize), Rau (Vegetables)",
    "indonesia":     "Padi (Rice), Kelapa sawit (Oil Palm), Karet (Rubber), Jagung (Maize), Singkong (Cassava), Tebu (Sugarcane), Kopi (Coffee)",
    # South Asia
    "india":         "Chawal/Dhan (Rice), Gehun (Wheat), Makka (Maize), Chana (Chickpea), Sarson (Mustard), Kapas (Cotton), Mungfali (Groundnut), Tur (Pigeon Pea), Soybean, Ganna (Sugarcane)",
    "pakistan":      "Gandum (Wheat), Chawal (Rice), Kapas (Cotton), Aik (Sugarcane), Makkai (Maize), Sarson (Mustard), Mash (Lentil)",
    # Africa
    "nigeria":       "Rice, Maize, Sorghum, Cassava, Yam, Cowpea, Groundnut, Millet, Soybean, Oil Palm, Cotton",
    "ethiopia":      "Teff, Wheat, Maize, Sorghum, Barley, Coffee, Chickpea, Lentil, Enset, Sesame",
    "kenya":         "Maize, Tea, Coffee, Wheat, Rice, Sugarcane, Sorghum, Pyrethrum, French Beans, Horticulture",
    "south africa":  "Maize, Wheat, Sunflower, Sugarcane, Soybean, Sorghum, Groundnut, Cotton, Barley, Tobacco",
    # Oceania
    "australia":     "Wheat, Barley, Canola, Sorghum, Cotton, Rice, Sugarcane, Oats, Lentil, Chickpea, Wool sheep",
    "new zealand":   "Wheat, Barley, Ryegrass, Kiwifruit, Apple, Grapes, Sweetcorn, Onion, Potato, Dairy pasture",
}

def _get_country_hints(country: str) -> str:
    """Return crop hint string for the given country."""
    c_lower = country.lower()
    for key, hints in _COUNTRY_CROP_HINTS.items():
        if key in c_lower:
            return hints
    return ""

# Known Indian/Hindi crop names — if these appear for a non-South-Asian country,
# Gemini has hallucinated the wrong geography
_HINDI_CROP_NAMES = {
    "chawal", "gehun", "makka", "chana", "sarson", "kapas", "mungfali",
    "bajra", "jowar", "tur", "arhar", "moong", "urad", "dhan", "ganna",
    "lahsun", "tamatar", "aloo", "pyaaz", "matar", "bhindi", "palak",
    "shimla mirch", "karela", "lauki", "turai",
}
_SOUTH_ASIAN_COUNTRIES = {"india", "pakistan", "bangladesh", "nepal", "sri lanka", "bhutan"}


def _validate_crops(crops: Optional[list], country: str) -> Optional[list]:
    """
    Validate that Gemini returned geographically correct crops.
    If Gemini returned Hindi crop names for a non-South-Asian country,
    returns None so the fallback table is used instead.
    """
    if not crops or not isinstance(crops, list):
        return None

    c_lower = country.lower()
    is_south_asian = any(sa in c_lower for sa in _SOUTH_ASIAN_COUNTRIES)

    if not is_south_asian:
        # Count how many crops have Hindi local names
        hindi_count = 0
        for crop in crops:
            local = (crop.get("local_name") or "").lower()
            name  = (crop.get("crop_name") or "").lower()
            if any(h in local for h in _HINDI_CROP_NAMES) or any(h in name for h in _HINDI_CROP_NAMES):
                hindi_count += 1
        if hindi_count >= 2:
            logger.warning(
                "[CropAgent] Gemini returned %d Hindi crop names for %s — rejecting, using fallback",
                hindi_count, country,
            )
            return None

    # Basic sanity: need at least 3 crop entries with real names
    valid = [c for c in crops if isinstance(c, dict) and c.get("crop_name") and c.get("crop_name") != "Unknown Crop"]
    if len(valid) < 3:
        return None

    return crops


def _build_prompt(
    country, state, district, season, climate, planning_days,
    irrigation, temp, humidity, current, forecast, soil, market, summary,
    include_search_context: bool = False,
) -> str:
    """Build a rich, geography-aware crop recommendation prompt with country-specific examples."""
    forecast_str = ""
    for f in forecast[:3]:
        forecast_str += (
            f"  {f.get('month', '')}: {f.get('temp_avg', '?')}°C avg, "
            f"{f.get('rainfall_mm', '?')}mm rain\n"
        )

    market_str = (
        ", ".join(f"{k}: {v}" for k, v in list(market.items())[:5])
        if market else "Use current local prices"
    )

    search_instruction = ""
    if include_search_context:
        search_instruction = (
            f"Search for current crop advisories, pest alerts, and market prices "
            f"in {district}, {state}, {country} before answering. "
        )

    # Embed country-specific crop hints directly in the prompt
    crop_hints = _get_country_hints(country)
    hints_str = (
        f"\nIMPORTANT — crops actually grown in {country}: {crop_hints}. "
        f"ONLY recommend crops from this list or close relatives. "
        f"NEVER recommend crops native to other continents.\n"
        if crop_hints else
        f"\nOnly recommend crops genuinely grown in {country}.\n"
    )

    hemisphere = (
        "Southern"
        if any(c in country.lower() for c in [
            "australia", "brazil", "argentina", "south africa",
            "new zealand", "chile", "peru", "uruguay", "zambia", "zimbabwe"
        ]) else "Northern"
    )

    return (
        f"SYSTEM: You are a senior agricultural expert ONLY for {district}, {state}, {country}. "
        f"Your recommendations MUST reflect actual farming practices in {country}. "
        f"{hints_str}"
        f"{search_instruction}"
        f"Season: {season}, Climate zone: {climate}, Hemisphere: {hemisphere}. "
        f"Irrigation: {irrigation}, Planning horizon: {planning_days} days. "
        f"Current conditions: {temp}°C, humidity {humidity}%, "
        f"rainfall last 7 days: {current.get('rainfall_7d_mm', '?')} mm. "
        f"Soil: {soil.get('type', 'Loam')} pH {soil.get('ph', 7.0)}, "
        f"organic matter: {soil.get('organic_matter', 'Medium')}, "
        f"drainage: {soil.get('drainage', 'Medium')}. "
        f"3-month forecast:\n{forecast_str.strip()}\n"
        f"Local market prices: {market_str}. "
        f"District context: {summary[:150] if summary else 'Standard agricultural region'}.\n\n"
        f"Return EXACTLY 6 crops for {country} ({season} season). "
        f"Sort by suitability_score descending. "
        f"Return ONLY a valid JSON array — no markdown, no explanation:\n"
        f'[{{"crop_name":"<official {country} crop name>","local_name":"<name in {country}\'s local language, e.g., German/French/etc.>","suitability_score":<0-100>,'
        f'"season_fit":"<Excellent/Good/Fair>","risk_level":"<Low/Medium/High>",'
        f'"duration_days":<int>,"water_need":"<Low/Medium/High>","estimated_yield":"<X-Y tons/ha>",'
        f'"planting_window":"<e.g. Oct 1 - Nov 15, in {country} context>","market_demand":"<High/Medium/Low>",'
        f'"reasons":["<reason specific to {state}, {country}>","<reason 2>"],'
        f'"warnings":["<real risk for {district}, {state}>"],'
        f'"growing_tip":"<tip from {country} agriculture extension practices>"}},...]\n'
    )


# ── Gemini callers ────────────────────────────────────────────────────────────

def _call_gemini_with_search(prompt: str) -> Optional[list]:
    """
    Call Gemini with Google Search Grounding for real-time crop advisories.
    Returns parsed crop list or None if search grounding not available.
    """
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
                    text = resp.text.strip() if resp.text else None
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
    """
    Call Gemini directly (no ThreadPoolExecutor).
    Rotates through all 4 keys and all models on 429 quota errors.
    """
    if not GEMINI_KEYS:
        return None

    # Try new google.genai SDK
    try:
        from google import genai as _g
        for api_key in GEMINI_KEYS:
            client = _g.Client(api_key=api_key)
            for model in _GEMINI_MODELS:
                try:
                    resp = client.models.generate_content(model=model, contents=prompt)
                    text = resp.text.strip() if resp.text else None
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

    # Fallback: legacy SDK
    try:
        import google.generativeai as genai  # type: ignore
        for api_key in GEMINI_KEYS:
            genai.configure(api_key=api_key)
            for model in ["gemini-2.5-flash-lite", "gemini-2.0-flash-lite"]:
                try:
                    resp = genai.GenerativeModel(model).generate_content(prompt)
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
    """Call Ollama with web search tool-calling. Returns crop list or None."""
    try:
        from src.agents.web_search_agent import call_ollama_with_search
        text = call_ollama_with_search(prompt, location=location, timeout=45)
        return _extract_json(text) if text else None
    except Exception as e:
        logger.debug("[CropAgent] Ollama web search failed: %s", e)
        return None


def _call_ollama(prompt: str) -> Optional[list]:
    """Plain Ollama call (no search tools)."""
    try:
        import ollama
        client = ollama.Client(host=OLLAMA_URL)
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

    Pipeline:
      cache → Gemini+Search → Gemini plain → Ollama+Search → Ollama → zone fallback
    """
    current  = gathered_data.get("current", {})
    forecast = gathered_data.get("forecast_6month", [])
    soil     = soil_override or gathered_data.get("soil", {})
    season   = gathered_data.get("season", "")
    climate  = gathered_data.get("climate_zone", "")
    market   = gathered_data.get("market_prices", {})
    summary  = gathered_data.get("district_summary", "")

    temp     = current.get("temperature_c") or 25
    humidity = current.get("humidity_pct")  or 65

    # ── 1. Cache ──────────────────────────────────────────────────────────────
    cache_key = (country, state, district, season, climate, irrigation)
    cached_ts, cached_val = _CROP_CACHE.get(cache_key, (0, None))
    if cached_val is not None and (time.time() - cached_ts) < _CROP_CACHE_TTL:
        logger.info("[CropAgent] Cache hit for %s", district)
        return cached_val

    location_str = f"{district}, {state}, {country}"

    # ── 2. Build prompt (with search context hint) ────────────────────────────
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
        raw = _call_gemini_with_search(prompt_with_search)
        crops = _validate_crops(raw, country)
        if crops:
            logger.info("[CropAgent] Using search-grounded Gemini result")

    # ── 4. Gemini plain (key rotation, model fallback) ────────────────────────
    if not crops and GEMINI_KEYS:
        raw = _call_gemini(prompt_plain)
        crops = _validate_crops(raw, country)
        if crops:
            logger.info("[CropAgent] Using plain Gemini result")

    # ── 5. Ollama with web search tool-calling ────────────────────────────────
    if not crops:
        raw = _call_ollama_with_search(prompt_with_search, location=location_str)
        crops = _validate_crops(raw, country)
        if crops:
            logger.info("[CropAgent] Using Ollama+search result")

    # ── 6. Ollama plain ───────────────────────────────────────────────────────
    if not crops:
        raw = _call_ollama(prompt_plain)
        crops = _validate_crops(raw, country)
        if crops:
            logger.info("[CropAgent] Using plain Ollama result")

    # ── 7. Geography-aware zone fallback ──────────────────────────────────────
    if not crops or not isinstance(crops, list):
        logger.warning("[CropAgent] All LLMs failed — using geography fallback for %s", country)
        crops = _fallback_crops(season, climate, country)


    # ── 8. Sanitize & sort ────────────────────────────────────────────────────
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


# ── Geography-aware fallback crop tables ─────────────────────────────────────
# Keyed by (climate_zone, hemisphere) — never India-only names globally

def _fallback_crops(season: str, climate: str, country: str = "") -> list:
    """
    Return geographically appropriate fallback crops instantly without any API call.
    Climate-zone and hemisphere aware — correct crops for Europe, Americas, Africa,
    Asia, Oceania etc.
    """
    c_lower = country.lower()

    # Southern hemisphere seasons are flipped
    _SH_COUNTRIES = {
        "australia", "new zealand", "south africa", "brazil", "argentina",
        "chile", "peru", "bolivia", "uruguay", "paraguay", "zambia",
        "zimbabwe", "mozambique", "namibia", "botswana", "tanzania",
        "kenya", "ethiopia", "madagascar",
    }
    is_southern = any(sc in c_lower for sc in _SH_COUNTRIES)

    # India / South Asia — special seasons
    _SOUTH_ASIA = {"india", "pakistan", "bangladesh", "nepal", "sri lanka", "bhutan"}
    is_south_asia = any(sa in c_lower for sa in _SOUTH_ASIA)

    clim = climate.lower() if climate else ""

    # ── Tropical ─────────────────────────────────────────────────────────────
    if "tropical" in clim:
        return [
            {"crop_name": "Rice",       "local_name": "Rice",       "suitability_score": 90, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 120, "water_need": "High",   "estimated_yield": "3-5 tons/ha",   "planting_window": "Main wet season",   "market_demand": "High",   "reasons": ["Staple crop for tropical climates", "High market demand"], "warnings": [], "growing_tip": "Ensure consistent flooding for paddy rice varieties."},
            {"crop_name": "Maize",      "local_name": "Corn/Maize", "suitability_score": 85, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 90,  "water_need": "Medium", "estimated_yield": "4-6 tons/ha",   "planting_window": "Start of rains",    "market_demand": "High",   "reasons": ["Versatile staple", "Suited for tropical soils"], "warnings": [], "growing_tip": "Apply nitrogen fertilizer at knee-high stage."},
            {"crop_name": "Cassava",    "local_name": "Cassava",    "suitability_score": 83, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 270, "water_need": "Low",    "estimated_yield": "15-25 tons/ha", "planting_window": "Year-round",        "market_demand": "High",   "reasons": ["Drought tolerant", "Major tropical staple"], "warnings": [], "growing_tip": "Plant stakes 2-3 months old for best yield."},
            {"crop_name": "Sweet Potato","local_name": "Sweet Potato","suitability_score": 80,"season_fit": "Good",     "risk_level": "Low",    "duration_days": 90,  "water_need": "Medium", "estimated_yield": "10-20 tons/ha", "planting_window": "Any season",        "market_demand": "High",   "reasons": ["Fast growing", "High nutrition value"], "warnings": [], "growing_tip": "Use well-drained soils to prevent rot."},
            {"crop_name": "Plantain",   "local_name": "Plantain",   "suitability_score": 78, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 300, "water_need": "High",   "estimated_yield": "20-30 tons/ha", "planting_window": "Year-round",        "market_demand": "High",   "reasons": ["Perennial income", "Strong local demand"], "warnings": [], "growing_tip": "Ensure good drainage, plant on ridges."},
            {"crop_name": "Groundnut",  "local_name": "Peanut",     "suitability_score": 74, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 100, "water_need": "Low",    "estimated_yield": "1.5-2.5 tons/ha","planting_window": "Early rains",      "market_demand": "High",   "reasons": ["Oil crop", "Nitrogen fixer", "Drought tolerant"], "warnings": [], "growing_tip": "Sandy loam soils give best pod fill."},
        ]

    # ── Arid / Semi-Arid ─────────────────────────────────────────────────────
    if "arid" in clim or "semi-arid" in clim:
        return [
            {"crop_name": "Sorghum",    "local_name": "Sorghum",    "suitability_score": 90, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 100, "water_need": "Low",    "estimated_yield": "2-4 tons/ha",   "planting_window": "Rainy season",      "market_demand": "High",   "reasons": ["Most drought-tolerant grain", "Excellent for arid zones"], "warnings": [], "growing_tip": "Plant at the start of rains for best establishment."},
            {"crop_name": "Millet",     "local_name": "Millet",     "suitability_score": 88, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 75,  "water_need": "Low",    "estimated_yield": "1-2.5 tons/ha", "planting_window": "Early rains",       "market_demand": "High",   "reasons": ["Extremely drought tolerant", "Short season"], "warnings": [], "growing_tip": "Thin to 15cm spacing after emergence."},
            {"crop_name": "Sesame",     "local_name": "Sesame",     "suitability_score": 82, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 80,  "water_need": "Low",    "estimated_yield": "0.5-1 ton/ha",  "planting_window": "Dry season start",  "market_demand": "High",   "reasons": ["High oil content", "Heat tolerant", "Export value"], "warnings": [], "growing_tip": "Avoid waterlogging — very sensitive to excess water."},
            {"crop_name": "Cowpea",     "local_name": "Cowpea",     "suitability_score": 80, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 65,  "water_need": "Low",    "estimated_yield": "0.8-1.5 tons/ha","planting_window": "Any rains",        "market_demand": "High",   "reasons": ["Nitrogen fixer", "Protein crop", "Drought tolerant"], "warnings": [], "growing_tip": "Intercrop with sorghum or millet for best results."},
            {"crop_name": "Dates",      "local_name": "Dates",      "suitability_score": 75, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 365, "water_need": "Low",    "estimated_yield": "5-10 tons/ha",  "planting_window": "Year-round",        "market_demand": "High",   "reasons": ["Perennial cash crop", "Extremely heat and drought tolerant"], "warnings": [], "growing_tip": "Hand-pollinate for reliable yields."},
            {"crop_name": "Cotton",     "local_name": "Cotton",     "suitability_score": 70, "season_fit": "Good",      "risk_level": "Medium", "duration_days": 160, "water_need": "Medium", "estimated_yield": "1.5-2.5 tons/ha","planting_window": "Rainy season",     "market_demand": "High",   "reasons": ["Major cash crop for arid regions", "High export value"], "warnings": ["Requires pest monitoring"], "growing_tip": "Drip irrigation significantly improves yield."},
        ]

    # ── Mediterranean ─────────────────────────────────────────────────────────
    if "mediterranean" in clim:
        return [
            {"crop_name": "Wheat",      "local_name": "Wheat",      "suitability_score": 92, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 150, "water_need": "Low",    "estimated_yield": "3-5 tons/ha",   "planting_window": "Oct - Dec",         "market_demand": "High",   "reasons": ["Ideal cool-wet winter crop", "Highest suitability for Mediterranean"], "warnings": [], "growing_tip": "Sow after first autumn rains for best germination."},
            {"crop_name": "Olive",      "local_name": "Olive",      "suitability_score": 90, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 365, "water_need": "Low",    "estimated_yield": "2-5 tons/ha",   "planting_window": "Year-round",        "market_demand": "High",   "reasons": ["Signature Mediterranean crop", "Drought tolerant perennial"], "warnings": [], "growing_tip": "Prune for open canopy to improve light penetration."},
            {"crop_name": "Grape",      "local_name": "Grape",      "suitability_score": 87, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 180, "water_need": "Low",    "estimated_yield": "5-15 tons/ha",  "planting_window": "Mar - Apr",         "market_demand": "High",   "reasons": ["Excellent for Mediterranean climate", "High value crop"], "warnings": [], "growing_tip": "Train on trellis, prune to 2-3 buds in winter."},
            {"crop_name": "Tomato",     "local_name": "Tomato",     "suitability_score": 83, "season_fit": "Good",      "risk_level": "Medium", "duration_days": 90,  "water_need": "Medium", "estimated_yield": "40-60 tons/ha", "planting_window": "Apr - May",         "market_demand": "High",   "reasons": ["High value summer vegetable", "Suits Mediterranean summers"], "warnings": ["Monitor for late blight"], "growing_tip": "Drip irrigate to reduce disease pressure."},
            {"crop_name": "Sunflower",  "local_name": "Sunflower",  "suitability_score": 80, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 100, "water_need": "Low",    "estimated_yield": "2-3 tons/ha",   "planting_window": "Apr - May",         "market_demand": "Medium", "reasons": ["Drought tolerant oil crop", "Suits hot dry summers"], "warnings": [], "growing_tip": "Plant in deep, well-drained soils for best root development."},
            {"crop_name": "Barley",     "local_name": "Barley",     "suitability_score": 76, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 120, "water_need": "Low",    "estimated_yield": "2-4 tons/ha",   "planting_window": "Oct - Nov",         "market_demand": "Medium", "reasons": ["Winter cereal", "Drought tolerant", "Multiple uses"], "warnings": [], "growing_tip": "Spring barley also viable with irrigation."},
        ]

    # ── Temperate / Continental (Europe, North America, Northern Asia) ─────────
    if "temperate" in clim or "continental" in clim:
        # European-specific
        if any(eu in c_lower for eu in ["germany", "france", "poland", "ukraine", "netherlands",
                                         "belgium", "czech", "austria", "hungary", "romania",
                                         "sweden", "denmark", "finland", "norway", "switzerland"]):
            season_map = {
                "Spring": [
                    {"crop_name": "Sugar Beet",   "local_name": "Zuckerrübe",    "suitability_score": 90, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 170, "water_need": "Medium", "estimated_yield": "50-80 tons/ha",  "planting_window": "Mar 15 - Apr 30",   "market_demand": "High",   "reasons": ["Major European cash crop", "Long growing season suits climate"], "warnings": [], "growing_tip": "Precision sow at 5-8cm depth."},
                    {"crop_name": "Spring Barley", "local_name": "Sommergerste",  "suitability_score": 88, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 90,  "water_need": "Low",    "estimated_yield": "4-6 tons/ha",    "planting_window": "Mar 1 - Apr 15",    "market_demand": "High",   "reasons": ["Malting barley premium", "Suits cool spring"], "warnings": [], "growing_tip": "Choose two-row malting varieties for premium prices."},
                    {"crop_name": "Potato",        "local_name": "Kartoffel",     "suitability_score": 85, "season_fit": "Excellent", "risk_level": "Medium", "duration_days": 100, "water_need": "Medium", "estimated_yield": "30-45 tons/ha",  "planting_window": "Apr 1 - May 15",    "market_demand": "High",   "reasons": ["Core European staple", "Excellent cool-season crop"], "warnings": ["Late blight risk — scout regularly"], "growing_tip": "Mound plants as they grow; harvest when tops die back."},
                    {"crop_name": "Spring Wheat",  "local_name": "Sommerweizen",  "suitability_score": 82, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 100, "water_need": "Medium", "estimated_yield": "4-6 tons/ha",    "planting_window": "Mar 1 - Apr 1",     "market_demand": "High",   "reasons": ["Bread wheat", "Strong European demand"], "warnings": [], "growing_tip": "Apply split nitrogen for best quality protein."},
                    {"crop_name": "Canola",        "local_name": "Raps",          "suitability_score": 78, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 90,  "water_need": "Medium", "estimated_yield": "3-4.5 tons/ha",  "planting_window": "Mar 20 - Apr 20",   "market_demand": "High",   "reasons": ["Oil crop", "EU biodiesel market", "Good rotation crop"], "warnings": [], "growing_tip": "Scout for flea beetles and cabbage stem flea beetle early."},
                    {"crop_name": "Pea",           "local_name": "Erbse",         "suitability_score": 75, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 80,  "water_need": "Medium", "estimated_yield": "3-5 tons/ha",    "planting_window": "Mar 15 - Apr 15",   "market_demand": "Medium", "reasons": ["Nitrogen fixer", "Cool season legume", "Good rotation"], "warnings": [], "growing_tip": "Inoculate with Rhizobium; avoid waterlogging."},
                ],
                "Autumn": [
                    {"crop_name": "Winter Wheat",  "local_name": "Winterweizen",  "suitability_score": 95, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 250, "water_need": "Medium", "estimated_yield": "6-9 tons/ha",    "planting_window": "Sep 15 - Oct 31",   "market_demand": "High",   "reasons": ["Europe's #1 arable crop", "High yield potential", "Stable prices"], "warnings": [], "growing_tip": "Choose fusarium-resistant varieties; apply fungicide at ear emergence."},
                    {"crop_name": "Winter Rape",   "local_name": "Winterraps",    "suitability_score": 90, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 270, "water_need": "Medium", "estimated_yield": "3-5 tons/ha",    "planting_window": "Aug 15 - Sep 15",   "market_demand": "High",   "reasons": ["Oil crop, EU mandate", "Winter hardy varieties available"], "warnings": ["Watch for slugs at establishment"], "growing_tip": "Target 30-40 plants/m² at harvest — don't over-sow."},
                    {"crop_name": "Winter Barley", "local_name": "Wintergerste",  "suitability_score": 85, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 240, "water_need": "Low",    "estimated_yield": "5-7 tons/ha",    "planting_window": "Sep 20 - Oct 15",   "market_demand": "High",   "reasons": ["Early harvest", "Good for maltsters"], "warnings": [], "growing_tip": "Earliest cereal harvest — frees up land for catch crops."},
                    {"crop_name": "Sugar Beet",    "local_name": "Zuckerrübe",    "suitability_score": 82, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 180, "water_need": "Medium", "estimated_yield": "50-75 tons/ha",  "planting_window": "Mar 15 - Apr 30",   "market_demand": "High",   "reasons": ["High-value root crop", "Major EU crop"], "warnings": [], "growing_tip": "Leave in field until Nov-Dec for highest sugar content."},
                    {"crop_name": "Winter Rye",    "local_name": "Winterroggen",  "suitability_score": 78, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 260, "water_need": "Low",    "estimated_yield": "4-6 tons/ha",    "planting_window": "Sep 15 - Oct 20",   "market_demand": "Medium", "reasons": ["Most cold-hardy cereal", "Suits sandy soils"], "warnings": [], "growing_tip": "Excellent on lighter, sandier soils where wheat struggles."},
                    {"crop_name": "Triticale",     "local_name": "Triticale",     "suitability_score": 74, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 250, "water_need": "Low",    "estimated_yield": "5-7 tons/ha",    "planting_window": "Oct 1 - Oct 30",    "market_demand": "Medium", "reasons": ["Wheat × rye hybrid", "High biomass", "Livestock feed"], "warnings": [], "growing_tip": "Good alternative on marginal soils."},
                ],
                "Summer": [
                    {"crop_name": "Maize",         "local_name": "Mais",          "suitability_score": 88, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 120, "water_need": "Medium", "estimated_yield": "8-12 tons/ha",   "planting_window": "Apr 20 - May 20",   "market_demand": "High",   "reasons": ["Major European feed crop", "High yield"], "warnings": [], "growing_tip": "Soil temperature must exceed 8°C before sowing."},
                    {"crop_name": "Sunflower",     "local_name": "Sonnenblume",   "suitability_score": 82, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 120, "water_need": "Medium", "estimated_yield": "3-4 tons/ha",    "planting_window": "Apr 15 - May 15",   "market_demand": "High",   "reasons": ["Oil crop", "Heat and drought tolerant"], "warnings": [], "growing_tip": "Plant 6-7 seeds/m², thin to 4-5 plants."},
                    {"crop_name": "Potato",        "local_name": "Kartoffel",     "suitability_score": 80, "season_fit": "Good",      "risk_level": "Medium", "duration_days": 100, "water_need": "Medium", "estimated_yield": "30-45 tons/ha",  "planting_window": "Apr 1 - May 1",     "market_demand": "High",   "reasons": ["European staple", "Multiple markets"], "warnings": ["Late blight: apply preventive fungicides"], "growing_tip": "Irrigate during tuber bulking for consistent yields."},
                    {"crop_name": "Field Bean",    "local_name": "Ackerbohne",    "suitability_score": 76, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 110, "water_need": "Medium", "estimated_yield": "3-5 tons/ha",    "planting_window": "Mar 15 - Apr 15",   "market_demand": "High",   "reasons": ["Protein crop", "Nitrogen fixer", "EU protein strategy"], "warnings": [], "growing_tip": "Excellent rotation break from cereals."},
                    {"crop_name": "Hemp",          "local_name": "Hanf",          "suitability_score": 72, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 100, "water_need": "Medium", "estimated_yield": "6-10 tons/ha",   "planting_window": "May 1 - Jun 1",     "market_demand": "High",   "reasons": ["Growing EU industrial hemp market", "Low pesticide need"], "warnings": ["Requires EU license to grow"], "growing_tip": "Sow at 25-35 kg/ha for fiber production."},
                    {"crop_name": "Oat",           "local_name": "Hafer",         "suitability_score": 70, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 100, "water_need": "Medium", "estimated_yield": "4-6 tons/ha",    "planting_window": "Mar 1 - Apr 15",    "market_demand": "Medium", "reasons": ["Health food market growing", "Suits wetter soils"], "warnings": [], "growing_tip": "Tolerates wetter soils better than wheat."},
                ],
                "Winter": [
                    # Same as Autumn for Europe (winter-sown crops)
                    {"crop_name": "Winter Wheat",  "local_name": "Winterweizen",  "suitability_score": 95, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 250, "water_need": "Medium", "estimated_yield": "6-9 tons/ha",    "planting_window": "Sep - Oct",         "market_demand": "High",   "reasons": ["Core European crop"], "warnings": [], "growing_tip": "Choose varieties with high Hagberg falling number for bread-making."},
                    {"crop_name": "Winter Rape",   "local_name": "Winterraps",    "suitability_score": 88, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 270, "water_need": "Medium", "estimated_yield": "3-5 tons/ha",    "planting_window": "Aug - Sep",         "market_demand": "High",   "reasons": ["Biodiesel and food oil demand"], "warnings": [], "growing_tip": "Establish a good leaf canopy before winter."},
                    {"crop_name": "Winter Barley", "local_name": "Wintergerste",  "suitability_score": 84, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 240, "water_need": "Low",    "estimated_yield": "5-7 tons/ha",    "planting_window": "Sep - Oct",         "market_demand": "High",   "reasons": ["Early harvest, frees up field"], "warnings": [], "growing_tip": "Use two-row varieties for malting premium."},
                    {"crop_name": "Spinach",       "local_name": "Spinat",        "suitability_score": 78, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 45,  "water_need": "Medium", "estimated_yield": "15-25 tons/ha",  "planting_window": "Aug - Sep",         "market_demand": "High",   "reasons": ["Winter salad market", "Fast crop"], "warnings": [], "growing_tip": "Harvest before bolting."},
                    {"crop_name": "Field Bean",    "local_name": "Ackerbohne",    "suitability_score": 74, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 180, "water_need": "Medium", "estimated_yield": "3-5 tons/ha",    "planting_window": "Oct - Nov",         "market_demand": "High",   "reasons": ["Overwinter variety available", "Nitrogen fixer"], "warnings": [], "growing_tip": "Winter varieties can establish Oct-Nov."},
                    {"crop_name": "Winter Rye",    "local_name": "Winterroggen",  "suitability_score": 70, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 260, "water_need": "Low",    "estimated_yield": "4-6 tons/ha",    "planting_window": "Sep - Oct",         "market_demand": "Medium", "reasons": ["Hardiest cereal", "Low input"], "warnings": [], "growing_tip": "Best on sandy, acid soils."},
                ],
            }
            # Find best season match
            return (
                season_map.get(season) or
                season_map.get("Autumn") or
                season_map["Winter"]
            )

        # North America
        if any(na in c_lower for na in ["united states", "canada"]):
            return [
                {"crop_name": "Corn",       "local_name": "Corn",       "suitability_score": 90, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 110, "water_need": "Medium", "estimated_yield": "10-14 tons/ha",  "planting_window": "May 1 - Jun 1",     "market_demand": "High",   "reasons": ["#1 US crop", "Strong export demand", "Multiple uses"], "warnings": [], "growing_tip": "Plant at 76cm row spacing for highest yield."},
                {"crop_name": "Soybean",    "local_name": "Soybean",    "suitability_score": 88, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 100, "water_need": "Medium", "estimated_yield": "3-4 tons/ha",    "planting_window": "May 1 - Jun 15",    "market_demand": "High",   "reasons": ["Strong export market", "Nitrogen fixer"], "warnings": [], "growing_tip": "Inoculate with Bradyrhizobium; plant after corn frost risk passes."},
                {"crop_name": "Winter Wheat","local_name": "Wheat",     "suitability_score": 84, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 240, "water_need": "Low",    "estimated_yield": "4-6 tons/ha",    "planting_window": "Sep 15 - Oct 31",   "market_demand": "High",   "reasons": ["Plains staple crop", "Government price support"], "warnings": [], "growing_tip": "Plant Hessian-fly-safe varieties after the fly-free date."},
                {"crop_name": "Canola",     "local_name": "Canola",     "suitability_score": 80, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 95,  "water_need": "Medium", "estimated_yield": "2-3 tons/ha",    "planting_window": "Apr 15 - May 15",   "market_demand": "High",   "reasons": ["Prairie oil crop", "High value", "Good rotation"], "warnings": [], "growing_tip": "Canola Club practices: 3-4 year rotation essential."},
                {"crop_name": "Cotton",     "local_name": "Cotton",     "suitability_score": 76, "season_fit": "Good",      "risk_level": "Medium", "duration_days": 155, "water_need": "Medium", "estimated_yield": "1.2-2 tons/ha",  "planting_window": "Apr 15 - May 30",   "market_demand": "High",   "reasons": ["Southern US staple", "Strong export market"], "warnings": ["Boll weevil monitoring required"], "growing_tip": "Defoliate before harvest for cleaner cotton."},
                {"crop_name": "Sunflower",  "local_name": "Sunflower",  "suitability_score": 72, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 100, "water_need": "Low",    "estimated_yield": "2-3 tons/ha",    "planting_window": "May 15 - Jun 15",   "market_demand": "Medium", "reasons": ["Plains oil crop", "Drought tolerant"], "warnings": [], "growing_tip": "Leave 2-3 weeks after frost-safe dates."},
            ]

        # General temperate fallback
        return [
            {"crop_name": "Wheat",      "local_name": "Wheat",      "suitability_score": 90, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 200, "water_need": "Medium", "estimated_yield": "4-6 tons/ha",   "planting_window": "Sep - Nov",         "market_demand": "High",   "reasons": ["Global staple crop", "Suits temperate climate"], "warnings": [], "growing_tip": "Choose disease-resistant varieties for your region."},
            {"crop_name": "Barley",     "local_name": "Barley",     "suitability_score": 85, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 100, "water_need": "Low",    "estimated_yield": "4-6 tons/ha",   "planting_window": "Sep - Oct / Mar - Apr", "market_demand": "High",   "reasons": ["Drought tolerant cereal", "Malting premium available"], "warnings": [], "growing_tip": "Two-row varieties preferred for malting."},
            {"crop_name": "Canola",     "local_name": "Rapeseed",   "suitability_score": 82, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 200, "water_need": "Medium", "estimated_yield": "3-4 tons/ha",   "planting_window": "Aug - Sep",         "market_demand": "High",   "reasons": ["Oil crop", "Strong global demand"], "warnings": [], "growing_tip": "Establish before winter for best spring growth."},
            {"crop_name": "Potato",     "local_name": "Potato",     "suitability_score": 80, "season_fit": "Good",      "risk_level": "Medium", "duration_days": 100, "water_need": "Medium", "estimated_yield": "25-40 tons/ha", "planting_window": "Mar - May",         "market_demand": "High",   "reasons": ["Global staple", "High yield per hectare"], "warnings": ["Late blight"], "growing_tip": "Use certified seed for disease-free crop."},
            {"crop_name": "Maize",      "local_name": "Maize/Corn", "suitability_score": 76, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 120, "water_need": "Medium", "estimated_yield": "8-12 tons/ha",  "planting_window": "May - Jun",         "market_demand": "High",   "reasons": ["High yield", "Multiple uses — feed, food, biogas"], "warnings": [], "growing_tip": "Wait for soil to reach 8°C before planting."},
            {"crop_name": "Oat",        "local_name": "Oat",        "suitability_score": 72, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 100, "water_need": "Medium", "estimated_yield": "4-6 tons/ha",   "planting_window": "Mar - Apr",         "market_demand": "Medium", "reasons": ["Growing health food demand", "Suits wetter soils"], "warnings": [], "growing_tip": "Sow early for best yields."},
        ]

    # ── South Asian (India, Pakistan, Bangladesh) ─────────────────────────────
    if is_south_asia:
        return [
            {"crop_name": "Rice",       "local_name": "Dhan",       "suitability_score": 90, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 120, "water_need": "High",   "estimated_yield": "4-6 tons/ha",   "planting_window": "Jun 15 - Jul 15", "market_demand": "High",   "reasons": ["Primary Kharif crop", "Thrives in monsoon rainfall"], "warnings": [], "growing_tip": "Transplant seedlings when soil is saturated."},
            {"crop_name": "Wheat",      "local_name": "Gehun",      "suitability_score": 88, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 120, "water_need": "Medium", "estimated_yield": "3-5 tons/ha",   "planting_window": "Nov 1 - Dec 1",   "market_demand": "High",   "reasons": ["Primary Rabi crop", "MSP support", "Cooler conditions"], "warnings": [], "growing_tip": "Irrigate at crown root initiation stage."},
            {"crop_name": "Maize",      "local_name": "Makka",      "suitability_score": 85, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 90,  "water_need": "Medium", "estimated_yield": "4-6 tons/ha",   "planting_window": "Jun 1 - Jul 1",   "market_demand": "High",   "reasons": ["Fast growing", "Good market prices"], "warnings": [], "growing_tip": "Apply nitrogen at knee-high stage."},
            {"crop_name": "Chickpea",   "local_name": "Chana",      "suitability_score": 82, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 100, "water_need": "Low",    "estimated_yield": "1-2 tons/ha",   "planting_window": "Oct 20 - Nov 10", "market_demand": "High",   "reasons": ["Nitrogen fixing legume", "Rabi crop"], "warnings": [], "growing_tip": "Avoid irrigation after flowering."},
            {"crop_name": "Cotton",     "local_name": "Kapas",      "suitability_score": 78, "season_fit": "Good",      "risk_level": "Medium", "duration_days": 180, "water_need": "Medium", "estimated_yield": "2-3 tons/ha",   "planting_window": "May 15 - Jun 15", "market_demand": "High",   "reasons": ["Cash crop", "Warm conditions ideal"], "warnings": ["Monitor for bollworm"], "growing_tip": "Use Bt cotton varieties."},
            {"crop_name": "Mustard",    "local_name": "Sarson",     "suitability_score": 75, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 110, "water_need": "Low",    "estimated_yield": "1.5-2.5 tons/ha","planting_window": "Oct 15 - Nov 15", "market_demand": "High",   "reasons": ["Rabi oil crop", "Short duration", "Low water"], "warnings": [], "growing_tip": "Sow in sandy loam for best yield."},
        ]

    # ── Default: use season as key ────────────────────────────────────────────
    _SEASON_DEFAULTS = {
        "Spring": [
            {"crop_name": "Pea",        "local_name": "Pea",        "suitability_score": 85, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 70,  "water_need": "Medium", "estimated_yield": "5-8 tons/ha",   "planting_window": "Mar - Apr",         "market_demand": "High",   "reasons": ["Cool season legume", "Spring ideal"], "warnings": [], "growing_tip": "Plant as soon as soil can be worked."},
            {"crop_name": "Lettuce",    "local_name": "Lettuce",    "suitability_score": 82, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 45,  "water_need": "Medium", "estimated_yield": "20-30 tons/ha", "planting_window": "Mar - Apr",         "market_demand": "High",   "reasons": ["Fast growing", "High value", "Cool weather"], "warnings": ["Bolt in heat"], "growing_tip": "Harvest before temperature exceeds 25°C."},
            {"crop_name": "Carrot",     "local_name": "Carrot",     "suitability_score": 78, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 80,  "water_need": "Medium", "estimated_yield": "30-40 tons/ha", "planting_window": "Mar - May",         "market_demand": "High",   "reasons": ["Cool season crop", "High nutritional value"], "warnings": [], "growing_tip": "Loose, deep soil gives best root formation."},
            {"crop_name": "Onion",      "local_name": "Onion",      "suitability_score": 75, "season_fit": "Good",      "risk_level": "Medium", "duration_days": 100, "water_need": "Medium", "estimated_yield": "20-30 tons/ha", "planting_window": "Mar - Apr",         "market_demand": "High",   "reasons": ["High demand", "Long shelf life"], "warnings": ["Monitor for thrips"], "growing_tip": "Stop irrigation 2 weeks before harvest."},
            {"crop_name": "Spinach",    "local_name": "Spinach",    "suitability_score": 72, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 40,  "water_need": "Medium", "estimated_yield": "10-15 tons/ha", "planting_window": "Feb - Apr",         "market_demand": "High",   "reasons": ["Cold tolerant", "Fast crop", "Nutritious"], "warnings": [], "growing_tip": "Multiple harvests — take outer leaves only."},
            {"crop_name": "Potato",     "local_name": "Potato",     "suitability_score": 70, "season_fit": "Good",      "risk_level": "Medium", "duration_days": 100, "water_need": "Medium", "estimated_yield": "25-40 tons/ha", "planting_window": "Apr - May",         "market_demand": "High",   "reasons": ["Staple crop", "High yield"], "warnings": ["Late blight"], "growing_tip": "Use certified seed potatoes."},
        ],
        "Summer": [
            {"crop_name": "Maize",      "local_name": "Corn",       "suitability_score": 88, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 110, "water_need": "Medium", "estimated_yield": "8-12 tons/ha",  "planting_window": "May - Jun",         "market_demand": "High",   "reasons": ["Warm season crop", "High yield", "Multiple uses"], "warnings": [], "growing_tip": "Plant in well-drained soil with full sun."},
            {"crop_name": "Sunflower",  "local_name": "Sunflower",  "suitability_score": 82, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 90,  "water_need": "Medium", "estimated_yield": "2-3 tons/ha",   "planting_window": "Apr - May",         "market_demand": "High",   "reasons": ["Drought tolerant", "Good oil crop", "Heat resistant"], "warnings": [], "growing_tip": "Ensure 60cm spacing between plants."},
            {"crop_name": "Tomato",     "local_name": "Tomato",     "suitability_score": 78, "season_fit": "Good",      "risk_level": "Medium", "duration_days": 75,  "water_need": "High",   "estimated_yield": "30-40 tons/ha", "planting_window": "Apr - May",         "market_demand": "High",   "reasons": ["High value crop", "Warm season ideal"], "warnings": ["Irrigate regularly"], "growing_tip": "Use shade nets during peak summer."},
            {"crop_name": "Pepper",     "local_name": "Pepper",     "suitability_score": 75, "season_fit": "Good",      "risk_level": "Medium", "duration_days": 80,  "water_need": "High",   "estimated_yield": "20-25 tons/ha", "planting_window": "Apr - May",         "market_demand": "High",   "reasons": ["High value vegetable", "Good summer crop"], "warnings": ["Monitor for aphids"], "growing_tip": "Mulch to keep roots cool."},
            {"crop_name": "Cucumber",   "local_name": "Cucumber",   "suitability_score": 72, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 55,  "water_need": "Medium", "estimated_yield": "15-20 tons/ha", "planting_window": "May - Jun",         "market_demand": "High",   "reasons": ["Short duration", "High summer demand"], "warnings": [], "growing_tip": "Use mulching to retain soil moisture."},
            {"crop_name": "Watermelon", "local_name": "Watermelon", "suitability_score": 70, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 75,  "water_need": "Medium", "estimated_yield": "20-30 tons/ha", "planting_window": "May 1 - Jun 1",     "market_demand": "High",   "reasons": ["Heat tolerant", "High market demand"], "warnings": [], "growing_tip": "Sandy loam soils give best flavor."},
        ],
        "Autumn": [
            {"crop_name": "Wheat",      "local_name": "Wheat",      "suitability_score": 90, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 200, "water_need": "Medium", "estimated_yield": "4-6 tons/ha",   "planting_window": "Oct - Nov",         "market_demand": "High",   "reasons": ["Prime autumn crop", "Cold tolerant", "High demand"], "warnings": [], "growing_tip": "Choose winter-hardy varieties for your region."},
            {"crop_name": "Barley",     "local_name": "Barley",     "suitability_score": 82, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 100, "water_need": "Low",    "estimated_yield": "3-5 tons/ha",   "planting_window": "Oct - Nov",         "market_demand": "Medium", "reasons": ["Drought tolerant", "Short growing season"], "warnings": [], "growing_tip": "Excellent rotation crop with legumes."},
            {"crop_name": "Canola",     "local_name": "Rapeseed",   "suitability_score": 78, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 220, "water_need": "Medium", "estimated_yield": "2-4 tons/ha",   "planting_window": "Sep - Oct",         "market_demand": "High",   "reasons": ["Good oil crop", "Cold resistant"], "warnings": [], "growing_tip": "Scout for flea beetles early."},
            {"crop_name": "Potato",     "local_name": "Potato",     "suitability_score": 75, "season_fit": "Good",      "risk_level": "Medium", "duration_days": 90,  "water_need": "Medium", "estimated_yield": "25-40 tons/ha", "planting_window": "Sep - Oct",         "market_demand": "High",   "reasons": ["High yield", "Versatile market demand"], "warnings": ["Store in cool dry conditions"], "growing_tip": "Mound soil over plants as they grow."},
            {"crop_name": "Garlic",     "local_name": "Garlic",     "suitability_score": 72, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 200, "water_need": "Low",    "estimated_yield": "8-12 tons/ha",  "planting_window": "Oct - Nov",         "market_demand": "High",   "reasons": ["Long shelf life", "High demand", "Low water needs"], "warnings": [], "growing_tip": "Plant cloves 5cm deep, tip up."},
            {"crop_name": "Onion",      "local_name": "Onion",      "suitability_score": 68, "season_fit": "Good",      "risk_level": "Medium", "duration_days": 150, "water_need": "Medium", "estimated_yield": "20-30 tons/ha", "planting_window": "Sep - Oct",         "market_demand": "High",   "reasons": ["Overwintered onion", "Early harvest next spring"], "warnings": [], "growing_tip": "Overwintering varieties — Japanese sets recommended."},
        ],
        "Winter": [
            {"crop_name": "Wheat",      "local_name": "Wheat",      "suitability_score": 92, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 200, "water_need": "Low",    "estimated_yield": "4-6 tons/ha",   "planting_window": "Oct - Nov",         "market_demand": "High",   "reasons": ["Cold season staple", "High demand"], "warnings": [], "growing_tip": "Vernalization improves yield in cold climates."},
            {"crop_name": "Rye",        "local_name": "Rye",        "suitability_score": 85, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 200, "water_need": "Low",    "estimated_yield": "3-5 tons/ha",   "planting_window": "Sep - Oct",         "market_demand": "Medium", "reasons": ["Very cold hardy", "Low input crop", "Good cover crop"], "warnings": [], "growing_tip": "Can be planted later than wheat."},
            {"crop_name": "Spinach",    "local_name": "Spinach",    "suitability_score": 78, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 40,  "water_need": "Medium", "estimated_yield": "10-15 tons/ha", "planting_window": "Oct - Feb",         "market_demand": "High",   "reasons": ["Cold tolerant leafy green", "Fast crop"], "warnings": [], "growing_tip": "Multiple cuts possible — harvest outer leaves."},
            {"crop_name": "Garlic",     "local_name": "Garlic",     "suitability_score": 75, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 200, "water_need": "Low",    "estimated_yield": "8-12 tons/ha",  "planting_window": "Oct - Nov",         "market_demand": "High",   "reasons": ["Long shelf life", "High demand", "Low water needs"], "warnings": [], "growing_tip": "Plant cloves 5cm deep, tip up."},
            {"crop_name": "Kale",       "local_name": "Kale",       "suitability_score": 72, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 80,  "water_need": "Medium", "estimated_yield": "15-25 tons/ha", "planting_window": "Jul - Sep",         "market_demand": "High",   "reasons": ["Cold hardy", "Growing health food market"], "warnings": [], "growing_tip": "Tastes sweeter after frost — harvest Nov-Feb."},
            {"crop_name": "Leek",       "local_name": "Leek",       "suitability_score": 68, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 150, "water_need": "Medium", "estimated_yield": "20-30 tons/ha", "planting_window": "Apr - May",         "market_demand": "High",   "reasons": ["Winter vegetable", "High market demand"], "warnings": [], "growing_tip": "Blanch stems by earthing up as plants grow."},
        ],
    }

    return (
        _SEASON_DEFAULTS.get(season)
        or _SEASON_DEFAULTS.get("Summer")
    )
