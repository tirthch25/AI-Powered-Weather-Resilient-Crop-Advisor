"""
Crop Recommendation Agent — v2 (Speed-Optimised)
=================================================
Strategy (priority order):
  1. In-memory cache (instant) — same location+season+zone returns cached result
  2. Gemini API with 10s hard timeout (fast, ~1-2s when quota available)
  3. Ollama/LLaMA with 20s timeout (fallback, only if Gemini fails)
  4. Zone-based built-in crop table (instant, no API) — final fallback
"""

import os
import json
import logging
import re
import time
from typing import Optional, Dict, List
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GEMINI_KEY   = os.getenv("GEMINI_API_KEY", "")

# ── In-memory result cache ──────────────────────────────────────────────────
# key: (country, state, district, season, climate_zone, irrigation)
# val: (timestamp, crops_list)
_CROP_CACHE: Dict[tuple, tuple] = {}
_CROP_CACHE_TTL = 3600  # 1 hour


def _extract_json(text: str) -> Optional[list]:
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
            return json.loads(match.group())
        except Exception:
            pass
    return None


def _build_prompt(
    country, state, district, season, climate, planning_days,
    irrigation, temp, humidity, current, forecast, soil, market, summary
) -> str:
    forecast_str = ""
    for f in forecast[:3]:  # only 3 months to keep prompt short
        forecast_str += (f"  {f.get('month','')}:  {f.get('temp_avg','?')}C avg, "
                         f"{f.get('rainfall_mm','?')}mm rain\n")

    market_str = ", ".join(f"{k}: {v}" for k, v in list(market.items())[:4]) if market else "N/A"

    return (
        f"Agricultural expert for {district}, {state}, {country}. "
        f"Season:{season}, Climate:{climate}, Irrigation:{irrigation}, Days:{planning_days}. "
        f"Temp:{temp}C, Humidity:{humidity}%, Rain7d:{current.get('rainfall_7d_mm','?')}mm. "
        f"Soil: {soil.get('type','Loam')} pH{soil.get('ph',7.0)}. "
        f"Forecast: {forecast_str.strip()}. Market: {market_str}. "
        f"Context: {summary[:120] if summary else ''}.\n\n"
        f"Return ONLY a valid JSON array of 6 crop recommendations:\n"
        f'[{{"crop_name":"<name>","local_name":"<local>","suitability_score":<0-100>,'
        f'"season_fit":"<Excellent/Good/Fair>","risk_level":"<Low/Medium/High>",'
        f'"duration_days":<int>,"water_need":"<Low/Medium/High>","estimated_yield":"<X-Y tons/ha>",'
        f'"planting_window":"<e.g. Jun 15 - Jul 15>","market_demand":"<High/Medium/Low>",'
        f'"reasons":["<r1>","<r2>"],"warnings":["<w1>"],"growing_tip":"<tip>"}},...]\n'
        f"Sort by suitability_score desc. Only crops genuinely suitable for this region and season."
    )


def _call_gemini(prompt: str) -> Optional[list]:
    try:
        from google import genai as _genai
        client = _genai.Client(api_key=GEMINI_KEY)
        resp = client.models.generate_content(model="gemini-2.0-flash-lite", contents=prompt)
        return _extract_json(resp.text.strip())
    except ImportError:
        try:
            import google.generativeai as genai  # type: ignore
            genai.configure(api_key=GEMINI_KEY)
            resp = genai.GenerativeModel("gemini-2.0-flash-lite").generate_content(prompt)
            return _extract_json(resp.text.strip())
        except Exception:
            return None
    except Exception as e:
        logger.warning(f"[CropAgent] Gemini failed: {e}")
        return None


def _call_ollama(prompt: str) -> Optional[list]:
    try:
        import ollama
        client = ollama.Client(host=OLLAMA_URL)
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": "Return valid JSON array only. No markdown or explanation."},
                {"role": "user",   "content": prompt},
            ],
            options={"temperature": 0.15, "num_ctx": 4096},  # reduced ctx for speed
        )
        return _extract_json(response["message"]["content"].strip())
    except Exception as e:
        logger.warning(f"[CropAgent] Ollama failed: {e}")
        return None


def recommend_crops_agent(
    country: str,
    state: str,
    district: str,
    gathered_data: dict,
    irrigation: str = "Limited",
    planning_days: int = 90,
    soil_override: Optional[dict] = None
) -> list:
    """
    Generate AI crop recommendations for any global location.
    Speed-optimised: cache → Gemini (10s timeout) → Ollama (20s timeout) → zone defaults.
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

    # ── 1. Check cache ────────────────────────────────────────────────────────
    cache_key = (country, state, district, season, climate, irrigation)
    cached_ts, cached_val = _CROP_CACHE.get(cache_key, (0, None))
    if cached_val is not None and (time.time() - cached_ts) < _CROP_CACHE_TTL:
        logger.info(f"[CropAgent] Cache hit for {district}")
        return cached_val

    # ── 2. Build prompt ───────────────────────────────────────────────────────
    prompt = _build_prompt(
        country, state, district, season, climate, planning_days,
        irrigation, temp, humidity, current, forecast, soil, market, summary
    )

    # ── 3. Try Gemini (10s hard timeout) ──────────────────────────────────────
    crops = None
    if GEMINI_KEY:
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(_call_gemini, prompt)
                crops = future.result(timeout=10)
                if crops:
                    logger.info(f"[CropAgent] Gemini returned {len(crops)} crops")
        except FuturesTimeout:
            logger.warning("[CropAgent] Gemini timed out (>10s)")
        except Exception as e:
            logger.warning(f"[CropAgent] Gemini executor error: {e}")

    # ── 4. Try Ollama (20s hard timeout) as fallback ──────────────────────────
    if not crops:
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(_call_ollama, prompt)
                crops = future.result(timeout=20)
                if crops:
                    logger.info(f"[CropAgent] Ollama returned {len(crops)} crops")
        except FuturesTimeout:
            logger.warning("[CropAgent] Ollama timed out (>20s) — using built-in table")
        except Exception as e:
            logger.warning(f"[CropAgent] Ollama executor error: {e}")

    # ── 5. Zone-based fallback ─────────────────────────────────────────────────
    if not crops or not isinstance(crops, list):
        logger.warning("[CropAgent] All LLMs failed/timed out — using zone fallback table")
        crops = _fallback_crops(season, climate, country)

    # ── 6. Sanitize & sort ────────────────────────────────────────────────────
    sanitized = []
    for crop in crops[:8]:
        if not isinstance(crop, dict):
            continue
        crop.setdefault("crop_name",        "Unknown Crop")
        crop.setdefault("local_name",       crop["crop_name"])
        crop.setdefault("suitability_score", 60)
        crop.setdefault("season_fit",       "Good")
        crop.setdefault("risk_level",       "Medium")
        crop.setdefault("duration_days",    90)
        crop.setdefault("water_need",       "Medium")
        crop.setdefault("estimated_yield",  "2-3 tons/hectare")
        crop.setdefault("planting_window",  "Current month")
        crop.setdefault("market_demand",    "Medium")
        crop.setdefault("reasons",         ["Suitable for local climate"])
        crop.setdefault("warnings",        [])
        crop.setdefault("growing_tip",     "Follow local agricultural department guidelines.")
        sanitized.append(crop)

    sanitized.sort(key=lambda x: x.get("suitability_score", 0), reverse=True)

    # Cache result
    _CROP_CACHE[cache_key] = (time.time(), sanitized)
    return sanitized


# ── Zone-based crop tables (instant, no API needed) ────────────────────────────

_ZONE_CROPS: Dict[str, list] = {
    "Kharif": [
        {"crop_name": "Rice",      "local_name": "Chawal",  "suitability_score": 90, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 120, "water_need": "High",   "estimated_yield": "4-6 tons/ha",   "planting_window": "Jun 15 - Jul 15",  "market_demand": "High",   "reasons": ["Primary Kharif crop", "Thrives in monsoon rainfall", "High market demand"], "warnings": [],                            "growing_tip": "Transplant seedlings when soil is saturated."},
        {"crop_name": "Maize",     "local_name": "Makka",   "suitability_score": 85, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 90,  "water_need": "Medium", "estimated_yield": "4-6 tons/ha",   "planting_window": "Jun 1 - Jul 1",    "market_demand": "High",   "reasons": ["Fast growing", "Good market prices", "Drought tolerant once established"], "warnings": [],                            "growing_tip": "Apply nitrogen fertilizer at knee-high stage."},
        {"crop_name": "Soybean",   "local_name": "Soya",    "suitability_score": 82, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 100, "water_need": "Medium", "estimated_yield": "2-3 tons/ha",   "planting_window": "Jun 20 - Jul 10",  "market_demand": "High",   "reasons": ["Nitrogen fixing", "Protein-rich cash crop", "Suits monsoon season"],        "warnings": [],                            "growing_tip": "Inoculate seeds with Rhizobium bacteria."},
        {"crop_name": "Cotton",    "local_name": "Kapas",   "suitability_score": 78, "season_fit": "Good",      "risk_level": "Medium", "duration_days": 180, "water_need": "Medium", "estimated_yield": "2-3 tons/ha",   "planting_window": "May 15 - Jun 15",  "market_demand": "High",   "reasons": ["High value cash crop", "Suits warm conditions"],                           "warnings": ["Monitor for bollworm"],       "growing_tip": "Use Bt cotton varieties for pest resistance."},
        {"crop_name": "Groundnut", "local_name": "Mungfali","suitability_score": 75, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 110, "water_need": "Low",    "estimated_yield": "2-3 tons/ha",   "planting_window": "Jun 10 - Jun 30",  "market_demand": "High",   "reasons": ["Good oil crop", "Sandy-loam soils preferred", "Drought tolerant"],          "warnings": [],                            "growing_tip": "Ensure good drainage to prevent pod rot."},
        {"crop_name": "Bajra",     "local_name": "Bajra",   "suitability_score": 72, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 80,  "water_need": "Low",    "estimated_yield": "2-4 tons/ha",   "planting_window": "Jun 20 - Jul 15",  "market_demand": "Medium", "reasons": ["Drought tolerant", "Short duration", "Good for dry areas"],                 "warnings": [],                            "growing_tip": "Avoid waterlogging — ensure well-drained fields."},
    ],
    "Rabi": [
        {"crop_name": "Wheat",     "local_name": "Gehun",   "suitability_score": 92, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 120, "water_need": "Medium", "estimated_yield": "3-5 tons/ha",   "planting_window": "Nov 1 - Dec 1",    "market_demand": "High",   "reasons": ["Primary Rabi crop", "MSP support", "Cooler temperature ideal"],            "warnings": [],                            "growing_tip": "Irrigate at crown root initiation stage."},
        {"crop_name": "Mustard",   "local_name": "Sarson",  "suitability_score": 85, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 110, "water_need": "Low",    "estimated_yield": "1.5-2.5 tons/ha","planting_window": "Oct 15 - Nov 15",  "market_demand": "High",   "reasons": ["Good oil crop", "Short duration", "Low water requirement"],                 "warnings": [],                            "growing_tip": "Sow in well-drained sandy loam soil."},
        {"crop_name": "Chickpea",  "local_name": "Chana",   "suitability_score": 82, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 100, "water_need": "Low",    "estimated_yield": "1-2 tons/ha",   "planting_window": "Oct 20 - Nov 10",  "market_demand": "High",   "reasons": ["Nitrogen fixing legume", "Dry cool season crop", "High protein value"],     "warnings": ["Sensitive to excess moisture"],  "growing_tip": "Avoid irrigation after flowering."},
        {"crop_name": "Potato",    "local_name": "Aloo",    "suitability_score": 80, "season_fit": "Good",      "risk_level": "Medium", "duration_days": 90,  "water_need": "Medium", "estimated_yield": "20-30 tons/ha", "planting_window": "Oct 15 - Nov 15",  "market_demand": "High",   "reasons": ["High yield vegetable", "Good market demand", "Multiple uses"],             "warnings": ["Monitor for late blight"],   "growing_tip": "Use certified seed potatoes for best results."},
        {"crop_name": "Pea",       "local_name": "Matar",   "suitability_score": 75, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 70,  "water_need": "Medium", "estimated_yield": "5-8 tons/ha",   "planting_window": "Oct 20 - Nov 10",  "market_demand": "High",   "reasons": ["Fast growing", "Good market price", "Cool season crop"],                   "warnings": [],                            "growing_tip": "Provide trellis support for climbing varieties."},
        {"crop_name": "Tomato",    "local_name": "Tamatar", "suitability_score": 70, "season_fit": "Good",      "risk_level": "Medium", "duration_days": 75,  "water_need": "Medium", "estimated_yield": "25-35 tons/ha", "planting_window": "Oct 1 - Nov 1",    "market_demand": "High",   "reasons": ["High value vegetable", "Multiple harvests", "Good cold tolerance"],        "warnings": ["Monitor for pests"],         "growing_tip": "Use drip irrigation for consistent moisture."},
    ],
    "Zaid": [
        {"crop_name": "Cucumber",  "local_name": "Kheera",  "suitability_score": 82, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 55,  "water_need": "Medium", "estimated_yield": "15-20 tons/ha", "planting_window": "Feb 15 - Mar 15",  "market_demand": "High",   "reasons": ["Short duration", "High summer demand", "Good returns"],                    "warnings": [],                            "growing_tip": "Use mulching to retain soil moisture."},
        {"crop_name": "Watermelon","local_name": "Tarbooz", "suitability_score": 80, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 75,  "water_need": "Medium", "estimated_yield": "20-30 tons/ha", "planting_window": "Feb 1 - Mar 1",    "market_demand": "High",   "reasons": ["Excellent summer crop", "High demand", "Heat tolerant"],                   "warnings": [],                            "growing_tip": "Sandy loam soils give best flavor."},
        {"crop_name": "Mung Bean", "local_name": "Moong",   "suitability_score": 78, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 65,  "water_need": "Low",    "estimated_yield": "0.8-1.2 tons/ha","planting_window": "Mar 1 - Apr 1",    "market_demand": "High",   "reasons": ["Short duration pulse", "Nitrogen fixing", "Good Zaid crop"],                "warnings": [],                            "growing_tip": "Harvest at 80% pod maturity."},
        {"crop_name": "Okra",      "local_name": "Bhindi",  "suitability_score": 75, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 60,  "water_need": "Medium", "estimated_yield": "8-12 tons/ha",  "planting_window": "Mar 15 - Apr 15",  "market_demand": "High",   "reasons": ["Popular vegetable", "Heat tolerant", "Fast growing"],                      "warnings": [],                            "growing_tip": "Harvest pods every 2 days to maintain production."},
    ],
    "Summer": [
        {"crop_name": "Maize",     "local_name": "Corn",    "suitability_score": 88, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 90,  "water_need": "Medium", "estimated_yield": "8-12 tons/ha",  "planting_window": "Apr 15 - May 31",  "market_demand": "High",   "reasons": ["Warm season crop", "High yield", "Multiple uses"],                         "warnings": [],                            "growing_tip": "Plant in well-drained soil with full sun."},
        {"crop_name": "Sunflower", "local_name": "Sunflower","suitability_score": 82,"season_fit": "Excellent", "risk_level": "Low",    "duration_days": 90,  "water_need": "Medium", "estimated_yield": "2-3 tons/ha",   "planting_window": "Apr 1 - May 15",   "market_demand": "High",   "reasons": ["Drought tolerant", "Good oil crop", "Heat resistant"],                     "warnings": [],                            "growing_tip": "Ensure 60cm spacing between plants."},
        {"crop_name": "Tomato",    "local_name": "Tomato",  "suitability_score": 78, "season_fit": "Good",      "risk_level": "Medium", "duration_days": 75,  "water_need": "High",   "estimated_yield": "30-40 tons/ha", "planting_window": "Apr 1 - May 1",    "market_demand": "High",   "reasons": ["High value crop", "Warm season ideal", "Good market prices"],              "warnings": ["Irrigate regularly"],        "growing_tip": "Use shade nets during peak summer."},
        {"crop_name": "Pepper",    "local_name": "Bell Pepper","suitability_score": 75,"season_fit":"Good",     "risk_level": "Medium", "duration_days": 80,  "water_need": "High",   "estimated_yield": "20-25 tons/ha", "planting_window": "Apr 15 - May 15",  "market_demand": "High",   "reasons": ["High value vegetable", "Good summer crop"],                                "warnings": ["Monitor for aphids"],        "growing_tip": "Mulch to keep roots cool."},
    ],
    "Autumn": [
        {"crop_name": "Wheat",     "local_name": "Wheat",   "suitability_score": 90, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 120, "water_need": "Medium", "estimated_yield": "4-6 tons/ha",   "planting_window": "Oct 1 - Nov 15",   "market_demand": "High",   "reasons": ["Prime autumn/winter crop", "Cold tolerant", "High demand"],                "warnings": [],                            "growing_tip": "Choose winter-hardy varieties for your region."},
        {"crop_name": "Barley",    "local_name": "Barley",  "suitability_score": 82, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 100, "water_need": "Low",    "estimated_yield": "3-5 tons/ha",   "planting_window": "Oct 1 - Nov 1",    "market_demand": "Medium", "reasons": ["Drought tolerant", "Short growing season", "Multiple uses"],               "warnings": [],                            "growing_tip": "Excellent rotation crop with legumes."},
        {"crop_name": "Canola",    "local_name": "Rapeseed","suitability_score": 78, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 110, "water_need": "Medium", "estimated_yield": "2-4 tons/ha",   "planting_window": "Sep 15 - Oct 31",  "market_demand": "High",   "reasons": ["Good oil crop", "Cold resistant", "Well-established market"],              "warnings": [],                            "growing_tip": "Scout for flea beetles early in the season."},
        {"crop_name": "Potato",    "local_name": "Potato",  "suitability_score": 75, "season_fit": "Good",      "risk_level": "Medium", "duration_days": 90,  "water_need": "Medium", "estimated_yield": "25-40 tons/ha", "planting_window": "Sep 1 - Oct 15",   "market_demand": "High",   "reasons": ["High yield", "Versatile market demand", "Suits cool soils"],              "warnings": ["Store in cool dry conditions"], "growing_tip": "Mound soil over plants as they grow."},
    ],
    "Spring": [
        {"crop_name": "Pea",       "local_name": "Pea",     "suitability_score": 88, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 70,  "water_need": "Medium", "estimated_yield": "5-8 tons/ha",   "planting_window": "Mar 1 - Apr 15",   "market_demand": "High",   "reasons": ["Cool season crop", "Excellent spring fit", "High demand"],                 "warnings": [],                            "growing_tip": "Plant as soon as soil can be worked."},
        {"crop_name": "Lettuce",   "local_name": "Lettuce", "suitability_score": 82, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 45,  "water_need": "Medium", "estimated_yield": "20-30 tons/ha", "planting_window": "Mar 15 - Apr 30",  "market_demand": "High",   "reasons": ["Fast growing", "High value", "Cool weather ideal"],                        "warnings": ["Bolt risk in heat"],         "growing_tip": "Harvest before temperatures exceed 25°C."},
        {"crop_name": "Carrot",    "local_name": "Carrot",  "suitability_score": 78, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 80,  "water_need": "Medium", "estimated_yield": "30-40 tons/ha", "planting_window": "Mar 1 - May 1",    "market_demand": "High",   "reasons": ["Good storage crop", "Spring optimal", "High nutritional value"],          "warnings": [],                            "growing_tip": "Loose, deep soil gives best root formation."},
        {"crop_name": "Onion",     "local_name": "Onion",   "suitability_score": 75, "season_fit": "Good",      "risk_level": "Medium", "duration_days": 100, "water_need": "Medium", "estimated_yield": "20-30 tons/ha", "planting_window": "Mar 1 - Apr 1",    "market_demand": "High",   "reasons": ["High demand", "Long shelf life", "Good spring crop"],                      "warnings": ["Monitor for thrips"],        "growing_tip": "Stop irrigation 2 weeks before harvest."},
    ],
    "Winter": [
        {"crop_name": "Wheat",     "local_name": "Wheat",   "suitability_score": 92, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 120, "water_need": "Low",    "estimated_yield": "3-5 tons/ha",   "planting_window": "Oct 1 - Nov 15",   "market_demand": "High",   "reasons": ["Cold season staple", "High demand", "Government MSP support"],             "warnings": [],                            "growing_tip": "Vernalization improves yield in cold climates."},
        {"crop_name": "Rye",       "local_name": "Rye",     "suitability_score": 85, "season_fit": "Excellent", "risk_level": "Low",    "duration_days": 150, "water_need": "Low",    "estimated_yield": "2-4 tons/ha",   "planting_window": "Sep 15 - Oct 31",  "market_demand": "Medium", "reasons": ["Very cold hardy", "Low input crop", "Good cover crop"],                    "warnings": [],                            "growing_tip": "Can be planted later than wheat."},
        {"crop_name": "Spinach",   "local_name": "Palak",   "suitability_score": 78, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 40,  "water_need": "Medium", "estimated_yield": "10-15 tons/ha", "planting_window": "Oct 15 - Feb 28",  "market_demand": "High",   "reasons": ["Cold tolerant leafy green", "Fast crop", "Nutritious"],                    "warnings": [],                            "growing_tip": "Multiple cuts possible — harvest outer leaves."},
        {"crop_name": "Garlic",    "local_name": "Lahsun",  "suitability_score": 72, "season_fit": "Good",      "risk_level": "Low",    "duration_days": 170, "water_need": "Low",    "estimated_yield": "8-12 tons/ha",  "planting_window": "Oct 1 - Nov 1",    "market_demand": "High",   "reasons": ["Long shelf life", "High demand", "Low water needs"],                       "warnings": [],                            "growing_tip": "Plant cloves 5cm deep, tip up."},
    ],
}


def _fallback_crops(season: str, climate: str, country: str = "") -> list:
    """Return zone-appropriate fallback crops instantly without any API call."""
    # Direct season match
    if season in _ZONE_CROPS:
        return _ZONE_CROPS[season]
    # Climate fallback
    if "tropical" in climate.lower():
        return _ZONE_CROPS.get("Kharif", [])
    if "arid" in climate.lower():
        return _ZONE_CROPS.get("Summer", [])
    # Month-based reasonable default
    return _ZONE_CROPS.get("Summer", [])
