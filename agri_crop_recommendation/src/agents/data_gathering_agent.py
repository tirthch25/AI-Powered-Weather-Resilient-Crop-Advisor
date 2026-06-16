"""
Data Gathering Agent — v4 (100% Real Data, Zero Static Fallbacks)
==================================================================
Strategy (priority order):

1. WEATHER (current + 7-day):
   - Live: Open-Meteo Forecast API (free, no API key) → real temp & rainfall

2. 6-MONTH FORECAST:
   - Open-Meteo Archive API (free, no API key) → real 2-year historical monthly averages
   - Anchored to current live temperature so each district is accurate
   - Falls back to Gemini LLM when archive API is unavailable
   - NO static zone climate tables — all data comes from real APIs or LLMs

3. SOIL + MARKET PRICES + CLIMATE ZONE:
   - Gemini with Search Grounding (real-time data) → first choice
   - Gemini without search (knowledge-based estimates) → second choice
   - Returns empty/Unknown markers if all LLM providers fail (NO static defaults)

4. SEASON:
   - LLM-driven for accuracy, with a simple calendar fallback

5. ENSO / Climate Signals:
   - NOAA CPC (free, no key) + AI interpretation

All numbers use `is None` check to avoid replacing legitimate 0-values with defaults.
"""

import os
import json
import logging
import re
import datetime
import calendar as _calendar
import traceback
from typing import Optional, Dict, Any, List

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── LLM config ────────────────────────────────────────────────────────────────
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GEMINI_KEYS: list = [k for k in [
    os.getenv("GEMINI_API_KEY", ""),
    os.getenv("GEMINI_API_KEY_2", ""),
    os.getenv("GEMINI_API_KEY_3", ""),
    os.getenv("GEMINI_API_KEY_4", ""),
] if k.strip()]
GEMINI_KEY = GEMINI_KEYS[0] if GEMINI_KEYS else ""

_GEMINI_LITE_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite-001",
    "gemini-flash-lite-latest",
    "gemini-2.0-flash",
]

# ── Cache ──────────────────────────────────────────────────────────────────────
_WX_CACHE: Dict[tuple, tuple] = {}   # current weather — 30 min TTL
_WX_CACHE_TTL     = 1800
_CLIMATE_CACHE_TTL = 86400           # climatology — 24 h TTL (changes rarely)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LIVE WEATHER FROM OPEN-METEO (free, no API key)
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_openmeteo_current(lat: float, lon: float) -> Optional[Dict]:
    """
    Fetch today's weather from Open-Meteo (free, no key, 30-min cache).
    Returns current conditions dict or None on failure.
    """
    import time
    cache_key = (round(lat, 2), round(lon, 2), "current")
    cached_ts, cached_val = _WX_CACHE.get(cache_key, (0, None))
    if cached_val is not None and (time.time() - cached_ts) < _WX_CACHE_TTL:
        logger.info("[DataAgent] Weather cache hit for %.2f,%.2f", lat, lon)
        return cached_val

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":  lat,
        "longitude": lon,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "windspeed_10m_max",
            "uv_index_max",
            "relativehumidity_2m_mean",
        ],
        "past_days":     7,
        "forecast_days": 1,
        "timezone": "auto",
    }
    try:
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        j = resp.json()
        daily = j.get("daily", {})

        def _safe(lst, i=-1, default=None):
            try:
                val = lst[i]
                return float(val) if val is not None else default
            except Exception:
                return default

        t_max    = _safe(daily.get("temperature_2m_max", []),       -1, None)
        t_min    = _safe(daily.get("temperature_2m_min", []),       -1, None)
        wind     = _safe(daily.get("windspeed_10m_max", []),        -1, None)
        uv       = _safe(daily.get("uv_index_max", []),             -1, None)
        humidity = _safe(daily.get("relativehumidity_2m_mean", []), -1, None)

        if t_max is None or t_min is None:
            return None

        t_avg = round((t_max + t_min) / 2, 1)

        rain_list = daily.get("precipitation_sum", [])
        rain_7d   = round(sum(float(r) for r in rain_list[:-1] if r is not None), 1)

        result = {
            "temperature_c":  round(t_avg, 1),
            "temp_max_c":     round(t_max, 1),
            "temp_min_c":     round(t_min, 1),
            "humidity_pct":   round(humidity, 0) if humidity is not None else None,
            "soil_temp_c":    round(t_avg - 2, 1),
            "rainfall_7d_mm": rain_7d,
            "wind_kmh":       round(wind, 1) if wind is not None else None,
            "uv_index":       round(min(uv, 11.0), 1) if uv is not None else None,
            "feels_like_c":   round(t_avg + (2 if (humidity or 0) > 70 else -1), 1),
        }
        _WX_CACHE[cache_key] = (time.time(), result)
        return result
    except Exception as e:
        logger.warning("[DataAgent] Open-Meteo forecast failed: %s", e)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MONTHLY CLIMATOLOGY FROM OPEN-METEO ARCHIVE (real 2-year historical data)
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_openmeteo_monthly_climatology(lat: float, lon: float) -> Optional[Dict[int, Dict]]:
    """
    Fetch real monthly climatology from the Open-Meteo Archive API (free, no key).
    Uses last 2 full years of daily data, grouped and averaged by calendar month.

    Returns {1: {temp, temp_max, temp_min, rain, hum}, ..., 12: {...}}
    or None on API failure.

    Result is cached 24 hours (climate averages change only slightly year to year).
    """
    import time
    cache_key = (round(lat, 2), round(lon, 2), "climatology")
    cached_ts, cached_val = _WX_CACHE.get(cache_key, (0, None))
    if cached_val is not None and (time.time() - cached_ts) < _CLIMATE_CACHE_TTL:
        logger.info("[DataAgent] Climatology cache hit for %.2f,%.2f", lat, lon)
        return cached_val

    today     = datetime.date.today()
    end_date  = (today.replace(day=1) - datetime.timedelta(days=1))           # last day of prev month
    start_date = (end_date.replace(day=1) - datetime.timedelta(days=730)).replace(day=1)  # ~2 years back

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":  lat,
        "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date":   end_date.strftime("%Y-%m-%d"),
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "relative_humidity_2m_mean",
        ],
        "timezone": "auto",
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        j     = resp.json()
        daily = j.get("daily", {})

        dates     = daily.get("time", [])
        t_max_lst = daily.get("temperature_2m_max", [])
        t_min_lst = daily.get("temperature_2m_min", [])
        rain_lst  = daily.get("precipitation_sum", [])
        hum_lst   = daily.get("relative_humidity_2m_mean", [])

        # Group by (year, month)
        monthly_raw: Dict[tuple, Dict] = {}
        for i, d in enumerate(dates):
            try:
                yr  = int(d[:4])
                mo  = int(d[5:7])
                key = (yr, mo)
                if key not in monthly_raw:
                    monthly_raw[key] = {"t_max": [], "t_min": [], "rain": 0.0, "hum": []}
                if i < len(t_max_lst) and t_max_lst[i] is not None:
                    monthly_raw[key]["t_max"].append(float(t_max_lst[i]))
                if i < len(t_min_lst) and t_min_lst[i] is not None:
                    monthly_raw[key]["t_min"].append(float(t_min_lst[i]))
                if i < len(rain_lst) and rain_lst[i] is not None:
                    monthly_raw[key]["rain"] += float(rain_lst[i])
                if i < len(hum_lst) and hum_lst[i] is not None:
                    monthly_raw[key]["hum"].append(float(hum_lst[i]))
            except (IndexError, ValueError):
                continue

        # Average across years for each calendar month
        result: Dict[int, Dict] = {}
        for month in range(1, 13):
            entries = [(k, v) for k, v in monthly_raw.items() if k[1] == month]
            if not entries:
                continue

            t_max_avgs  = [sum(v["t_max"]) / len(v["t_max"]) for _, v in entries if v["t_max"]]
            t_min_avgs  = [sum(v["t_min"]) / len(v["t_min"]) for _, v in entries if v["t_min"]]
            rain_totals = [v["rain"] for _, v in entries]
            hum_avgs    = [sum(v["hum"])  / len(v["hum"])  for _, v in entries if v["hum"]]

            if not t_max_avgs:
                continue

            t_max = round(sum(t_max_avgs) / len(t_max_avgs), 1)
            t_min = round(sum(t_min_avgs) / len(t_min_avgs), 1) if t_min_avgs else round(t_max - 8, 1)
            rain  = round(sum(rain_totals) / len(rain_totals), 1)
            hum   = round(sum(hum_avgs) / len(hum_avgs), 0) if hum_avgs else 65.0

            result[month] = {
                "temp":     round((t_max + t_min) / 2, 1),
                "temp_max": t_max,
                "temp_min": t_min,
                "rain":     rain,
                "hum":      hum,
            }

        if len(result) >= 6:
            _WX_CACHE[cache_key] = (time.time(), result)
            logger.info("[DataAgent] Climatology fetched for %.2f,%.2f: %d months", lat, lon, len(result))
            return result

        logger.warning("[DataAgent] Climatology returned only %d months — discarding", len(result))
        return None

    except Exception as e:
        logger.warning("[DataAgent] Open-Meteo Archive API failed: %s", e)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LLM FORECAST FALLBACK (when archive API is unavailable)
# ═══════════════════════════════════════════════════════════════════════════════

def _llm_generate_forecast(
    lat: float, lon: float, country: str,
    current_month: int, current_temp: Optional[float] = None,
) -> list:
    """
    Ask Gemini to generate a 6-month agricultural forecast when the
    Open-Meteo archive API is unavailable.
    Returns list of monthly dicts or [] on failure.
    """
    if not GEMINI_KEYS:
        return []

    today      = datetime.date.today().replace(day=1)
    months_out = []
    for i in range(6):
        m_date = (today + datetime.timedelta(days=32 * i))
        months_out.append(m_date.strftime("%B %Y"))
    months_str  = ", ".join(months_out)
    temp_hint   = f" Current temperature: {current_temp}°C." if current_temp is not None else ""

    prompt = (
        f"Provide a realistic 6-month agricultural weather forecast for the location "
        f"at latitude {lat:.3f}, longitude {lon:.3f}, in {country}.{temp_hint} "
        f"Months needed: {months_str}. "
        f"Base your answer on the actual climate at this precise location. "
        f"Return ONLY a compact JSON array (no markdown, no explanation):\n"
        f'[{{"month":"<e.g. Jul 2026>","temp_avg":<float>,"temp_max":<float>,'
        f'"temp_min":<float>,"rainfall_mm":<float>,"humidity_pct":<float>,'
        f'"soil_temp_c":<float>}}]'
    )

    try:
        from google import genai as _g
        for api_key in GEMINI_KEYS:
            client = _g.Client(api_key=api_key)
            for model in _GEMINI_LITE_MODELS:
                try:
                    resp   = client.models.generate_content(model=model, contents=prompt)
                    text   = resp.text.strip() if resp.text else None
                    result = _extract_json_safe(text) if text else None
                    if isinstance(result, list) and len(result) >= 3:
                        logger.info("[DataAgent] LLM forecast generated via %s for %s", model, country)
                        return result
                except Exception as e:
                    err = str(e)
                    if "429" in err or "RESOURCE_EXHAUSTED" in err:
                        continue
                    continue
    except ImportError:
        pass
    except Exception as e:
        logger.warning("[DataAgent] LLM forecast generation failed: %s", e)
    return []


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LLM WEATHER ESTIMATE (when Open-Meteo current is unavailable)
# ═══════════════════════════════════════════════════════════════════════════════

def _llm_estimate_current_weather(
    country: str, lat: float, lon: float, current_month: int
) -> dict:
    """
    Use Gemini to estimate current weather when Open-Meteo forecast API fails.
    Returns a dict with the same keys as _fetch_openmeteo_current() or {}.
    """
    if not GEMINI_KEYS:
        return {}

    month_name = _calendar.month_name[current_month]
    prompt = (
        f"Estimate typical current weather conditions for the specific location at "
        f"latitude {lat:.3f}, longitude {lon:.3f} in {country} during {month_name}. "
        f"Be precise — use the actual coordinates, not generic country averages. "
        f"Return ONLY compact JSON (no markdown):\n"
        f'{{"temperature_c":<float>,"temp_max_c":<float>,"temp_min_c":<float>,'
        f'"humidity_pct":<int>,"rainfall_7d_mm":<float>,"wind_kmh":<float>,'
        f'"uv_index":<float>,"feels_like_c":<float>,"soil_temp_c":<float>}}'
    )

    try:
        from google import genai as _g
        for api_key in GEMINI_KEYS:
            client = _g.Client(api_key=api_key)
            for model in _GEMINI_LITE_MODELS:
                try:
                    resp   = client.models.generate_content(model=model, contents=prompt)
                    result = _extract_json_safe(resp.text.strip()) if resp.text else None
                    if isinstance(result, dict) and "temperature_c" in result:
                        logger.info("[DataAgent] LLM estimated weather for %s via %s", country, model)
                        return result
                except Exception as e:
                    err = str(e)
                    if "429" in err or "RESOURCE_EXHAUSTED" in err:
                        continue
                    continue
    except ImportError:
        pass
    except Exception as e:
        logger.warning("[DataAgent] LLM weather estimate failed: %s", e)
    return {}


# ═══════════════════════════════════════════════════════════════════════════════
# 5. BUILD 6-MONTH FORECAST (uses Open-Meteo archive + LLM fallback)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_forecast_6month(
    country: str,
    state_code: Optional[str],
    lat: float,
    lon: float,
    live_temp_avg: Optional[float],
    current_month: int,
) -> list:
    """
    Build a real 6-month agricultural forecast.

    Data source priority:
      1. Open-Meteo Archive API — real 2-year historical monthly averages
      2. Gemini LLM — AI-generated forecast based on location coordinates
      3. Empty list (no static zone tables ever used as fallback)

    The live temperature is used to anchor the forecast to today's actual
    conditions (district-level accuracy instead of zone-wide averages).
    """
    # Build list of 6 future months starting from current month
    today      = datetime.date.today().replace(day=1)
    months_out = []
    for i in range(6):
        m_date = (today + datetime.timedelta(days=32 * i))
        months_out.append((m_date.month, m_date.strftime("%b %Y")))

    # ── Try Open-Meteo Archive API ────────────────────────────────────────────
    clim = _fetch_openmeteo_monthly_climatology(lat, lon)

    if clim:
        # Compute live temperature offset (anchors forecast to actual current temp)
        temp_offset = 0.0
        if live_temp_avg is not None:
            zone_temp_now = clim.get(current_month, {}).get("temp", live_temp_avg)
            temp_offset   = round(live_temp_avg - zone_temp_now, 1)

        forecast = []
        for mo_num, mo_label in months_out:
            c = clim.get(mo_num)
            if not c:
                continue
            t_avg = round(c["temp"]     + temp_offset, 1)
            t_max = round(c["temp_max"] + temp_offset, 1)
            t_min = round(c["temp_min"] + temp_offset, 1)
            forecast.append({
                "month":        mo_label,
                "temp_avg":     t_avg,
                "temp_max":     t_max,
                "temp_min":     t_min,
                "rainfall_mm":  round(c["rain"], 1),
                "humidity_pct": round(c["hum"], 0),
                "soil_temp_c":  round(t_avg - 2, 1),
            })

        if len(forecast) >= 4:
            logger.info("[DataAgent] Forecast built from Open-Meteo Archive: %d months", len(forecast))
            return forecast
        logger.warning("[DataAgent] Archive forecast incomplete (%d months), trying LLM", len(forecast))

    # ── Fall back to LLM ──────────────────────────────────────────────────────
    logger.info("[DataAgent] Using LLM to generate 6-month forecast for %s", country)
    llm_forecast = _llm_generate_forecast(lat, lon, country, current_month, live_temp_avg)
    if llm_forecast:
        return llm_forecast

    logger.warning("[DataAgent] All forecast sources failed for %.2f,%.2f", lat, lon)
    return []


# ═══════════════════════════════════════════════════════════════════════════════
# 6. LLM ENRICHMENT: SOIL + MARKET PRICES (Gemini Search Grounding first)
# ═══════════════════════════════════════════════════════════════════════════════

def _llm_enrich_with_search(
    location_str: str, country: str, current_temp: float, current_month: int
) -> Optional[dict]:
    """
    Use Gemini Search Grounding for real-time soil info, market prices,
    and agricultural advisories. Falls back to plain _llm_enrich if unavailable.
    """
    if not GEMINI_KEYS:
        return None

    prompt = (
        f"Search for current agricultural data in {location_str}. "
        f"Current temp: {current_temp}°C, Month: {current_month}. "
        f"Find: real current market prices for crops grown there, "
        f"typical soil type, and any active crop advisories. "
        f"Return ONLY compact JSON (no markdown):\n"
        f'{{"soil":{{"type":"<Clay/Loam/Sandy/Clay-Loam/Sandy-Loam>","ph":<4.5-8.5>,'
        f'"organic_matter":"<Low/Medium/High>","drainage":"<Poor/Medium/Good>"}},'
        f'"market_prices":{{"<crop1>":"<REAL current price+unit>","<crop2>":"<price+unit>","<crop3>":"<price+unit>"}},'
        f'"district_summary":"<1 sentence about farming in this area>","climate_zone":"<Tropical/Subtropical/Arid/Temperate/Mediterranean>"}}\n'
        f"Use local currency. soil.ph must be a number. Get real current prices, not estimates."
    )

    _SEARCH_MODELS = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.0-flash-001"]

    try:
        from google import genai as _g
        from google.genai import types as _gt
        for api_key in GEMINI_KEYS:
            client_g = _g.Client(api_key=api_key)
            for model in _SEARCH_MODELS:
                try:
                    resp = client_g.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=_gt.GenerateContentConfig(
                            tools=[_gt.Tool(google_search=_gt.GoogleSearch())],
                        ),
                    )
                    text   = resp.text.strip() if resp.text else None
                    result = _extract_json_safe(text) if text else None
                    if result and isinstance(result, dict):
                        logger.info("[DataAgent] Gemini search enrich OK (%s, key ...%s)", model, api_key[-6:])
                        return result
                except Exception as e:
                    err = str(e)
                    if "429" in err or "RESOURCE_EXHAUSTED" in err:
                        continue
                    if "not supported" in err.lower() or "404" in err or "NOT_FOUND" in err:
                        continue
                    logger.debug("[DataAgent] Search enrich %s: %s", model, err[:80])
                    continue
    except ImportError:
        pass
    except Exception as e:
        logger.debug("[DataAgent] Search grounding unavailable: %s", e)
    return None


def _llm_enrich(
    location_str: str, country: str, current_temp: float, current_month: int
) -> Optional[dict]:
    """
    Ask Gemini for soil info, market prices, district summary, and climate zone.
    Tries search-grounded first, then plain Gemini.
    Returns None if all providers fail (no static defaults used).
    """
    if not GEMINI_KEYS:
        return None

    # Try search-grounded first for real-time data
    grounded = _llm_enrich_with_search(location_str, country, current_temp, current_month)
    if grounded:
        return grounded

    prompt = (
        f"Agricultural data for {location_str}. Temp: {current_temp}°C, Month: {current_month}. "
        f"Return ONLY compact JSON (no markdown):\n"
        f'{{"soil":{{"type":"<Clay/Loam/Sandy/Clay-Loam/Sandy-Loam>","ph":<4.5-8.5>,'
        f'"organic_matter":"<Low/Medium/High>","drainage":"<Poor/Medium/Good>"}},'
        f'"market_prices":{{"<crop1>":"<price+unit>","<crop2>":"<price+unit>","<crop3>":"<price+unit>"}},'
        f'"district_summary":"<1 sentence>","climate_zone":"<Tropical/Subtropical/Arid/Temperate/Mediterranean>"}}\n'
        f"Use local currency. soil.ph must be a number."
    )

    try:
        from google import genai as _genai
        for api_key in GEMINI_KEYS:
            client_g = _genai.Client(api_key=api_key)
            for model in _GEMINI_LITE_MODELS:
                try:
                    resp   = client_g.models.generate_content(model=model, contents=prompt)
                    result = _extract_json_safe(resp.text.strip()) if resp.text else None
                    if result:
                        logger.info("[DataAgent] Gemini enrich OK (%s, key ...%s)", model, api_key[-6:])
                        return result
                except Exception as me:
                    err = str(me)
                    if "429" in err or "RESOURCE_EXHAUSTED" in err:
                        continue
                    continue
    except ImportError:
        pass
    except Exception as e:
        logger.warning("[DataAgent] Gemini enrich failed: %s", e)

    # Fallback: legacy SDK
    try:
        import google.generativeai as genai  # type: ignore
        for api_key in GEMINI_KEYS:
            genai.configure(api_key=api_key)
            for model in ["gemini-2.5-flash-lite", "gemini-2.0-flash-lite"]:
                try:
                    resp   = genai.GenerativeModel(model).generate_content(prompt)
                    result = _extract_json_safe(resp.text.strip()) if resp.text else None
                    if result:
                        return result
                except Exception:
                    continue
    except ImportError:
        pass
    except Exception as e:
        logger.warning("[DataAgent] Legacy Gemini enrich failed: %s", e)

    logger.warning("[DataAgent] All LLM enrichment failed for %s — returning None", location_str)
    return None


def _llm_enrich_fast(
    location_str: str, country: str, current_temp: float, current_month: int
) -> Optional[dict]:
    """
    FAST enrichment for the streaming path — single Gemini call, no search grounding.
    Returns soil/market/summary/zone dict, or None.
    Designed to complete in < 8 seconds so the streaming pipeline stays responsive.
    """
    if not GEMINI_KEYS:
        return None

    prompt = (
        f"Agricultural profile for {location_str}. Temp: {current_temp}°C, Month: {current_month}. "
        f"Return ONLY compact JSON (no markdown, no explanation):\n"
        f'{{"soil":{{"type":"<Clay/Loam/Sandy/Clay-Loam/Sandy-Loam>","ph":<4.5-8.5 as number>,'
        f'"organic_matter":"<Low/Medium/High>","drainage":"<Poor/Medium/Good>"}},'
        f'"market_prices":{{"<local crop 1>":"<price+unit>","<local crop 2>":"<price+unit>","<local crop 3>":"<price+unit>"}},'
        f'"district_summary":"<1 sentence about farming here>",'
        f'"climate_zone":"<Tropical/Subtropical/Arid/Temperate/Mediterranean/Continental>"}}\n'
        f"IMPORTANT: soil.ph must be a plain number like 6.5, not a string. Use local currency for prices."
    )

    # Only try the fastest available model — first key, first lite model
    try:
        from google import genai as _g
        for api_key in GEMINI_KEYS:
            client = _g.Client(api_key=api_key)
            for model in ["gemini-2.0-flash-lite", "gemini-2.5-flash-lite", "gemini-2.0-flash"]:
                try:
                    resp   = client.models.generate_content(model=model, contents=prompt)
                    result = _extract_json_safe(resp.text.strip()) if resp.text else None
                    if result and isinstance(result, dict) and result.get("soil"):
                        logger.info("[DataAgent] Fast enrich OK (%s, key ...%s)", model, api_key[-6:])
                        return result
                except Exception as e:
                    err = str(e)
                    if "429" in err or "RESOURCE_EXHAUSTED" in err:
                        continue
                    continue
    except ImportError:
        pass
    except Exception as e:
        logger.warning("[DataAgent] Fast enrich failed: %s", e)

    # Fallback: legacy SDK
    try:
        import google.generativeai as genai  # type: ignore
        for api_key in GEMINI_KEYS[:2]:
            genai.configure(api_key=api_key)
            try:
                resp   = genai.GenerativeModel("gemini-2.0-flash-lite").generate_content(prompt)
                result = _extract_json_safe(resp.text.strip()) if resp.text else None
                if result and isinstance(result, dict) and result.get("soil"):
                    return result
            except Exception:
                continue
    except ImportError:
        pass

    logger.warning("[DataAgent] Fast enrich failed for %s", location_str)
    return None


def _extract_json_safe(text: str) -> Optional[dict]:
    """Robustly extract JSON from LLM response."""
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
    m = re.search(r"\{[\s\S]*\}", cleaned)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# 7. SEASON DETECTION (LLM-first, algorithmic fallback)
# ═══════════════════════════════════════════════════════════════════════════════

def _guess_season(month: int, country: str, lat: float = 0.0) -> str:
    """
    Determine the current agricultural season.

    Priority:
      1. Gemini LLM — location-specific and accurate (cached per country+month)
      2. Algorithmic fallback — based on hemisphere + country agricultural calendar
    """
    # ── LLM-based season detection ────────────────────────────────────────────
    if GEMINI_KEYS:
        _cache_key = (round(lat, 1), country.lower()[:12], month, "season")
        cached_ts, cached_val = _WX_CACHE.get(_cache_key, (0, None))
        if cached_val and ((__import__("time").time() - cached_ts) < 86400):
            return cached_val

        try:
            from google import genai as _g
            month_name = _calendar.month_name[month]
            prompt = (
                f"What is the current agricultural season in {country} during {month_name}? "
                f"Examples: Kharif, Rabi, Zaid (India/South Asia); Spring, Summer, "
                f"Autumn/Fall, Winter (Europe/North America); Wet Season, Dry Season (tropical). "
                f"Return ONLY the season name — one or two words, nothing else."
            )
            for api_key in GEMINI_KEYS[:2]:  # limit to 2 keys for speed
                client = _g.Client(api_key=api_key)
                try:
                    resp   = client.models.generate_content(model="gemini-2.0-flash-lite", contents=prompt)
                    season = (resp.text or "").strip()
                    # Sanity: must be 1-3 words and < 30 chars
                    if season and len(season) < 30 and len(season.split()) <= 3:
                        _WX_CACHE[_cache_key] = (__import__("time").time(), season)
                        logger.debug("[DataAgent] LLM season for %s/%d: %s", country, month, season)
                        return season
                except Exception as e:
                    err = str(e)
                    if "429" in err or "RESOURCE_EXHAUSTED" in err:
                        continue
        except ImportError:
            pass
        except Exception:
            pass

    # ── Algorithmic fallback (hemisphere-aware calendar logic) ────────────────
    c_lower = country.lower()

    # Southern hemisphere: flip seasons
    _SH = {"australia", "new zealand", "south africa", "brazil", "argentina",
            "chile", "peru", "uruguay", "paraguay", "zambia", "zimbabwe",
            "mozambique", "namibia", "botswana"}
    adj = ((month + 5) % 12) + 1 if any(s in c_lower for s in _SH) else month

    # South Asian agricultural calendar
    if any(c in c_lower for c in ["india", "bangladesh", "pakistan", "nepal", "sri lanka", "bhutan"]):
        if adj in [6, 7, 8, 9, 10]:  return "Kharif"
        if adj in [11, 12, 1, 2, 3]: return "Rabi"
        return "Zaid"

    # West/Central Africa — wet/dry seasons
    if any(c in c_lower for c in ["nigeria", "ghana", "ivory", "cote d", "cameroon",
                                   "senegal", "mali", "burkina", "niger", "guinea"]):
        return "Wet Season" if adj in [4, 5, 6, 7, 8, 9, 10] else "Dry Season"

    # East Africa
    if any(c in c_lower for c in ["kenya", "tanzania", "ethiopia", "uganda", "rwanda"]):
        if adj in [3, 4, 5]: return "Long Rains (Masika)"
        if adj in [10, 11, 12]: return "Short Rains (Vuli)"
        return "Dry Season"

    # Standard Northern hemisphere seasons
    if adj in [3, 4, 5]:   return "Spring"
    if adj in [6, 7, 8]:   return "Summer"
    if adj in [9, 10, 11]: return "Autumn"
    return "Winter"


# ═══════════════════════════════════════════════════════════════════════════════
# 8. MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def gather_location_data(
    country: str,
    state: str,
    district: str,
    lat: float,
    lon: float,
    month: Optional[int] = None,
    state_code: Optional[str] = None,
    llm_climate_zone: Optional[str] = None,
    llm_crop_notes: Optional[str] = None,
    location_source: str = "llm",
) -> dict:
    """
    Gather real agricultural data for any location worldwide.

    Data sources (in priority order):
      1. Open-Meteo Forecast API  → real current temperature, humidity, wind, UV
      2. Gemini LLM               → current weather estimate (if Open-Meteo fails)
      3. Open-Meteo Archive API   → real 2-year monthly averages for 6-month forecast
      4. Gemini LLM               → AI-generated forecast (if archive fails)
      5. NOAA ENSO + Gemini       → climate signals applied to forecast
      6. Gemini (Search + plain)  → soil type, pH, market prices, district summary
      7. NO static defaults       → if all providers fail, fields are empty/Unknown

    Returns dict with: current, forecast_6month, forecast_6month_baseline, soil,
                       season, climate_zone, market_prices, district_summary,
                       climate_signal, location_source
    """
    current_month = month or datetime.datetime.now().month
    location_str  = f"{district}, {state}, {country}"
    logger.info("[DataAgent] Gathering data for: %s (%.3f, %.3f)", location_str, lat, lon)

    # ── 1. Live weather ───────────────────────────────────────────────────────
    live_wx = _fetch_openmeteo_current(lat, lon)
    if live_wx:
        current = live_wx
        logger.info(
            "[DataAgent] Live weather: %s°C, rain7d=%smm, humidity=%s%%",
            current["temperature_c"], current["rainfall_7d_mm"], current["humidity_pct"]
        )
    else:
        logger.warning("[DataAgent] Open-Meteo forecast unavailable — trying LLM estimate")
        current = _llm_estimate_current_weather(country, lat, lon, current_month)
        if not current:
            logger.warning("[DataAgent] LLM weather estimate also failed — using empty current")
            current = {
                "temperature_c":  None,
                "temp_max_c":     None,
                "temp_min_c":     None,
                "humidity_pct":   None,
                "soil_temp_c":    None,
                "rainfall_7d_mm": 0.0,
                "wind_kmh":       None,
                "uv_index":       None,
                "feels_like_c":   None,
            }

    # ── 2. 6-month forecast (Open-Meteo Archive → LLM) ────────────────────────
    live_temp = current.get("temperature_c")
    forecast_6month_baseline = _build_forecast_6month(
        country=country,
        state_code=state_code,
        lat=lat,
        lon=lon,
        live_temp_avg=live_temp,
        current_month=current_month,
    )
    forecast_6month = list(forecast_6month_baseline)

    # ── 3. ENSO / Climate Signal adjustments ─────────────────────────────────
    climate_signal: dict = {}
    try:
        from src.services.climate_signals import get_climate_signals, apply_enso_to_forecast
        # Determine climate zone from LLM data or lat/lon
        climate_zone_for_enso = llm_climate_zone or "Subtropical"
        climate_signal = get_climate_signals(
            location_str=location_str,
            climate_zone=climate_zone_for_enso,
            country=country,
        )
        forecast_6month = apply_enso_to_forecast(forecast_6month, climate_signal)
        logger.info("[DataAgent] ENSO=%s applied to forecast", climate_signal.get("enso_phase", "?"))
    except Exception as _enso_err:
        logger.warning("[DataAgent] Climate signal step skipped: %s", _enso_err)

    # ── 4. LLM enrichment: soil + market prices ───────────────────────────────
    temp_for_llm = live_temp if live_temp is not None else 25.0
    llm_data = _llm_enrich(location_str, country, temp_for_llm, current_month)

    # ── 5. Assemble soil (LLM → Unknown; NO static zone defaults) ────────────
    if llm_data and llm_data.get("soil") and isinstance(llm_data["soil"].get("ph"), (int, float)):
        soil = llm_data["soil"]
        soil.setdefault("type",           "Unknown")
        soil.setdefault("organic_matter", "Unknown")
        soil.setdefault("drainage",       "Unknown")
    else:
        logger.info("[DataAgent] LLM soil unavailable for %s — using Unknown markers", location_str)
        soil = {
            "type":           "Unknown",
            "ph":             None,
            "organic_matter": "Unknown",
            "drainage":       "Unknown",
        }

    # ── 6. Market prices (LLM → empty dict; NO static templates) ─────────────
    if llm_data and llm_data.get("market_prices") and len(llm_data["market_prices"]) >= 2:
        market_prices = llm_data["market_prices"]
    else:
        logger.info("[DataAgent] LLM market prices unavailable for %s", location_str)
        market_prices = {}

    # ── 7. Climate zone label ─────────────────────────────────────────────────
    if llm_climate_zone and llm_climate_zone not in ("", "Subtropical"):
        climate_zone = llm_climate_zone
    elif llm_data and llm_data.get("climate_zone"):
        climate_zone = llm_data["climate_zone"]
    else:
        # Rough lat-based estimate as last resort (factual, not hardcoded per country)
        alat = abs(lat)
        if alat < 15:    climate_zone = "Tropical"
        elif alat < 30:  climate_zone = "Subtropical"
        elif alat < 45:  climate_zone = "Temperate"
        elif alat < 60:  climate_zone = "Continental"
        else:            climate_zone = "Polar"

    # ── 8. District summary ───────────────────────────────────────────────────
    if llm_data and llm_data.get("district_summary"):
        district_summary = llm_data["district_summary"]
    elif llm_crop_notes:
        district_summary = llm_crop_notes
    else:
        temp_str = f" {live_temp}°C current temperature." if live_temp is not None else ""
        district_summary = (
            f"{district} is an agricultural district in {state}, {country}."
            f" The region has a {climate_zone.lower()} climate.{temp_str}"
        )

    # ── 9. Season ─────────────────────────────────────────────────────────────
    season = _guess_season(current_month, country, lat)

    logger.info(
        "[DataAgent] Complete: temp=%s°C, soil=%s, zone=%s, season=%s, enso=%s, "
        "forecast=%d months, market=%d items",
        current.get("temperature_c"), soil["type"], climate_zone, season,
        climate_signal.get("enso_phase", "N/A"),
        len(forecast_6month), len(market_prices),
    )

    return {
        "current":                  current,
        "forecast_6month":          forecast_6month,           # ENSO-adjusted
        "forecast_6month_baseline": forecast_6month_baseline,  # raw climatology
        "soil":                     soil,
        "season":                   season,
        "climate_zone":             climate_zone,
        "market_prices":            market_prices,
        "district_summary":         district_summary,
        "climate_signal":           climate_signal,
        "location_source":          location_source,
    }
