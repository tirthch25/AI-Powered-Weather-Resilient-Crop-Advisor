"""
Data Gathering Agent — v2 (Real Data Pipeline)
===============================================
Strategy (priority order):

1. WEATHER (current + 6-month forecast):
   - Live: Open-Meteo API (free, no API key) → real temp & rainfall for today
   - 6-month forecast: zone-based climatology from history.py, anchored to
     the live Open-Meteo temperature so each district shows accurate seasonal
     curves (not generic India-wide 28 °C)

2. SOIL:
   - LLaMA / Gemini AI call with location context
   - Falls back to zone-based defaults when LLM unavailable

3. MARKET PRICES:
   - LLaMA / Gemini AI call with country + crop context
   - Falls back to currency-aware generic prices

All numbers use `is None` check (not `or`) to avoid replacing legitimate
0-values (e.g. winter rainfall) with hardcoded defaults.
"""

import os
import json
import logging
import re
import datetime
import traceback
from typing import Optional, Dict, Any

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── LLM config (for soil/market enrichment only) ──────────────────────────────
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_URL   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GEMINI_KEY   = os.getenv("GEMINI_API_KEY", "")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LIVE WEATHER FROM OPEN-METEO (free, no API key)
# ═══════════════════════════════════════════════════════════════════════════════

# ── In-memory weather cache: {(lat2dp, lon2dp): (timestamp, result)} ──────────
_WX_CACHE: Dict[tuple, tuple] = {}  # key: (lat, lon) rounded to 2dp, val: (ts, dict)
_WX_CACHE_TTL = 1800  # 30 minutes


def _fetch_openmeteo_current(lat: float, lon: float) -> Optional[Dict]:
    """
    Fetch today's weather from Open-Meteo (free, no key).
    Uses an in-memory 30-minute cache so repeated requests for the same
    location are instant.
    """
    import time
    cache_key = (round(lat, 2), round(lon, 2))
    cached_ts, cached_val = _WX_CACHE.get(cache_key, (0, None))
    if cached_val is not None and (time.time() - cached_ts) < _WX_CACHE_TTL:
        logger.info(f"[DataAgent] Weather cache hit for {cache_key}")
        return cached_val

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":  lat,
        "longitude": lon,
        # Daily only (no heavy hourly payload)
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "windspeed_10m_max",
            "uv_index_max",
            "relativehumidity_2m_mean",  # daily mean — much lighter than hourly
        ],
        "past_days":     7,
        "forecast_days": 1,
        "timezone": "auto",
    }
    try:
        resp = requests.get(url, params=params, timeout=8)  # reduced from 12s
        resp.raise_for_status()
        j = resp.json()
        daily = j.get("daily", {})

        def _safe(lst, i=-1, default=None):
            try:
                val = lst[i]
                return float(val) if val is not None else default
            except Exception:
                return default

        t_max    = _safe(daily.get("temperature_2m_max", []),         -1, 30.0)
        t_min    = _safe(daily.get("temperature_2m_min", []),         -1, 20.0)
        t_avg    = round((t_max + t_min) / 2, 1)
        wind     = _safe(daily.get("windspeed_10m_max", []),          -1, 12.0)
        uv       = _safe(daily.get("uv_index_max", []),               -1,  6.0)
        humidity = _safe(daily.get("relativehumidity_2m_mean", []),   -1, 65.0)

        # 7-day rainfall = sum of past 7 days' precipitation_sum
        rain_list = daily.get("precipitation_sum", [])
        rain_7d   = round(sum(float(r) for r in rain_list[:-1] if r is not None), 1)

        result = {
            "temperature_c":  round(t_avg, 1),
            "temp_max_c":     round(t_max, 1),
            "temp_min_c":     round(t_min, 1),
            "humidity_pct":   round(humidity, 0),
            "soil_temp_c":    round(t_avg - 2, 1),
            "rainfall_7d_mm": rain_7d,
            "wind_kmh":       round(wind, 1),
            "uv_index":       round(min(uv, 11.0), 1),
            "feels_like_c":   round(t_avg + (2 if humidity > 70 else -1), 1),
        }
        _WX_CACHE[cache_key] = (time.time(), result)
        return result
    except Exception as e:
        logger.warning(f"[DataAgent] Open-Meteo failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 6-MONTH FORECAST FROM CLIMATOLOGY + LIVE ANCHOR
# ═══════════════════════════════════════════════════════════════════════════════

# World-scale zone lookup — maps country + lat/lon band to a climate zone
# that resolves to one of: North, South, East, West, Central, Northeast,
# Tropical, Arid, Temperate, Mediterranean, Continental, Oceanic

_COUNTRY_TO_ZONE: Dict[str, str] = {
    # India — handled via history.py state code mapping
    "india": "India",
    # Tropical
    "bangladesh": "Tropical", "myanmar": "Tropical", "cambodia": "Tropical",
    "thailand": "Tropical", "vietnam": "Tropical", "malaysia": "Tropical",
    "indonesia": "Tropical", "philippines": "Tropical", "sri lanka": "Tropical",
    "ghana": "Tropical", "nigeria": "Tropical", "ivory coast": "Tropical",
    "côte d'ivoire": "Tropical", "uganda": "Tropical", "tanzania": "Tropical",
    "ethiopia": "Tropical", "kenya": "Tropical",
    # Subtropical / Semi-Arid
    "pakistan": "Subtropical", "nepal": "Subtropical", "afghanistan": "Subtropical",
    "egypt": "Arid", "morocco": "Mediterranean", "tunisia": "Mediterranean",
    "sudan": "Arid", "iraq": "Arid", "iran": "Arid", "saudi arabia": "Arid",
    # Temperate
    "germany": "Temperate", "france": "Temperate", "united kingdom": "Temperate",
    "poland": "Temperate", "ukraine": "Temperate", "romania": "Temperate",
    "italy": "Mediterranean", "spain": "Mediterranean", "portugal": "Mediterranean",
    "greece": "Mediterranean", "turkey": "Mediterranean",
    # Continental (hot summers, cold winters)
    "russia": "Continental", "china": "Continental", "ukraine": "Continental",
    # Americas
    "united states": "Temperate_Americas", "canada": "Temperate_Americas",
    "mexico": "Subtropical", "brazil": "Tropical_Americas",
    "argentina": "Subtropical_S", "colombia": "Tropical_Americas",
    "peru": "Subtropical",
    # Oceania
    "australia": "Arid_Oceania", "new zealand": "Temperate",
}

# Monthly climate lookup per zone: {month: {temp, temp_max, temp_min, rainfall, humidity}}
_ZONE_CLIMATE: Dict[str, Dict[int, Dict]] = {
    "Tropical": {
        1:  {"temp": 28, "temp_max": 33, "temp_min": 22, "rain": 45,  "hum": 82},
        2:  {"temp": 29, "temp_max": 34, "temp_min": 23, "rain": 35,  "hum": 80},
        3:  {"temp": 30, "temp_max": 35, "temp_min": 24, "rain": 50,  "hum": 78},
        4:  {"temp": 30, "temp_max": 35, "temp_min": 24, "rain": 90,  "hum": 80},
        5:  {"temp": 29, "temp_max": 33, "temp_min": 24, "rain": 180, "hum": 84},
        6:  {"temp": 28, "temp_max": 32, "temp_min": 23, "rain": 230, "hum": 87},
        7:  {"temp": 27, "temp_max": 31, "temp_min": 23, "rain": 250, "hum": 88},
        8:  {"temp": 27, "temp_max": 31, "temp_min": 23, "rain": 240, "hum": 88},
        9:  {"temp": 28, "temp_max": 32, "temp_min": 23, "rain": 200, "hum": 85},
        10: {"temp": 28, "temp_max": 33, "temp_min": 23, "rain": 150, "hum": 83},
        11: {"temp": 28, "temp_max": 33, "temp_min": 22, "rain": 80,  "hum": 81},
        12: {"temp": 28, "temp_max": 33, "temp_min": 22, "rain": 55,  "hum": 82},
    },
    "Subtropical": {
        1:  {"temp": 18, "temp_max": 25, "temp_min": 10, "rain": 30,  "hum": 65},
        2:  {"temp": 20, "temp_max": 27, "temp_min": 12, "rain": 25,  "hum": 60},
        3:  {"temp": 24, "temp_max": 31, "temp_min": 16, "rain": 20,  "hum": 55},
        4:  {"temp": 29, "temp_max": 36, "temp_min": 21, "rain": 15,  "hum": 48},
        5:  {"temp": 33, "temp_max": 40, "temp_min": 25, "rain": 20,  "hum": 42},
        6:  {"temp": 33, "temp_max": 38, "temp_min": 26, "rain": 65,  "hum": 65},
        7:  {"temp": 30, "temp_max": 34, "temp_min": 24, "rain": 120, "hum": 78},
        8:  {"temp": 29, "temp_max": 33, "temp_min": 24, "rain": 110, "hum": 79},
        9:  {"temp": 29, "temp_max": 33, "temp_min": 23, "rain": 80,  "hum": 74},
        10: {"temp": 26, "temp_max": 32, "temp_min": 20, "rain": 35,  "hum": 66},
        11: {"temp": 22, "temp_max": 28, "temp_min": 15, "rain": 22,  "hum": 63},
        12: {"temp": 18, "temp_max": 25, "temp_min": 11, "rain": 28,  "hum": 64},
    },
    "Arid": {
        1:  {"temp": 14, "temp_max": 20, "temp_min":  8, "rain":  8,  "hum": 55},
        2:  {"temp": 16, "temp_max": 23, "temp_min": 10, "rain":  6,  "hum": 48},
        3:  {"temp": 20, "temp_max": 28, "temp_min": 13, "rain":  4,  "hum": 40},
        4:  {"temp": 26, "temp_max": 35, "temp_min": 18, "rain":  2,  "hum": 32},
        5:  {"temp": 31, "temp_max": 40, "temp_min": 22, "rain":  1,  "hum": 25},
        6:  {"temp": 35, "temp_max": 44, "temp_min": 26, "rain":  0,  "hum": 22},
        7:  {"temp": 36, "temp_max": 45, "temp_min": 27, "rain":  0,  "hum": 25},
        8:  {"temp": 35, "temp_max": 44, "temp_min": 26, "rain":  0,  "hum": 28},
        9:  {"temp": 31, "temp_max": 40, "temp_min": 22, "rain":  1,  "hum": 32},
        10: {"temp": 24, "temp_max": 32, "temp_min": 16, "rain":  3,  "hum": 40},
        11: {"temp": 18, "temp_max": 25, "temp_min": 11, "rain":  5,  "hum": 48},
        12: {"temp": 14, "temp_max": 20, "temp_min":  8, "rain":  8,  "hum": 55},
    },
    "Mediterranean": {
        1:  {"temp": 10, "temp_max": 14, "temp_min":  6, "rain": 80,  "hum": 72},
        2:  {"temp": 11, "temp_max": 15, "temp_min":  7, "rain": 65,  "hum": 70},
        3:  {"temp": 13, "temp_max": 18, "temp_min":  9, "rain": 50,  "hum": 65},
        4:  {"temp": 16, "temp_max": 21, "temp_min": 11, "rain": 40,  "hum": 60},
        5:  {"temp": 20, "temp_max": 25, "temp_min": 15, "rain": 25,  "hum": 55},
        6:  {"temp": 25, "temp_max": 31, "temp_min": 19, "rain": 10,  "hum": 48},
        7:  {"temp": 28, "temp_max": 34, "temp_min": 22, "rain":  5,  "hum": 42},
        8:  {"temp": 28, "temp_max": 34, "temp_min": 22, "rain":  8,  "hum": 44},
        9:  {"temp": 24, "temp_max": 30, "temp_min": 18, "rain": 25,  "hum": 52},
        10: {"temp": 18, "temp_max": 23, "temp_min": 13, "rain": 60,  "hum": 62},
        11: {"temp": 13, "temp_max": 18, "temp_min":  9, "rain": 85,  "hum": 70},
        12: {"temp": 10, "temp_max": 14, "temp_min":  6, "rain": 90,  "hum": 74},
    },
    "Temperate": {
        1:  {"temp":  4, "temp_max":  8, "temp_min":  0, "rain": 55,  "hum": 80},
        2:  {"temp":  5, "temp_max":  9, "temp_min":  1, "rain": 45,  "hum": 78},
        3:  {"temp":  8, "temp_max": 13, "temp_min":  3, "rain": 50,  "hum": 74},
        4:  {"temp": 13, "temp_max": 18, "temp_min":  7, "rain": 55,  "hum": 70},
        5:  {"temp": 17, "temp_max": 23, "temp_min": 12, "rain": 60,  "hum": 68},
        6:  {"temp": 21, "temp_max": 26, "temp_min": 15, "rain": 65,  "hum": 65},
        7:  {"temp": 23, "temp_max": 28, "temp_min": 17, "rain": 70,  "hum": 65},
        8:  {"temp": 22, "temp_max": 27, "temp_min": 17, "rain": 65,  "hum": 66},
        9:  {"temp": 18, "temp_max": 23, "temp_min": 13, "rain": 60,  "hum": 70},
        10: {"temp": 13, "temp_max": 17, "temp_min":  8, "rain": 60,  "hum": 75},
        11: {"temp":  7, "temp_max": 11, "temp_min":  3, "rain": 60,  "hum": 80},
        12: {"temp":  4, "temp_max":  7, "temp_min":  0, "rain": 55,  "hum": 82},
    },
    "Continental": {
        1:  {"temp": -5, "temp_max":  0, "temp_min": -10, "rain": 30,  "hum": 78},
        2:  {"temp": -3, "temp_max":  2, "temp_min":  -8, "rain": 28,  "hum": 75},
        3:  {"temp":  4, "temp_max": 10, "temp_min":  -2, "rain": 32,  "hum": 70},
        4:  {"temp": 12, "temp_max": 18, "temp_min":   5, "rain": 40,  "hum": 65},
        5:  {"temp": 19, "temp_max": 25, "temp_min":  12, "rain": 50,  "hum": 62},
        6:  {"temp": 24, "temp_max": 30, "temp_min":  17, "rain": 65,  "hum": 65},
        7:  {"temp": 26, "temp_max": 32, "temp_min":  19, "rain": 70,  "hum": 68},
        8:  {"temp": 25, "temp_max": 31, "temp_min":  18, "rain": 60,  "hum": 66},
        9:  {"temp": 18, "temp_max": 24, "temp_min":  12, "rain": 45,  "hum": 68},
        10: {"temp": 10, "temp_max": 16, "temp_min":   4, "rain": 40,  "hum": 72},
        11: {"temp":  2, "temp_max":  7, "temp_min":  -3, "rain": 35,  "hum": 78},
        12: {"temp": -4, "temp_max":  1, "temp_min":  -9, "rain": 30,  "hum": 80},
    },
    "Temperate_Americas": {
        1:  {"temp":  2, "temp_max":  7, "temp_min": -3,  "rain": 60,  "hum": 78},
        2:  {"temp":  4, "temp_max":  9, "temp_min": -1,  "rain": 55,  "hum": 75},
        3:  {"temp":  9, "temp_max": 14, "temp_min":  3,  "rain": 65,  "hum": 72},
        4:  {"temp": 14, "temp_max": 20, "temp_min":  8,  "rain": 75,  "hum": 68},
        5:  {"temp": 19, "temp_max": 25, "temp_min": 13,  "rain": 85,  "hum": 65},
        6:  {"temp": 24, "temp_max": 30, "temp_min": 18,  "rain": 90,  "hum": 68},
        7:  {"temp": 26, "temp_max": 32, "temp_min": 20,  "rain": 80,  "hum": 70},
        8:  {"temp": 25, "temp_max": 31, "temp_min": 19,  "rain": 75,  "hum": 70},
        9:  {"temp": 20, "temp_max": 26, "temp_min": 14,  "rain": 70,  "hum": 68},
        10: {"temp": 14, "temp_max": 19, "temp_min":  9,  "rain": 65,  "hum": 70},
        11: {"temp":  7, "temp_max": 12, "temp_min":  2,  "rain": 60,  "hum": 75},
        12: {"temp":  3, "temp_max":  8, "temp_min": -2,  "rain": 60,  "hum": 78},
    },
    "Tropical_Americas": {
        1:  {"temp": 27, "temp_max": 32, "temp_min": 21, "rain": 200, "hum": 82},
        2:  {"temp": 27, "temp_max": 32, "temp_min": 21, "rain": 180, "hum": 82},
        3:  {"temp": 28, "temp_max": 33, "temp_min": 22, "rain": 200, "hum": 83},
        4:  {"temp": 28, "temp_max": 33, "temp_min": 22, "rain": 250, "hum": 84},
        5:  {"temp": 27, "temp_max": 32, "temp_min": 21, "rain": 300, "hum": 87},
        6:  {"temp": 26, "temp_max": 31, "temp_min": 21, "rain": 180, "hum": 85},
        7:  {"temp": 26, "temp_max": 31, "temp_min": 20, "rain": 130, "hum": 82},
        8:  {"temp": 27, "temp_max": 32, "temp_min": 21, "rain": 140, "hum": 82},
        9:  {"temp": 27, "temp_max": 32, "temp_min": 21, "rain": 180, "hum": 85},
        10: {"temp": 27, "temp_max": 32, "temp_min": 21, "rain": 250, "hum": 87},
        11: {"temp": 27, "temp_max": 32, "temp_min": 21, "rain": 270, "hum": 86},
        12: {"temp": 27, "temp_max": 32, "temp_min": 21, "rain": 220, "hum": 84},
    },
    "Subtropical_S": {  # Southern hemisphere subtropical (Argentina, etc.)
        1:  {"temp": 24, "temp_max": 30, "temp_min": 18, "rain": 90,  "hum": 70},
        2:  {"temp": 23, "temp_max": 29, "temp_min": 17, "rain": 80,  "hum": 70},
        3:  {"temp": 20, "temp_max": 26, "temp_min": 14, "rain": 70,  "hum": 68},
        4:  {"temp": 16, "temp_max": 22, "temp_min": 10, "rain": 55,  "hum": 68},
        5:  {"temp": 12, "temp_max": 18, "temp_min":  7, "rain": 50,  "hum": 72},
        6:  {"temp":  9, "temp_max": 15, "temp_min":  4, "rain": 45,  "hum": 76},
        7:  {"temp":  9, "temp_max": 15, "temp_min":  3, "rain": 35,  "hum": 75},
        8:  {"temp": 11, "temp_max": 17, "temp_min":  5, "rain": 40,  "hum": 72},
        9:  {"temp": 14, "temp_max": 20, "temp_min":  8, "rain": 55,  "hum": 70},
        10: {"temp": 18, "temp_max": 24, "temp_min": 12, "rain": 80,  "hum": 68},
        11: {"temp": 21, "temp_max": 27, "temp_min": 15, "rain": 85,  "hum": 68},
        12: {"temp": 23, "temp_max": 29, "temp_min": 17, "rain": 90,  "hum": 70},
    },
    "Arid_Oceania": {  # Australia interior / semi-arid
        1:  {"temp": 32, "temp_max": 38, "temp_min": 21, "rain": 35,  "hum": 35},
        2:  {"temp": 31, "temp_max": 37, "temp_min": 21, "rain": 40,  "hum": 38},
        3:  {"temp": 28, "temp_max": 34, "temp_min": 18, "rain": 35,  "hum": 42},
        4:  {"temp": 23, "temp_max": 29, "temp_min": 14, "rain": 25,  "hum": 45},
        5:  {"temp": 18, "temp_max": 23, "temp_min": 10, "rain": 30,  "hum": 55},
        6:  {"temp": 14, "temp_max": 19, "temp_min":  7, "rain": 40,  "hum": 62},
        7:  {"temp": 13, "temp_max": 18, "temp_min":  6, "rain": 35,  "hum": 62},
        8:  {"temp": 15, "temp_max": 21, "temp_min":  7, "rain": 30,  "hum": 58},
        9:  {"temp": 19, "temp_max": 24, "temp_min": 11, "rain": 25,  "hum": 50},
        10: {"temp": 23, "temp_max": 29, "temp_min": 14, "rain": 25,  "hum": 44},
        11: {"temp": 27, "temp_max": 33, "temp_min": 17, "rain": 25,  "hum": 37},
        12: {"temp": 30, "temp_max": 36, "temp_min": 20, "rain": 30,  "hum": 34},
    },
}


def _get_world_zone(country: str, lat: float) -> str:
    """Map country name to a climate zone key."""
    c = country.lower().strip()
    # Check exact/substring match
    for key, zone in _COUNTRY_TO_ZONE.items():
        if key in c:
            return zone
    # Latitude-based fallback
    alat = abs(lat)
    if alat < 15:    return "Tropical"
    if alat < 30:    return "Subtropical"
    if alat < 45:    return "Temperate"
    if alat < 60:    return "Continental"
    return "Temperate"


def _build_forecast_6month(
    country: str,
    state_code: Optional[str],
    lat: float,
    live_temp_avg: Optional[float],
    current_month: int,
) -> list:
    """
    Build a 6-month forecast using zone-based climatology, anchored
    to the live temperature so each district shows accurate values.
    """
    months_out = []
    today = datetime.date.today().replace(day=1)
    for i in range(7):
        m_date = (today + datetime.timedelta(days=32 * i))
        months_out.append((m_date.month, m_date.strftime("%b %Y")))

    # --- Try India-specific history.py first ---
    zone_clim: Dict[int, Dict] = {}
    if country.lower() in ("india", "भारत") and state_code:
        try:
            from src.weather.history import get_zone_for_region, get_monthly_climate
            region_hint = state_code.upper()  # e.g. "MH", "UP"
            zone = get_zone_for_region(region_hint)
            for mo in range(1, 13):
                c = get_monthly_climate(zone, mo)
                zone_clim[mo] = {
                    "temp":     c["temperature"],
                    "temp_max": c.get("temp_max", c["temperature"] + 7),
                    "temp_min": c.get("temp_min", c["temperature"] - 7),
                    "rain":     c["rainfall"],
                    "hum":      c["humidity"],
                }
        except Exception:
            logger.debug("[DataAgent] India history.py lookup failed, using world zone")

    # --- Fallback: world zone table ---
    if not zone_clim:
        world_zone = _get_world_zone(country, lat)
        raw = _ZONE_CLIMATE.get(world_zone, _ZONE_CLIMATE["Subtropical"])
        for mo in range(1, 13):
            r = raw.get(mo, raw.get(1, {}))
            zone_clim[mo] = {
                "temp":     r.get("temp", 25),
                "temp_max": r.get("temp_max", r.get("temp", 25) + 6),
                "temp_min": r.get("temp_min", r.get("temp", 25) - 6),
                "rain":     r.get("rain", 60),
                "hum":      r.get("hum", 65),
            }

    # Compute live temperature offset (district vs zone average for current month)
    temp_offset = 0.0
    if live_temp_avg is not None:
        zone_temp_now = zone_clim.get(current_month, {}).get("temp", live_temp_avg)
        temp_offset = round(live_temp_avg - zone_temp_now, 1)

    forecast = []
    for mo_num, mo_label in months_out:
        c = zone_clim.get(mo_num, zone_clim.get(1, {}))
        t_avg = round(c["temp"]     + temp_offset, 1)
        t_max = round(c["temp_max"] + temp_offset, 1)
        t_min = round(c["temp_min"] + temp_offset, 1)
        forecast.append({
            "month":       mo_label,
            "temp_avg":    t_avg,
            "temp_max":    t_max,
            "temp_min":    t_min,
            "rainfall_mm": round(c["rain"], 0),
            "humidity_pct":round(c["hum"],  0),
            "soil_temp_c": round(t_avg - 2, 1),
        })
    return forecast


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LLM ENRICHMENT: SOIL + MARKET PRICES
# ═══════════════════════════════════════════════════════════════════════════════

def _llm_enrich(location_str: str, country: str, current_temp: float, current_month: int) -> Optional[dict]:
    """
    Ask Gemini for soil info, market prices, and district summary.
    Hard 5-second timeout — returns None immediately if LLM is slow/down.
    Skips Ollama (too slow) — zone-based defaults are used as fallback.
    """
    if not GEMINI_KEY:
        return None  # No API key → use zone defaults instantly

    prompt = (
        f"Agricultural data for {location_str}. Temp: {current_temp}°C, Month: {current_month}. "
        f"Return ONLY compact JSON (no markdown):\n"
        f'{{"soil":{{"type":"<Clay/Loam/Sandy/Clay-Loam/Sandy-Loam>","ph":<4.5-8.5>,'
        f'"organic_matter":"<Low/Medium/High>","drainage":"<Poor/Medium/Good>"}},'
        f'"market_prices":{{"<crop1>":"<price+unit>","<crop2>":"<price+unit>","<crop3>":"<price+unit>"}},'
        f'"district_summary":"<1 sentence>","climate_zone":"<Tropical/Subtropical/Arid/Temperate/Mediterranean>"}}\n'
        f"Use local currency. soil.ph must be a number."
    )

    def _call_gemini():
        try:
            from google import genai as _genai
            client_g = _genai.Client(api_key=GEMINI_KEY)
            resp = client_g.models.generate_content(model="gemini-2.0-flash-lite", contents=prompt)
            return _extract_json_safe(resp.text.strip())
        except ImportError:
            try:
                import google.generativeai as genai  # type: ignore
                genai.configure(api_key=GEMINI_KEY)
                resp = genai.GenerativeModel("gemini-2.0-flash-lite").generate_content(prompt)
                return _extract_json_safe(resp.text.strip())
            except Exception:
                return None
        except Exception:
            return None

    try:
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
        with ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(_call_gemini)
            result = future.result(timeout=5)  # hard 5-second cap
            if result:
                logger.info("[DataAgent] Gemini enrich OK")
            return result
    except FuturesTimeout:
        logger.warning("[DataAgent] Gemini enrich timed out (>5s) — using zone defaults")
        return None
    except Exception as e:
        logger.warning(f"[DataAgent] Gemini enrich failed: {e}")
        return None

def _extract_json_safe(text: str) -> Optional[dict]:
    """Robustly extract JSON from LLM response."""
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
# 4. SOIL + MARKET DEFAULTS (location-aware)
# ═══════════════════════════════════════════════════════════════════════════════

_COUNTRY_CURRENCY = {
    "india": "₹", "pakistan": "PKR ", "bangladesh": "৳",
    "united states": "$", "canada": "CAD $", "mexico": "MXN $",
    "brazil": "R$", "argentina": "ARS $", "colombia": "COP $",
    "germany": "€", "france": "€", "italy": "€", "spain": "€",
    "united kingdom": "£", "ukraine": "UAH ", "russia": "₽",
    "china": "¥", "japan": "¥", "australia": "A$", "new zealand": "NZ$",
    "ghana": "GH₵", "nigeria": "₦", "kenya": "KES ", "ethiopia": "Br ",
    "egypt": "EGP ", "turkey": "₺", "indonesia": "Rp ", "vietnam": "₫",
    "thailand": "฿", "malaysia": "RM ", "philippines": "₱",
}

_ZONE_SOIL_DEFAULTS = {
    "Tropical":        {"type": "Clay-Loam",   "ph": 6.2, "organic_matter": "High",   "drainage": "Medium"},
    "Subtropical":     {"type": "Loam",         "ph": 7.0, "organic_matter": "Medium", "drainage": "Good"},
    "Arid":            {"type": "Sandy",        "ph": 7.8, "organic_matter": "Low",    "drainage": "Good"},
    "Mediterranean":   {"type": "Clay-Loam",   "ph": 7.2, "organic_matter": "Medium", "drainage": "Good"},
    "Temperate":       {"type": "Loam",         "ph": 6.5, "organic_matter": "Medium", "drainage": "Good"},
    "Continental":     {"type": "Clay-Loam",   "ph": 6.8, "organic_matter": "Medium", "drainage": "Medium"},
    "Temperate_Americas": {"type": "Loam",     "ph": 6.5, "organic_matter": "High",   "drainage": "Good"},
    "Tropical_Americas":  {"type": "Clay",     "ph": 5.8, "organic_matter": "High",   "drainage": "Poor"},
    "Subtropical_S":  {"type": "Loam",         "ph": 6.8, "organic_matter": "Medium", "drainage": "Good"},
    "Arid_Oceania":   {"type": "Sandy-Loam",  "ph": 7.5, "organic_matter": "Low",    "drainage": "Good"},
}

_ZONE_MARKET_TEMPLATES = {
    "India": [
        ("Wheat",   "₹2,400/quintal"), ("Rice",   "₹3,200/quintal"),
        ("Tomato",  "₹25-35/kg"),      ("Onion",  "₹20-30/kg"),
        ("Potato",  "₹15-22/kg"),      ("Maize",  "₹1,900/quintal"),
    ],
    "Tropical": [
        ("Rice",    "{cur}450/100kg"),  ("Maize",    "{cur}380/100kg"),
        ("Cassava", "{cur}120/50kg"),   ("Plantain", "{cur}80/dozen"),
        ("Tomato",  "{cur}60/kg"),      ("Cocoa",    "{cur}2200/50kg"),
    ],
    "Subtropical": [
        ("Wheat",   "{cur}280/quintal"),("Cotton",   "{cur}6000/quintal"),
        ("Tomato",  "{cur}35/kg"),      ("Onion",   "{cur}25/kg"),
        ("Rice",    "{cur}220/50kg"),   ("Sugarcane","{cur}350/tonne"),
    ],
    "Temperate": [
        ("Wheat",   "{cur}220/tonne"),  ("Corn/Maize","{cur}190/tonne"),
        ("Soybean", "{cur}400/tonne"),  ("Potato",   "{cur}180/tonne"),
        ("Canola",  "{cur}500/tonne"),  ("Barley",   "{cur}180/tonne"),
    ],
    "Temperate_Americas": [
        ("Corn",    "{cur}5.50/bushel"),("Soybeans", "{cur}13.00/bushel"),
        ("Wheat",   "{cur}5.80/bushel"),("Cotton",   "{cur}0.82/lb"),
        ("Sorghum", "{cur}5.00/bushel"),("Sunflower","{cur}0.22/lb"),
    ],
    "Mediterranean": [
        ("Olive",   "{cur}2.50/kg"),   ("Tomato",  "{cur}0.60/kg"),
        ("Wheat",   "{cur}220/tonne"), ("Grape",   "{cur}0.80/kg"),
        ("Orange",  "{cur}0.40/kg"),   ("Almonds", "{cur}5.00/kg"),
    ],
    "Arid": [
        ("Dates",   "{cur}4.00/kg"),   ("Wheat",   "{cur}350/tonne"),
        ("Cotton",  "{cur}1.20/kg"),   ("Sesame",  "{cur}2.50/kg"),
        ("Sorghum", "{cur}280/tonne"), ("Millet",  "{cur}260/tonne"),
    ],
}

def _default_market_prices(country: str, zone: str) -> dict:
    """Return currency-correct default market prices."""
    cur = "USD "
    for k, v in _COUNTRY_CURRENCY.items():
        if k in country.lower():
            cur = v
            break

    # India special case
    if country.lower() in ("india", "भारत"):
        return {k: v for k, v in _ZONE_MARKET_TEMPLATES["India"]}

    template_key = zone if zone in _ZONE_MARKET_TEMPLATES else "Subtropical"
    template = _ZONE_MARKET_TEMPLATES.get(template_key, _ZONE_MARKET_TEMPLATES["Subtropical"])
    return {k: v.replace("{cur}", cur) for k, v in template}


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SEASON DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _guess_season(month: int, country: str) -> str:
    country_lower = country.lower()
    # Southern hemisphere flip
    if any(c in country_lower for c in ["australia", "brazil", "argentina", "south africa", "new zealand"]):
        month = ((month + 5) % 12) + 1
    if country_lower in ["india", "bangladesh", "pakistan", "nepal", "sri lanka"]:
        if month in [6, 7, 8, 9, 10]:  return "Kharif"
        if month in [11, 12, 1, 2, 3]: return "Rabi"
        return "Zaid"
    if month in [3, 4, 5]:   return "Spring"
    if month in [6, 7, 8]:   return "Summer"
    if month in [9, 10, 11]: return "Autumn"
    return "Winter"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def gather_location_data(
    country: str,
    state: str,
    district: str,
    lat: float,
    lon: float,
    month: Optional[int] = None,
    state_code: Optional[str] = None,
) -> dict:
    """
    Gather real agricultural data for any location worldwide.

    Data sources (in priority order):
      1. Open-Meteo API → real current temperature, rainfall, wind, UV
      2. Zone climatology (India: history.py; World: inline table) → 6-month forecast
         anchored to live temperature so each district is accurate
      3. Gemini / Ollama LLM → soil type, pH, market prices, district summary
      4. Zone-aware defaults → when LLM is unavailable

    Returns dict with: current, forecast_6month, soil, season, climate_zone,
                       market_prices, district_summary
    """
    current_month = month or datetime.datetime.now().month
    location_str  = f"{district}, {state}, {country}"
    logger.info(f"[DataAgent] Gathering data for: {location_str} ({lat:.3f}, {lon:.3f})")

    # ── 1. Live weather from Open-Meteo ──────────────────────────────────────
    live_wx = _fetch_openmeteo_current(lat, lon)
    if live_wx:
        logger.info(f"[DataAgent] Live weather: {live_wx['temperature_c']}°C, "
                    f"rain7d={live_wx['rainfall_7d_mm']}mm, humidity={live_wx['humidity_pct']}%")
        current = live_wx
    else:
        logger.warning("[DataAgent] Open-Meteo unavailable, using zone-based estimate")
        # Build a reasonable current from zone climatology
        world_zone = _get_world_zone(country, lat)
        zc = (_ZONE_CLIMATE.get(world_zone) or _ZONE_CLIMATE["Subtropical"]).get(current_month, {})
        t_avg = zc.get("temp", 25)
        current = {
            "temperature_c":  t_avg,
            "temp_max_c":     zc.get("temp_max", t_avg + 6),
            "temp_min_c":     zc.get("temp_min", t_avg - 6),
            "humidity_pct":   zc.get("hum", 65),
            "soil_temp_c":    round(t_avg - 2, 1),
            "rainfall_7d_mm": round(zc.get("rain", 60) / 4, 1),
            "wind_kmh":       12.0,
            "uv_index":       6.0,
            "feels_like_c":   round(t_avg + 2, 1),
        }

    # ── 2. 6-month forecast from climatology + live anchor ───────────────────
    # Use state_code for India zone mapping; for other countries use country name
    forecast_6month = _build_forecast_6month(
        country=country,
        state_code=state_code,
        lat=lat,
        live_temp_avg=current["temperature_c"],
        current_month=current_month,
    )

    # ── 3. LLM enrichment for soil + market prices ───────────────────────────
    llm_data = _llm_enrich(location_str, country, current["temperature_c"], current_month)

    # ── 4. Assemble final result ─────────────────────────────────────────────
    world_zone = _get_world_zone(country, lat)

    # Soil
    if llm_data and llm_data.get("soil") and isinstance(llm_data["soil"].get("ph"), (int, float)):
        soil = llm_data["soil"]
        # Ensure all fields present
        soil.setdefault("type",          _ZONE_SOIL_DEFAULTS.get(world_zone, {}).get("type", "Loam"))
        soil.setdefault("organic_matter","Medium")
        soil.setdefault("drainage",      "Medium")
    else:
        soil = dict(_ZONE_SOIL_DEFAULTS.get(world_zone, {"type": "Loam", "ph": 7.0,
                                                          "organic_matter": "Medium", "drainage": "Good"}))

    # Market prices
    if llm_data and llm_data.get("market_prices") and len(llm_data["market_prices"]) >= 2:
        market_prices = llm_data["market_prices"]
    else:
        market_prices = _default_market_prices(country, world_zone)

    # Climate zone label
    if llm_data and llm_data.get("climate_zone"):
        climate_zone = llm_data["climate_zone"]
    else:
        zone_labels = {
            "Tropical": "Tropical", "Subtropical": "Subtropical",
            "Arid": "Arid", "Mediterranean": "Mediterranean",
            "Temperate": "Temperate", "Continental": "Continental",
            "Temperate_Americas": "Temperate", "Tropical_Americas": "Tropical",
            "Subtropical_S": "Subtropical", "Arid_Oceania": "Semi-Arid",
        }
        climate_zone = zone_labels.get(world_zone, "Subtropical")

    # District summary
    if llm_data and llm_data.get("district_summary"):
        district_summary = llm_data["district_summary"]
    else:
        district_summary = (
            f"{district} is an agricultural district in {state}, {country}. "
            f"The region experiences {climate_zone.lower()} climate conditions "
            f"with current temperature around {current['temperature_c']}°C."
        )

    # Season
    season = _guess_season(current_month, country)

    logger.info(f"[DataAgent] Complete: temp={current['temperature_c']}°C, "
                f"soil={soil['type']}, zone={climate_zone}, crops={len(market_prices)}")

    return {
        "current":         current,
        "forecast_6month": forecast_6month,
        "soil":            soil,
        "season":          season,
        "climate_zone":    climate_zone,
        "market_prices":   market_prices,
        "district_summary":district_summary,
    }
