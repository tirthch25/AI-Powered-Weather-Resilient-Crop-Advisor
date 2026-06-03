"""
Historical Weather Reader for Indian Agro-Climatic Zones

Reads the generated historical_weather.csv and provides:
  - Zone-level climatological baselines (10-year monthly averages)
  - Region -> Zone mapping using state prefix codes
  - Seasonal baselines (avg temp, monthly rainfall, humidity) used by:
      • forecast.py  (climatology forecast fallback)
      • recommender.py  (weather_conditions humidity)
      • app.py  (monthly climate chart for the UI)

Zone reference:
  North    : UP, Punjab, Haryana, Rajasthan, Himachal, Uttarakhand, Delhi, J&K
  South    : Karnataka, Tamil Nadu, AP, Telangana, Kerala
  East     : West Bengal, Bihar, Odisha, Jharkhand
  West     : Maharashtra, Gujarat
  Central  : MP, Chhattisgarh
  Northeast: Assam, Arunachal, Manipur, Meghalaya, Sikkim, Nagaland, Tripura
"""

import csv
import os
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# -- Default data path (relative to working directory = agri_crop_recommendation/)
_DEFAULT_CSV = os.path.join("data", "weather", "zone", "historical_weather.csv")

# -- Region-prefix -> Zone mapping (humidity lookup only; temperature is 100% live API)
_STATE_TO_ZONE: Dict[str, str] = {
    # North (Indo-Gangetic plains)
    "UP": "North", "PB": "North", "HR": "North", "HP": "North",
    "UK": "North", "DL": "North", "JK": "North", "RJ": "North",
    "CH": "North",    # Chandigarh UT (North plains city)
    # Highland / Alpine (Ladakh UT cold desert at 3500+ m)
    "LA": "Highland",
    # South
    "KA": "South", "TN": "South", "AP": "South", "TS": "South", "KL": "South",
    "TL": "South",    # Telangana (regions.json uses TL, not TS)
    "PY": "South",    # Puducherry UT (coastal Tamil Nadu climate)
    "AN": "South",    # Andaman & Nicobar Islands (tropical island, South humidity)
    # East
    "WB": "East",  "BR": "East",  "OD": "East",  "JH": "East",
    # West (coastal Maharashtra + Gujarat)
    "MH": "West",  "GJ": "West",
    "GA": "West",     # Goa (coastal West zone)
    # Central
    "MP": "Central", "CG": "Central",
    # Northeast
    "AS": "Northeast", "AR": "Northeast", "MN": "Northeast", "MZ": "Northeast",
    "MG": "Northeast", "SK": "Northeast", "NL": "Northeast", "TR": "Northeast",
    "ML": "Northeast",  # Meghalaya (was missing, defaulted to wrong North zone)
}

# ── Inline Highland/Alpine climate table (Leh–Ladakh cold desert, ~3 500 m) ───
# Source: IMD / World Meteorological Organization station data for Leh.
# month → { temperature (°C avg), rainfall (mm/month), humidity (%) }
_HIGHLAND_CLIMATE: Dict[int, Dict[str, float]] = {
    1:  {"temperature": -7.0, "rainfall":  9.0, "humidity": 45.0},
    2:  {"temperature": -4.5, "rainfall":  9.5, "humidity": 42.0},
    3:  {"temperature":  1.5, "rainfall": 11.0, "humidity": 37.0},
    4:  {"temperature":  7.5, "rainfall":  8.0, "humidity": 30.0},
    5:  {"temperature": 12.5, "rainfall":  7.5, "humidity": 27.0},
    6:  {"temperature": 17.0, "rainfall":  7.0, "humidity": 25.0},
    7:  {"temperature": 19.5, "rainfall": 20.0, "humidity": 40.0},
    8:  {"temperature": 18.5, "rainfall": 17.5, "humidity": 42.0},
    9:  {"temperature": 13.5, "rainfall": 11.5, "humidity": 36.0},
    10: {"temperature":  5.5, "rainfall":  7.5, "humidity": 32.0},
    11: {"temperature": -2.5, "rainfall":  7.5, "humidity": 40.0},
    12: {"temperature": -6.5, "rainfall":  9.0, "humidity": 45.0},
}

# ── District-level overrides for Maharashtra sub-zones ───────────────────────
# Marathwada: hot semi-arid plateau (avg +3°C vs coastal MH, 30% less rainfall)
# IDs must exactly match regions.json region_id values
_MARATHWADA_DISTRICTS = {
    "MH_LATUR", "MH_SOLAPUR", "MH_OSMANABAD", "MH_NANDED",
    "MH_PARBHANI", "MH_HINGOLI", "MH_BEED", "MH_JALNA",
    "MH_CHHATRAPATI_SAMBHAJINAGAR",  # Aurangabad (renamed)
}
# Vidarbha: extreme heat zone (avg +4°C vs coastal MH, semi-arid)
# Note: MH_BHANDARA does NOT exist in regions.json — omitted to avoid silent fallback
_VIDARBHA_DISTRICTS = {
    "MH_NAGPUR", "MH_AMRAVATI", "MH_AKOLA", "MH_WASHIM",
    "MH_YAVATMAL", "MH_WARDHA", "MH_CHANDRAPUR", "MH_GADCHIROLI",
    "MH_GONDIA", "MH_BULDHANA",
}

# ── Monthly diurnal range (daily_max - daily_min) per zone ────────────────────
# Used to compute temp_max and temp_min for the climate chart range band.
# Values sourced from IMD station data (°C swing between day-max and night-min).
# Summer months have large swings (low humidity, clear skies).
# Monsoon months have small swings (cloud cover, high humidity).
_ZONE_DIURNAL_RANGE: Dict[str, Dict[int, float]] = {
    "North": {
        1: 12.0, 2: 13.0, 3: 14.0, 4: 14.0, 5: 13.0, 6: 11.0,
        7:  8.0, 8:  8.0, 9:  9.0, 10: 12.0, 11: 12.0, 12: 11.0,
    },
    "South": {
        1: 10.0, 2: 10.0, 3: 11.0, 4: 11.0, 5: 12.0, 6: 10.0,
        7:  8.0, 8:  8.0, 9:  8.0, 10:  8.0, 11:  9.0, 12: 10.0,
    },
    "East": {
        1: 11.0, 2: 12.0, 3: 13.0, 4: 13.0, 5: 12.0, 6:  9.0,
        7:  7.0, 8:  7.0, 9:  8.0, 10: 10.0, 11: 11.0, 12: 11.0,
    },
    "West": {
        # Thane/Pune May: highs 35-43°C, lows 24-27°C → swing ~14°C
        1: 11.0, 2: 12.0, 3: 13.0, 4: 14.0, 5: 14.0, 6:  9.0,
        7:  7.0, 8:  7.0, 9:  8.0, 10: 10.0, 11: 11.0, 12: 11.0,
    },
    "Central": {
        1: 13.0, 2: 14.0, 3: 15.0, 4: 15.0, 5: 14.0, 6: 10.0,
        7:  7.0, 8:  7.0, 9:  8.0, 10: 12.0, 11: 13.0, 12: 12.0,
    },
    "Northeast": {
        1: 10.0, 2: 11.0, 3: 12.0, 4: 10.0, 5:  9.0, 6:  7.0,
        7:  6.0, 8:  6.0, 9:  7.0, 10:  9.0, 11: 10.0, 12: 10.0,
    },
    "Marathwada": {
        1: 13.0, 2: 14.0, 3: 15.0, 4: 15.0, 5: 15.0, 6: 10.0,
        7:  7.0, 8:  7.0, 9:  8.0, 10: 12.0, 11: 13.0, 12: 12.0,
    },
    "Vidarbha": {
        1: 13.0, 2: 14.0, 3: 16.0, 4: 16.0, 5: 15.0, 6: 10.0,
        7:  7.0, 8:  7.0, 9:  8.0, 10: 12.0, 11: 13.0, 12: 12.0,
    },
    "Highland": {
        1: 14.0, 2: 14.0, 3: 15.0, 4: 16.0, 5: 17.0, 6: 14.0,
        7: 10.0, 8: 10.0, 9: 12.0, 10: 14.0, 11: 14.0, 12: 14.0,
    },
}


def _get_diurnal_range(zone: str, month: int) -> float:
    """Return the monthly diurnal range (max - min) for a zone."""
    zone_ranges = _ZONE_DIURNAL_RANGE.get(zone, _ZONE_DIURNAL_RANGE.get("North", {}))
    return zone_ranges.get(month, 12.0)


def get_zone_for_region(region_id: Optional[str]) -> str:
    """
    Derive the agro-climatic zone from a region_id like 'UP_LUCKNOW'.

    Returns one of: North, South, East, West, Central, Northeast,
                    Highland, Marathwada, Vidarbha.

    Sub-zone overrides (checked before state prefix):
      - 'Highland'    for all Ladakh districts (LA_*)
      - 'Marathwada'  for Latur, Solapur, Osmanabad, Nanded, etc.
      - 'Vidarbha'    for Nagpur, Amravati, Akola, Wardha, etc.
      - 'West'        for all other Maharashtra / Gujarat districts

    Defaults to 'North' if unknown.
    """
    if not region_id:
        return "North"
    # Check Marathwada sub-zone first
    if region_id in _MARATHWADA_DISTRICTS:
        return "Marathwada"
    # Check Vidarbha sub-zone
    if region_id in _VIDARBHA_DISTRICTS:
        return "Vidarbha"
    state_code = region_id.split("_")[0].upper()
    return _STATE_TO_ZONE.get(state_code, "North")


# ── In-memory cache keyed by CSV path ─────────────────────────────────────────
_cache: Dict[str, Dict] = {}


def _load_csv(csv_path: str) -> Dict:
    """
    Load and aggregate historical_weather.csv into nested lookup:
      { zone: { month: { 'temperature': float, 'rainfall': float, 'humidity': float } } }
    Averages over all years in the file.
    """
    if csv_path in _cache:
        return _cache[csv_path]

    # Accumulator: { zone: { month: [values] } }
    acc: Dict[str, Dict[int, Dict[str, list]]] = {}

    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                zone  = row["zone"]
                month = int(row["month"])
                temp  = float(row["temperature"])
                rain  = float(row["rainfall"])
                hum   = float(row["humidity"])

                acc.setdefault(zone, {}).setdefault(month, {
                    "temperature": [], "rainfall": [], "humidity": []
                })
                acc[zone][month]["temperature"].append(temp)
                acc[zone][month]["rainfall"].append(rain)
                acc[zone][month]["humidity"].append(hum)

        # Average over years
        result: Dict[str, Dict[int, Dict[str, float]]] = {}
        for zone, months in acc.items():
            result[zone] = {}
            for month, vals in months.items():
                result[zone][month] = {
                    "temperature": round(sum(vals["temperature"]) / len(vals["temperature"]), 1),
                    "rainfall":    round(sum(vals["rainfall"])    / len(vals["rainfall"]),    1),
                    "humidity":    round(sum(vals["humidity"])     / len(vals["humidity"]),    1),
                }

        _cache[csv_path] = result
        logger.info(f"Historical weather loaded: {len(result)} zones from {csv_path}")
        return result

    except FileNotFoundError:
        logger.warning(f"Historical weather CSV not found at {csv_path}. Run scripts/generate_historical_weather.py first.")
        return {}
    except Exception as e:
        logger.error(f"Error loading historical weather: {e}")
        return {}


def get_monthly_climate(
    zone: str,
    month: int,
    csv_path: str = _DEFAULT_CSV
) -> Dict[str, float]:
    """
    Return average temperature (°C), monthly rainfall (mm), and humidity (%)
    for a zone and calendar month (1–12).

    Returns sensible defaults if data is unavailable.
    """
    data = _load_csv(csv_path)
    fallback = {"temperature": 28.0, "temp_max": 35.0, "temp_min": 21.0, "rainfall": 80.0, "humidity": 60.0}
    if not data:
        return fallback

    def _with_range(result: dict, z: str, m: int) -> dict:
        """Enrich a climate dict with temp_max and temp_min from the diurnal range table."""
        half = _get_diurnal_range(z, m) / 2.0
        t = result["temperature"]
        result["temp_max"] = round(t + half, 1)
        result["temp_min"] = round(t - half, 1)
        return result

    # ── Highland / Alpine zone (Leh–Ladakh) — served from inline table ──────
    # The CSV only covers broad plains zones.  Highland uses IMD station data
    # for Leh at 3 524 m; returning directly avoids the ~40°C plains bias.
    if zone == "Highland":
        base = dict(_HIGHLAND_CLIMATE.get(month, _HIGHLAND_CLIMATE[6]))
        return _with_range(base, "Highland", month)

    # ── Sub-zone derivation for Marathwada / Vidarbha ────────────────────────
    # These sub-zones are NOT in the historical CSV (which only has broad zones).
    # We derive their climate from the "West" zone with calibrated offsets:
    #   Marathwada : +3°C warmer, 30% drier, 8% lower humidity
    #   Vidarbha   : +4°C warmer, 25% drier, 5% lower humidity
    if zone == "Marathwada":
        base = data.get("West", data.get("North", {})).get(month, fallback)
        result = {
            "temperature": round(base["temperature"] + 3.0, 1),
            "rainfall":    round(base["rainfall"] * 0.70, 1),
            "humidity":    round(max(base["humidity"] - 8.0, 25.0), 1),
        }
        return _with_range(result, "Marathwada", month)
    if zone == "Vidarbha":
        base = data.get("West", data.get("North", {})).get(month, fallback)
        result = {
            "temperature": round(base["temperature"] + 4.0, 1),
            "rainfall":    round(base["rainfall"] * 0.75, 1),
            "humidity":    round(max(base["humidity"] - 5.0, 28.0), 1),
        }
        return _with_range(result, "Vidarbha", month)

    zone_data = data.get(zone, data.get("North", {}))
    raw = zone_data.get(month, fallback)
    return _with_range(dict(raw), zone, month)


def get_seasonal_climate(
    zone: str,
    season: str,
    csv_path: str = _DEFAULT_CSV
) -> Dict[str, float]:
    """
    Return aggregated temperature (avg), total rainfall (mm), and mean
    humidity (%) for an agricultural season.

    Season → Calendar months:
        Kharif : June–October  (6–10)
        Rabi   : November–March (11, 12, 1, 2, 3)
        Zaid   : April–May     (4, 5)
    """
    season_months = {
        "Kharif": [6, 7, 8, 9, 10],
        "Rabi":   [11, 12, 1, 2, 3],
        "Zaid":   [4, 5],
    }
    months = season_months.get(season, [6, 7, 8, 9, 10])

    temps, rains, hums = [], [], []
    for m in months:
        clim = get_monthly_climate(zone, m, csv_path)
        temps.append(clim["temperature"])
        rains.append(clim["rainfall"])
        hums.append(clim["humidity"])

    return {
        "avg_temperature":   round(sum(temps) / len(temps), 1),
        "total_rainfall_mm": round(sum(rains), 1),
        "avg_humidity":      round(sum(hums)  / len(hums),  1),
    }


def get_climate_for_region(
    region_id: Optional[str],
    season: str,
    csv_path: str = _DEFAULT_CSV
) -> Dict[str, float]:
    """
    Convenience: derive zone from region_id and return seasonal climate.
    """
    zone = get_zone_for_region(region_id)
    return get_seasonal_climate(zone, season, csv_path)
