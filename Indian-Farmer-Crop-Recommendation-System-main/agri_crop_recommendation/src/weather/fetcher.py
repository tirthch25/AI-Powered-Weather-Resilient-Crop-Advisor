"""
Weather Data Fetcher — Open-Meteo API + Historical Fallback

Fetches the 16-day live forecast from Open-Meteo.
If the API is unavailable, falls back to zone-based historical averages
so the recommendation engine never crashes due to network issues.

Additionally enriches the returned DataFrame with:
  - temp_avg   (mean of max and min)
  - temp_range (daily swing)
  - humidity   (from historical data, because Open-Meteo free tier doesn't include it)
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def fetch_weather(
    latitude: float,
    longitude: float,
    days: int = 16,
    region_id: Optional[str] = None,
    season: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch weather data and return an enriched DataFrame.

    Columns returned:
        date, temp_max, temp_min, temp_avg, temp_range, rainfall, humidity

    Falls back to historical zone-based climatology when the API is unavailable.

    Args:
        latitude:  Decimal latitude.
        longitude: Decimal longitude.
        days:      Forecast horizon (default 16).
        region_id: Optional region ID used to determine agro-climatic zone
                   for humidity enrichment and fallback.
        season:    Optional season name used for historical lookup.
    """
    df = _fetch_from_api(latitude, longitude, days)

    if df is None:
        logger.warning("Open-Meteo API unavailable — using historical baseline as weather data.")
        df = _historical_baseline_as_weather(region_id, season, days)
    else:
        # Enrich with derived columns
        df = _enrich(df, region_id, season)

    return df


# ── Private helpers ────────────────────────────────────────────────────────────

def _fetch_from_api(latitude: float, longitude: float, days: int) -> Optional[pd.DataFrame]:
    """Call Open-Meteo and return a raw DataFrame, or None on failure.

    Requests 14 days of past data + forecast_days of future data so the
    returned DataFrame has ~30 rows — enough for the LSTM's 30-day lookback
    without any padding.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
        ],
        "past_days":     14,      # ← include last 14 days of observed data
        "forecast_days": days,    # ← still request the full forward window
        "timezone": "auto",
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        daily = response.json()["daily"]
        df = pd.DataFrame({
            "date":     daily["time"],
            "temp_max": daily["temperature_2m_max"],
            "temp_min": daily["temperature_2m_min"],
            "rainfall": daily["precipitation_sum"],
        })
        # Fill any null API values with 0
        df["rainfall"] = df["rainfall"].fillna(0.0)
        df["temp_max"] = df["temp_max"].ffill().fillna(30.0)
        df["temp_min"] = df["temp_min"].ffill().fillna(20.0)
        return df
    except Exception as e:
        logger.error(f"Open-Meteo API error: {e}")
        return None


def _enrich(df: pd.DataFrame, region_id: Optional[str], season: Optional[str]) -> pd.DataFrame:
    """
    Add derived columns and humidity from historical data.
    """
    df = df.copy()

    # Derived temperature columns
    df["temp_avg"]   = (df["temp_max"] + df["temp_min"]) / 2
    df["temp_range"] = df["temp_max"] - df["temp_min"]

    # Dry spell tracking
    df["dry_spell_days"] = _calculate_dry_spell(df["rainfall"])

    # Humidity: use today's month to pick the right historical monthly value
    try:
        current_month = datetime.now().month
        from src.weather.history import get_zone_for_region, get_monthly_climate
        zone = get_zone_for_region(region_id)
        clim = get_monthly_climate(zone, current_month)
        base_humidity = clim["humidity"]
    except Exception:
        base_humidity = 60.0

    # Add some daily variation correlated with rainfall
    rng = np.random.default_rng(seed=42)
    rain_bonus = np.clip(df["rainfall"] * 0.15, 0, 12)
    noise = rng.normal(0, 3.0, len(df))
    df["humidity"] = np.clip(base_humidity + rain_bonus + noise, 15, 98).round(1)

    return df


def _calculate_dry_spell(rainfall_series: pd.Series) -> pd.Series:
    """Count consecutive days with < 2 mm rainfall up to each day."""
    dry_spell = []
    count = 0
    for r in rainfall_series:
        if r < 2.0:
            count += 1
        else:
            count = 0
        dry_spell.append(count)
    return pd.Series(dry_spell, index=rainfall_series.index)


def _historical_baseline_as_weather(
    region_id: Optional[str],
    season: Optional[str],
    days: int
) -> pd.DataFrame:
    """
    Build a synthetic weather DataFrame from historical zone averages.
    Used when the Open-Meteo API is completely unavailable.
    """
    current_month = datetime.now().month
    det_season = season

    # Determine season from month if not supplied
    if not det_season:
        if current_month in (6, 7, 8, 9, 10):
            det_season = "Kharif"
        elif current_month in (4, 5):
            det_season = "Zaid"
        else:
            det_season = "Rabi"

    try:
        from src.weather.history import get_zone_for_region, get_monthly_climate
        zone = get_zone_for_region(region_id)
        clim = get_monthly_climate(zone, current_month)
        avg_temp = clim["temperature"]
        daily_rain = clim["rainfall"] / 30.0   # monthly → daily
        humidity = clim["humidity"]
    except Exception:
        avg_temp   = 28.0
        daily_rain = 3.0
        humidity   = 60.0

    dates = [
        (datetime.today() + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(days)
    ]

    # Simple sinusoidal daily variation around monthly average
    rng = np.random.default_rng(seed=99)
    temp_max = avg_temp + 4.0 + rng.normal(0, 1.5, days)
    temp_min = avg_temp - 5.0 + rng.normal(0, 1.2, days)
    rainfall = np.maximum(
        rng.exponential(daily_rain, days) if daily_rain > 0.1 else np.zeros(days),
        0
    ).round(1)
    rain_bonus = np.clip(rainfall * 0.15, 0, 12)
    hum = np.clip(humidity + rain_bonus + rng.normal(0, 3.0, days), 15, 98).round(1)

    df = pd.DataFrame({
        "date":     dates,
        "temp_max": temp_max.round(1),
        "temp_min": temp_min.round(1),
        "temp_avg": ((temp_max + temp_min) / 2).round(1),
        "temp_range": (temp_max - temp_min).round(1),
        "rainfall": rainfall,
        "humidity": hum,
    })
    df["dry_spell_days"] = _calculate_dry_spell(df["rainfall"])
    return df
