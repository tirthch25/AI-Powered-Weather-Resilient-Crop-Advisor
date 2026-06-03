"""
Medium-Range Weather Forecasting (ML-Enhanced)

Provides weather forecasts for agricultural planning:
- Uses ML ensemble (LSTM + XGBoost) when trained models are available
- Falls back to climatology-based estimation otherwise

This module replaces the original simple climatology extrapolation
with ML-powered predictions while maintaining backward compatibility.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def forecast_days_17_90(weather_df: pd.DataFrame, planning_days: int = 90, region_id: str = None) -> Dict:
    """
    Generate medium-range weather forecast for agricultural planning.
    
    Attempts ML-based forecasting first, falls back to climatology.
    
    Args:
        weather_df: Recent weather DataFrame with temp_avg, rainfall, dry_spell_days
        planning_days: Planning horizon in days
        region_id: Region ID for loading ML models (optional)
        
    Returns:
        Dictionary with expected temperature, rainfall, and risk levels
    """
    ml_forecast = None
    
    # --- Try ML-based forecasting ---
    if region_id:
        ml_forecast = _get_ml_forecast(weather_df, planning_days, region_id)
    
    if ml_forecast:
        logger.info(f"Using ML forecast for region {region_id}")
        ml_sum = ml_forecast.get('summary', {})
        return {
            "expected_avg_temp":    ml_sum.get('avg_temp', 25.0),
            "expected_temp_max":    ml_sum.get('avg_temp_max', ml_sum.get('avg_temp', 30.0)),
            "expected_temp_min":    ml_sum.get('avg_temp_min', ml_sum.get('avg_temp', 20.0)),
            "expected_rainfall_mm": ml_sum.get('total_rainfall', 100.0),
            "dry_spell_risk":       _calculate_dry_spell_risk(ml_forecast),
            "forecast_source":      ml_forecast.get('model_used', 'ML ensemble'),
            "confidence":           ml_forecast.get('confidence', 'medium'),
            "daily_predictions":    ml_forecast.get('predictions', []),
            "predictions":          ml_forecast.get('predictions', []),
            "summary":              ml_sum,
            "ml_summary":           ml_sum,
        }
    
    # --- Fallback: Climatology-based estimation (uses historical zone data) ---
    logger.info("Using historical/climatology-based forecast (no ML models available)")
    return _climatology_forecast(weather_df, planning_days, region_id=region_id)


def _get_ml_forecast(weather_df: pd.DataFrame, planning_days: int, region_id: str) -> Optional[Dict]:
    """
    ML-based weather forecast using LSTM → XGBoost → None fallback chain.

    Tries LSTM first (most accurate), then XGBoost, then returns None
    so the caller falls back to climatology-based estimation.

    Both models are loaded lazily and cached in module-level variables.
    Failures are caught gracefully so the app never crashes due to missing models.
    """
    # ── Try LSTM ──────────────────────────────────────────────────────────────
    lstm_model = _load_lstm_model()
    if lstm_model is not None:
        try:
            result = lstm_model.predict(weather_df, district_id=region_id,
                                        horizon=min(7, planning_days))
            logger.info(f"LSTM forecast obtained for {region_id}")
            return result
        except Exception as e:
            logger.warning(f"LSTM forecast failed for {region_id}: {e}")

    # ── Try XGBoost ───────────────────────────────────────────────────────────
    xgb_model = _load_xgboost_model()
    if xgb_model is not None:
        try:
            result = xgb_model.predict(weather_df, district_id=region_id,
                                       horizon=min(7, planning_days))
            logger.info(f"XGBoost forecast obtained for {region_id}")
            return result
        except Exception as e:
            logger.warning(f"XGBoost forecast failed for {region_id}: {e}")

    return None


# ── Lazy model loaders (module-level cache) ───────────────────────────────────
_lstm_model_cache   = None
_xgboost_model_cache = None


def _load_lstm_model():
    """Load and cache the LSTM weather model. Returns None if not available."""
    global _lstm_model_cache
    if _lstm_model_cache is not None:
        return _lstm_model_cache
    try:
        from src.ml.lstm_weather import LSTMWeatherForecaster
        _lstm_model_cache = LSTMWeatherForecaster.load()
        logger.info("LSTM weather model loaded successfully.")
        return _lstm_model_cache
    except Exception as e:
        logger.debug(f"LSTM model not available: {e}")
        return None


def _load_xgboost_model():
    """Load and cache the XGBoost weather model. Returns None if not available."""
    global _xgboost_model_cache
    if _xgboost_model_cache is not None:
        return _xgboost_model_cache
    try:
        from src.ml.xgboost_weather import XGBoostWeatherForecaster
        _xgboost_model_cache = XGBoostWeatherForecaster.load()
        logger.info("XGBoost weather model loaded successfully.")
        return _xgboost_model_cache
    except Exception as e:
        logger.debug(f"XGBoost model not available: {e}")
        return None



def _climatology_forecast(weather_df: pd.DataFrame, planning_days: int, region_id: str = None) -> Dict:
    """
    Live-API-first forecast (fallback when no ML models).

    Temperature: 100% from Open-Meteo live API using the district's exact lat/lon.
      Zone averages are NOT blended in -- they produce wrong values for high-altitude
      districts (Leh, Tawang, Gangtok) and any district that differs from its zone mean.

    Humidity:  from historical zone data (Open-Meteo free tier omits it).
    Rainfall:  from live API, zone seasonal total used only as a sanity-check floor.
    """
    current_month = datetime.now().month  # noqa: F841

    # ── Humidity from zone data (only field not in the free API) ─────────────
    hist_hum  = None
    hist_rain = None
    try:
        from src.weather.history import get_zone_for_region, get_seasonal_climate
        from src.utils.seasons import detect_season
        from datetime import datetime as _dt

        zone   = get_zone_for_region(region_id)
        season = detect_season(_dt.now(), region_id)
        seas   = get_seasonal_climate(zone, season)
        hist_hum  = seas["avg_humidity"]

        # Seasonal rainfall total -- used only for rainfall floor, never temperature
        hist_rain = seas["total_rainfall_mm"]
        season_lengths = {"Kharif": 153, "Rabi": 151, "Zaid": 61}
        season_days    = season_lengths.get(season, 120)
        if planning_days != season_days:
            hist_rain = round(hist_rain * (planning_days / season_days), 1)

        logger.info(
            f"Zone={zone} season={season}: hum={hist_hum}%  "
            f"(temperature is 100%% live API -- zone temp NOT blended)"
        )
    except Exception as e:
        logger.debug(f"Zone humidity lookup failed: {e}")

    # ── Live API signals -- these ARE the district's real temperatures ────────
    avg_temp_api     = float(weather_df["temp_avg"].mean())
    avg_temp_max_api = float(weather_df["temp_max"].mean())
    avg_temp_min_api = float(weather_df["temp_min"].mean())
    avg_daily_rain   = float(weather_df["rainfall"].mean())
    dry_spell_risk   = int(weather_df["dry_spell_days"].max())

    if avg_daily_rain < 0.5:
        avg_daily_rain = 1.5  # conservative climatological floor

    # ── Temperature: 100% live API, no zone blending ─────────────────────────
    # Open-Meteo is called with each district's actual lat/lon, so the returned
    # temperature already accounts for altitude, coastal proximity, etc.
    # Blending in zone averages introduces error for any district that diverges
    # from its zone mean (Leh, Tawang, Gangtok, coastal vs inland MH, ...).
    temp_trend    = (
        weather_df["temp_avg"].iloc[-5:].mean() - weather_df["temp_avg"].iloc[:5].mean()
    )
    temp_adjust   = 1.0 if temp_trend > 0 else (-1.0 if temp_trend < 0 else 0.0)
    expected_temp = round(avg_temp_api + temp_adjust, 1)

    # ── Rainfall: live API extrapolated; zone total as floor only ─────────────
    live_rain_proj = round(avg_daily_rain * planning_days, 1)
    if hist_rain is not None and live_rain_proj < 0.5:
        # API shows near-zero during a clearly wet season -- use zone floor
        expected_rain = hist_rain
    else:
        expected_rain = live_rain_proj

    # ── Humidity: only zone-sourced field (not in Open-Meteo free tier) ───────
    if hist_hum is not None:
        expected_hum    = hist_hum
        forecast_source = "live_api"
        confidence      = "high"
    else:
        expected_hum = (
            float(weather_df["humidity"].mean())
            if "humidity" in weather_df.columns else 65.0
        )
        forecast_source = "live_api"
        confidence      = "medium"

    # Build daily predictions from live weather_df (Days 1-16, actual API data)
    daily_predictions = []
    for _, row in weather_df.iterrows():
        daily_predictions.append({
            "temp_max": float(row.get("temp_max", avg_temp_max_api)),
            "temp_min": float(row.get("temp_min", avg_temp_min_api)),
            "rainfall": float(row.get("rainfall", 0)),
        })

    return {
        "expected_avg_temp":     expected_temp,
        "expected_temp_max":     round(avg_temp_max_api, 1),
        "expected_temp_min":     round(avg_temp_min_api, 1),
        "expected_rainfall_mm":  expected_rain,
        "expected_humidity":     round(expected_hum, 1),
        "dry_spell_risk": (
            "High"     if dry_spell_risk > 7 else
            "Moderate" if dry_spell_risk > 4 else
            "Low"
        ),
        "forecast_source": forecast_source,
        "confidence":      confidence,
        "daily_predictions": daily_predictions,
        "predictions":       daily_predictions,   # alias for risk engine
        "summary": {
            "avg_temp":      expected_temp,
            "avg_temp_max":  round(avg_temp_max_api, 1),
            "avg_temp_min":  round(avg_temp_min_api, 1),
            "total_rainfall": expected_rain,
            "avg_humidity":  round(expected_hum, 1),
        },
        "ml_summary": {}
    }




def _calculate_dry_spell_risk(forecast: Dict) -> str:
    """Calculate dry spell risk from ML forecast predictions."""
    predictions = forecast.get('predictions', [])
    
    if not predictions:
        return "Unknown"
    
    # Count consecutive days with < 2mm rainfall
    max_dry_spell = 0
    current_dry = 0
    
    for pred in predictions:
        if pred.get('rainfall', 0) < 2.0:
            current_dry += 1
            max_dry_spell = max(max_dry_spell, current_dry)
        else:
            current_dry = 0
    
    if max_dry_spell > 7:
        return "High"
    elif max_dry_spell > 4:
        return "Moderate"
    else:
        return "Low"
