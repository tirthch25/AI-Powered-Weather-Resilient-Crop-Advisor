"""
Enhanced Crop Recommendation Engine

Integrates historical data, ML predictions, soil compatibility, risk assessment,
and season-aware logic to provide comprehensive crop recommendations for Indian farmers.

ML Integration:
    - When Random Forest model is available: final_score = 0.6*ML + 0.4*rule_based
    - Falls back to pure rule-based scoring when ML model is not available

LLM Integration (Hybrid Mode):
    - Gemini LLM filters crops by regional farming knowledge BEFORE ML scoring
    - Solves the 552-region coverage gap in the static crop database
    - Gracefully falls back to rule-based if LLM is unavailable
"""

from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import logging
import traceback

from src.crops.database import crop_db, get_regional_enrichment
from src.crops.models import CropInfo
from src.crops.soil import SoilInfo, calculate_soil_compatibility_score
from src.utils.regions import RegionManager
from src.utils.seasons import detect_season, is_season_transition, get_season_water_adjustment

# LLM filter (optional — graceful fallback if unavailable)
try:
    from src.services.llm_filter import llm_filter_crops
    _LLM_AVAILABLE = True
except ImportError:
    _LLM_AVAILABLE = False
    llm_filter_crops = None

logger = logging.getLogger(__name__)

# Cache for ML model (loaded once)
_ml_model_cache = None



def _get_regional_score(crop, region_id: str) -> float:
    """
    Look up regional suitability score for a crop.
    Priority: (1) Gemini enrichment JSON → (2) static database → (3) 0.50 fallback.
    """
    if region_id is None:
        return 0.50
    # 1. Check Gemini enrichment JSON first
    enrichment = get_regional_enrichment(region_id)
    if enrichment:
        approved = enrichment.get("approved", {})
        if crop.crop_id in approved:
            return float(approved[crop.crop_id])
        # Crop explicitly excluded — give low score
        excluded = enrichment.get("excluded", [])
        if crop.crop_id in excluded:
            return 0.20
    # 2. Try full ID in static database
    score = crop.regional_suitability.get(region_id)
    if score is not None:
        return score
    # 3. Try stripping state prefix (e.g. MH_PUNE -> PUNE)
    parts = region_id.split('_', 1)
    if len(parts) == 2:
        short_key = parts[1]  # e.g. PUNE
        score = crop.regional_suitability.get(short_key)
        if score is not None:
            return score
    # Default below-average for unknown regions — no free pass
    return 0.50


def _load_crop_ml_model():
    """Load the Random Forest crop suitability model (cached)."""
    global _ml_model_cache
    
    if _ml_model_cache is not None:
        return _ml_model_cache
    
    try:
        from src.ml.predictor import CropSuitabilityRF
        model = CropSuitabilityRF.load()
        if model is not None:
            _ml_model_cache = model
            logger.info("Loaded ML crop suitability model")
        return model
    except Exception as e:
        logger.warning(f"ML model load failed: {e}\n" + traceback.format_exc())
        return None


def _get_ml_score(ml_model, crop, region_id, season, soil, avg_temp, rainfall, dry_spell, irrigation):
    """Get ML prediction for a crop-condition combination."""
    if ml_model is None:
        return None
    
    try:
        features = {
            'crop_id': crop.crop_id,
            'region_id': region_id or 'PUNE',
            'season': season,
            'avg_temp': avg_temp,
            'total_rainfall': rainfall,
            'max_dry_spell': int(dry_spell),
            'soil_texture': soil.texture if soil else 'Loam',
            'soil_ph': soil.ph if soil else 6.5,
            'organic_matter': soil.organic_matter if soil else 'Medium',
            'drainage': soil.drainage if soil else 'Medium',
            'irrigation': int(bool(irrigation)),  # 0 or 1 — consistent with training encoding
            'crop_temp_min': crop.temp_min,
            'crop_temp_max': crop.temp_max,
            'crop_water_req': crop.water_requirement_mm,
            'crop_duration': crop.duration_days,
            'drought_tolerance': crop.drought_tolerance,
            'regional_suitability': _get_regional_score(crop, region_id or 'PUNE')
        }
        
        score = ml_model.predict_score(features)
        return max(0, min(100, score))
    except Exception as e:
        logger.warning(f"ML prediction failed for {crop.crop_id}: {e}\n" + traceback.format_exc())
        return None


def recommend_crops(
    weather_df: pd.DataFrame,
    season: str,
    region_id: Optional[str] = None,
    soil: Optional[SoilInfo] = None,
    irrigation_available: bool = True,
    planning_days: int = 90,
) -> List[Dict]:
    """
    Generate crop recommendations based on weather, soil, and regional data.
    
    Uses ML-enhanced scoring when trained models are available,
    falls back to rule-based scoring otherwise.
    
    Args:
        weather_df: Weather forecast DataFrame
        season: Agricultural season (Kharif, Rabi, Zaid)
        region_id: Region identifier (optional)
        soil: Soil information (optional)
        irrigation_available: Whether irrigation is available
        planning_days: Planning horizon in days
        
    Returns:
        List of crop recommendations sorted by suitability score
    """
    logger.info(f"Generating recommendations for season={season}, region={region_id}")
    
    # Calculate weather statistics
    avg_temp = weather_df["temp_avg"].mean()
    avg_daily_rain = weather_df["rainfall"].mean()
    max_dry_spell = weather_df["dry_spell_days"].max()
    
    # Climatological fallback
    if avg_daily_rain < 0.5:
        avg_daily_rain = 1.5
    
    expected_rainfall = avg_daily_rain * planning_days
    
    # Get crops for season
    season_crops = crop_db.get_crops_by_season(season)
    logger.info(f"Found {len(season_crops)} crops for {season} season")
    
    # Filter by region if provided
    # - If enrichment JSON has this region: use low threshold (0.20) — enrichment gate does real selection
    # - Otherwise use 0.45 to reduce irrelevant crop pass-through
    if region_id:
        _pre_enrichment = get_regional_enrichment(region_id)
        _threshold = 0.20 if _pre_enrichment else 0.45
        season_crops = [c for c in season_crops
                        if _get_regional_score(c, region_id) >= _threshold]
        logger.info(f"Filtered to {len(season_crops)} crops suitable for {region_id} (threshold={_threshold})")

    # ── Enrichment-based Regional Gate (fast, deterministic) ─────────────────
    # If this district is already in regional_crops.json (Gemini enrichment),
    # use that pre-computed approved list — no runtime LLM call needed.
    _enrichment_used = False
    if region_id:
        enrichment = get_regional_enrichment(region_id)
        if enrichment:
            approved_ids_set = set(enrichment.get("approved", {}).keys())
            if approved_ids_set:
                before_count = len(season_crops)
                season_crops = [c for c in season_crops if c.crop_id in approved_ids_set]
                logger.info(
                    f"Enrichment gate: {before_count} → {len(season_crops)} crops "
                    f"for {region_id} (source: {enrichment.get('source', 'enrichment')})"
                )
                _enrichment_used = True

    # ── LLM Regional Gate (Hybrid Mode) ──────────────────────────────────────
    # Only invoked for regions NOT yet in the enrichment JSON.
    # Falls back silently to the rule-based list if LLM is unavailable.
    if region_id and _LLM_AVAILABLE and season_crops and not _enrichment_used:
        try:
            crop_ids_list   = [c.crop_id    for c in season_crops]
            crop_names_list = [c.common_name for c in season_crops]

            # Resolve region name for the prompt
            try:
                rm = RegionManager()
                region_obj  = rm.get_region_profile(region_id)
                region_name = region_obj.name  if region_obj else region_id
                state_name  = getattr(region_obj, 'state', '') if region_obj else ''
            except Exception:
                region_name = region_id
                state_name  = ''

            approved_ids = llm_filter_crops(
                crop_ids=crop_ids_list,
                crop_names=crop_names_list,
                region_id=region_id,
                region_name=region_name,
                season=season,
                state=state_name,
            )

            if approved_ids is not None:
                approved_set = set(approved_ids)
                season_crops = [c for c in season_crops if c.crop_id in approved_set]
                logger.info(
                    f"LLM gate: {len(crop_ids_list)} → {len(season_crops)} crops "
                    f"approved for {region_name}"
                )
            else:
                logger.info("LLM gate unavailable — using rule-based list")
        except Exception as e:
            logger.warning(f"LLM gate error (falling back): {e}\n" + traceback.format_exc())
    # ── End Gates ────────────────────────────────────────────────────────────

    # Filter by soil if provided
    if soil:
        season_crops = crop_db.filter_by_soil(season_crops, soil, min_score=40.0)
        logger.info(f"Filtered to {len(season_crops)} crops compatible with soil")

    # Filter by planning horizon — only recommend crops that can be harvested within the period.
    # Use the shorter end of duration_range when available, falling back to duration_days.
    # Allow a 20% grace margin so crops only slightly over the limit are not silently dropped.
    max_duration = int(planning_days * 1.2)
    duration_filtered = []
    for c in season_crops:
        # Use shortest variant if crop has a range (e.g. 60–90 days → use 60)
        min_dur = min(c.duration_range) if hasattr(c, 'duration_range') and c.duration_range else c.duration_days
        if min_dur <= max_duration:
            duration_filtered.append(c)
    if duration_filtered:
        season_crops = duration_filtered
        logger.info(f"Filtered to {len(season_crops)} crops fitting planning_days={planning_days} (max_duration={max_duration})")
    else:
        logger.warning(f"No crops fit within planning_days={planning_days} after duration filter — skipping filter")

    # Try to load ML crop suitability model
    ml_model = _load_crop_ml_model()
    
    # Prepare weather conditions for risk/pest assessment
    # Use humidity from DataFrame if available (enriched by historical data), else derive from zone
    if "humidity" in weather_df.columns:
        avg_humidity = float(weather_df["humidity"].mean())
    else:
        try:
            from src.weather.history import get_climate_for_region
            from src.utils.seasons import detect_season
            hist = get_climate_for_region(region_id, season)
            avg_humidity = hist.get("avg_humidity", 65.0)
        except Exception:
            logger.warning("[humidity lookup failed]\n" + traceback.format_exc())
            avg_humidity = 65.0

    weather_conditions = {
        'avg_temp_max':      float(weather_df['temp_max'].mean()),
        'avg_temp_min':      float(weather_df['temp_min'].mean()),
        'avg_temp':          float(avg_temp),
        'total_rainfall':    float(expected_rainfall),
        'avg_daily_rainfall': float(avg_daily_rain),
        'avg_humidity':      round(avg_humidity, 1),
        'forecast_days':     planning_days
    }
    
    # Score each crop
    recommendations = []
    for crop in season_crops:
        # Rule-based score
        rule_score = calculate_suitability_score(
            crop=crop,
            avg_temp=avg_temp,
            expected_rainfall=expected_rainfall,
            max_dry_spell=max_dry_spell,
            season=season,
            region_id=region_id,
            soil=soil,
            irrigation_available=irrigation_available,
            planning_days=planning_days
        )
        
        # ML score (if model available)
        ml_score = _get_ml_score(
            ml_model, crop, region_id, season, soil,
            avg_temp, expected_rainfall, max_dry_spell, irrigation_available
        )
        
        # Blend scores: 60% ML + 40% rule-based (or pure rule-based if no ML)
        if ml_score is not None:
            final_score = 0.6 * ml_score + 0.4 * rule_score
            score_source = "ml_blended"
        else:
            final_score = rule_score
            score_source = "rule_based"
        
        # Calculate water requirements
        irrigation_buffer = min(int(1.2 * planning_days), 200) if irrigation_available else 0
        water_available = expected_rainfall + irrigation_buffer
        irrigation_needed = max(0, crop.water_requirement_mm - expected_rainfall)
        
        # Determine risk
        risk = determine_risk_level(crop, max_dry_spell, water_available)

        recommendations.append({
            "crop": crop.common_name,
            "crop_id": crop.crop_id,
            "suitability_score": float(round(final_score, 2)),
            "rule_based_score": float(round(rule_score, 2)),
            "ml_score": float(round(ml_score, 2)) if ml_score is not None else None,
            "score_source": score_source,
            "expected_rainfall_mm": float(round(expected_rainfall, 1)),
            "water_required_mm": crop.water_requirement_mm,
            "irrigation_needed_mm": float(round(irrigation_needed, 1)),
            "growth_duration_days": crop.duration_days,
            "min_duration_days": min(crop.duration_range) if crop.duration_range else crop.duration_days,
            "duration_range": list(crop.duration_range) if crop.duration_range else [crop.duration_days, crop.duration_days],
            "risk_note": risk,
            "drought_tolerance": crop.drought_tolerance,
            "regional_suitability": _get_regional_score(crop, region_id),
        })
    
    # Sort by suitability score
    recommendations.sort(key=lambda x: x["suitability_score"], reverse=True)
    
    logger.info(f"Generated {len(recommendations)} recommendations")
    return recommendations


def calculate_suitability_score(
    crop: CropInfo,
    avg_temp: float,
    expected_rainfall: float,
    max_dry_spell: int,
    season: str,
    region_id: Optional[str] = None,
    soil: Optional[SoilInfo] = None,
    irrigation_available: bool = True,
    planning_days: int = 90
) -> float:
    """
    Calculate comprehensive suitability score (0-100).
    
    Scoring components:
    - Temperature compatibility: 25%
    - Water availability: 25%
    - Soil compatibility: 15%
    - Regional suitability: 15%
    - Seasonal adjustment: 10%
    - Drought tolerance bonus: 10%
    
    Args:
        crop: Crop information
        avg_temp: Average temperature
        expected_rainfall: Expected rainfall in mm
        max_dry_spell: Maximum dry spell days
        season: Current season
        region_id: Region identifier
        soil: Soil information
        irrigation_available: Whether irrigation is available
        
    Returns:
        Suitability score (0-100)
    """
    score = 0.0
    
    # 1. Temperature score (25%)
    temp_score = calculate_temperature_score(crop, avg_temp)
    score += temp_score * 0.25
    
    # 2. Water score (25%)
    water_score = calculate_water_score(crop, expected_rainfall, irrigation_available, season, planning_days)
    score += water_score * 0.25
    
    # 3. Soil score (15%)
    if soil:
        soil_score = calculate_soil_compatibility_score(crop, soil)
    else:
        soil_score = 70.0  # Default if no soil info
    score += soil_score * 0.15
    
    # 4. Regional score (15%)
    regional_score = _get_regional_score(crop, region_id) * 100
    score += regional_score * 0.15
    
    # 5. Seasonal adjustment (10%)
    seasonal_score = 100.0 if season in crop.seasons else 50.0
    score += seasonal_score * 0.10
    
    # 6. Drought tolerance bonus (10%)
    drought_score = calculate_drought_tolerance_score(crop, max_dry_spell)
    score += drought_score * 0.10
    
    return min(score, 100.0)


def calculate_temperature_score(crop: CropInfo, avg_temp: float) -> float:
    """Calculate temperature compatibility score (0-100).

    Uses a three-zone model:
      - Within optimal range: 100
      - Between optimal and absolute max/min: linear decay to 60
      - Slightly beyond absolute max/min (up to 8°C): gentle decay to 20
      - Further beyond absolute limits: 0

    The 8°C grace margin accounts for the 30-day average being lower than
    peak daily highs, and for modern irrigation/shade structures that let
    farmers grow crops slightly outside their listed temperature range.
    """
    if crop.temp_optimal_min <= avg_temp <= crop.temp_optimal_max:
        return 100.0
    elif crop.temp_min <= avg_temp <= crop.temp_max:
        # Linear decay from optimal range toward absolute limit
        if avg_temp < crop.temp_optimal_min:
            range_size = crop.temp_optimal_min - crop.temp_min
            distance = crop.temp_optimal_min - avg_temp
        else:
            range_size = crop.temp_max - crop.temp_optimal_max
            distance = avg_temp - crop.temp_optimal_max

        if range_size > 0:
            return 100.0 - (distance / range_size) * 40.0
        return 60.0
    else:
        # Graceful decay for temperatures slightly beyond absolute limits (up to 8°C overshoot)
        GRACE_MARGIN = 8.0
        if avg_temp > crop.temp_max:
            overshoot = avg_temp - crop.temp_max
        else:
            overshoot = crop.temp_min - avg_temp

        if overshoot <= GRACE_MARGIN:
            # Decay from 20 down to 0 linearly across the grace margin
            return max(0.0, 20.0 * (1.0 - overshoot / GRACE_MARGIN))
        return 0.0


def calculate_water_score(
    crop: CropInfo,
    expected_rainfall: float,
    irrigation_available: bool,
    season: str,
    planning_days: int = 90
) -> float:
    """Calculate water availability score (0-100)."""
    # Adjust water requirement based on season
    adjusted_requirement = get_season_water_adjustment(season, crop.water_requirement_mm)
    
    # Calculate available water — scale buffer by planning_days so longer horizons
    # correctly reflect more total irrigation supply (capped to avoid over-inflation)
    if irrigation_available:
        irrigation_buffer = min(int(1.2 * planning_days), 200)  # ~1.2mm/day, max 200mm
    else:
        irrigation_buffer = 0
    water_available = expected_rainfall + irrigation_buffer
    
    # Calculate ratio
    water_ratio = water_available / adjusted_requirement if adjusted_requirement > 0 else 1.0
    
    if water_ratio >= 1.0:
        # Sufficient water
        return 100.0
    elif water_ratio >= 0.8:
        # Slight deficit
        if crop.drought_tolerance == "High":
            return 90.0
        elif crop.drought_tolerance == "Moderate":
            return 75.0
        else:
            return 60.0
    elif water_ratio >= 0.6:
        # Moderate deficit
        if crop.drought_tolerance == "High":
            return 75.0
        elif crop.drought_tolerance == "Moderate":
            return 50.0
        else:
            return 30.0
    else:
        # Severe deficit
        if crop.drought_tolerance == "High":
            return 50.0
        else:
            return 0.0


def calculate_drought_tolerance_score(crop: CropInfo, max_dry_spell: int) -> float:
    """Calculate drought tolerance bonus score (0-100)."""
    if max_dry_spell <= 4:
        return 100.0  # Low drought risk
    elif max_dry_spell <= 7:
        # Moderate drought risk
        if crop.drought_tolerance == "High":
            return 100.0
        elif crop.drought_tolerance == "Moderate":
            return 70.0
        else:
            return 40.0
    else:
        # High drought risk
        if crop.drought_tolerance == "High":
            return 80.0
        elif crop.drought_tolerance == "Moderate":
            return 40.0
        else:
            return 0.0


def determine_risk_level(crop: CropInfo, max_dry_spell: int, water_available: float) -> str:
    """Determine overall risk level."""
    risks = []
    
    # Drought risk
    if max_dry_spell > 7:
        if crop.drought_tolerance == "Low":
            risks.append("High drought risk")
        elif crop.drought_tolerance == "Moderate":
            risks.append("Moderate drought risk")
    
    # Water deficit risk
    water_ratio = water_available / crop.water_requirement_mm if crop.water_requirement_mm > 0 else 1.0
    if water_ratio < 0.8:
        risks.append("Water deficit risk")
    
    if not risks:
        return "Low risk"
    elif len(risks) == 1:
        return risks[0]
    else:
        return "Multiple risks: " + ", ".join(risks)
