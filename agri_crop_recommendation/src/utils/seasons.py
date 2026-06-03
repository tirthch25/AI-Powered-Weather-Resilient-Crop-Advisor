"""
Season Detection for Indian Agriculture

Determines agricultural seasons (Kharif, Rabi, Zaid) based on date and region.
Uses region-specific seasonal calendars for accurate season detection.
"""

from datetime import datetime
from typing import Optional, Tuple
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

# Load all season data from central knowledge database
_KNOWLEDGE_PATH = Path("data/reference/crop_knowledge.json")
_PLANTING_WINDOWS: dict = {}
_SEASON_INFO: dict = {}
_WATER_ADJUSTMENTS: dict = {}
try:
    if _KNOWLEDGE_PATH.exists():
        with open(_KNOWLEDGE_PATH, 'r', encoding='utf-8') as _f:
            _db = json.load(_f)
            _PLANTING_WINDOWS    = _db.get("season_planting_windows", {})
            _SEASON_INFO         = _db.get("season_info", {})
            _WATER_ADJUSTMENTS   = _db.get("season_water_adjustments", {})
except Exception as _e:
    logger.warning(f"Could not load season data from crop_knowledge.json: {_e}")


# Season definitions for Indian agriculture
# Kharif: Monsoon season (June-October)
# Rabi: Winter season (October-March)
# Zaid: Summer season (March-June)

SEASON_CALENDAR = {
    # Format: (start_month, start_day, end_month, end_day)
    "Kharif": (6, 1, 10, 31),    # June 1 - October 31
    "Rabi": (11, 1, 3, 31),       # November 1 - March 31
    "Zaid": (3, 1, 6, 30)         # March 1 - June 30 (overlaps with Rabi end)
}


def detect_season(date: datetime, region_id: Optional[str] = None) -> str:
    """
    Detect the current agricultural season based on date.
    
    Args:
        date: Date to check
        region_id: Optional region identifier for region-specific calendars
        
    Returns:
        Season name: "Kharif", "Rabi", or "Zaid"
    """
    month = date.month
    day = date.day
    
    # Kharif season: June 1 - October 31
    if (month == 6 and day >= 1) or (6 < month < 10) or (month == 10 and day <= 31):
        return "Kharif"
    
    # Rabi season: November 1 - March 31
    elif (month == 11 and day >= 1) or (month == 12) or (month <= 2) or (month == 3 and day <= 31):
        return "Rabi"
    
    # Zaid season: April 1 - May 31 (between Rabi and Kharif)
    elif month in [4, 5]:
        return "Zaid"
    
    # Default to Kharif for edge cases
    else:
        logger.warning(f"Unexpected date {date}, defaulting to Kharif")
        return "Kharif"


def is_season_transition(date: datetime, days_threshold: int = 30) -> Tuple[bool, Optional[str]]:
    """
    Check if date is within a season transition period.
    
    Args:
        date: Date to check
        days_threshold: Number of days before season end to consider transition
        
    Returns:
        Tuple of (is_transition, next_season)
    """
    current_season = detect_season(date)
    
    # Calculate days until season end
    month = date.month
    day = date.day
    
    # Check if within threshold days of season end
    if current_season == "Kharif":
        # Kharif ends October 31
        if month == 10 and day >= (31 - days_threshold):
            return True, "Rabi"
    elif current_season == "Rabi":
        # Rabi ends March 31
        if month == 3 and day >= (31 - days_threshold):
            return True, "Zaid"
    elif current_season == "Zaid":
        # Zaid ends May 31
        if month == 5 and day >= (31 - days_threshold):
            return True, "Kharif"
    
    return False, None


def get_season_info(season: str) -> dict:
    """
    Get information about an agricultural season.

    Data loaded from data/reference/crop_knowledge.json (season_info).

    Args:
        season: Season name (Kharif, Rabi, Zaid)

    Returns:
        Dictionary with season description, typical crops, water source, characteristics
    """
    if _SEASON_INFO and season in _SEASON_INFO:
        return _SEASON_INFO[season]

    # Fallback if crop_knowledge.json unavailable
    _fallback = {
        "Kharif": {
            "name": "Kharif", "description": "Monsoon season crops (June-October)",
            "typical_crops": ["Bajra", "Jowar", "Rice", "Maize", "Cotton", "Soybean"],
            "water_source": "Primarily monsoon rainfall",
            "characteristics": "High rainfall, warm temperatures, humid conditions"
        },
        "Rabi": {
            "name": "Rabi", "description": "Winter season crops (November-March)",
            "typical_crops": ["Wheat", "Chickpea", "Mustard", "Barley", "Peas"],
            "water_source": "Irrigation and residual soil moisture",
            "characteristics": "Cool temperatures, low rainfall, requires irrigation"
        },
        "Zaid": {
            "name": "Zaid", "description": "Summer season crops (March-June)",
            "typical_crops": ["Watermelon", "Cucumber", "Muskmelon", "Vegetables"],
            "water_source": "Primarily irrigation",
            "characteristics": "Hot temperatures, dry conditions, high water requirement"
        }
    }
    return _fallback.get(season, {})


def get_season_water_adjustment(season: str, base_requirement: float) -> float:
    """
    Adjust water requirements based on season.

    Multipliers are loaded from data/reference/crop_knowledge.json
    (season_water_adjustments).

    Args:
        season: Season name (Kharif, Rabi, Zaid)
        base_requirement: Base water requirement in mm

    Returns:
        Adjusted water requirement in mm
    """
    # Load from JSON; fallback to hardcoded defaults
    _fallback_adjustments = {"Kharif": 0.85, "Rabi": 0.95, "Zaid": 1.10}
    adjustments = _WATER_ADJUSTMENTS if _WATER_ADJUSTMENTS else _fallback_adjustments

    # Skip the _comment key if present
    multiplier = adjustments.get(season)
    if not isinstance(multiplier, (int, float)):
        multiplier = 1.0

    adjusted = base_requirement * multiplier

    logger.debug(
        f"Water adjustment for {season}: {base_requirement}mm -> {adjusted}mm "
        f"(multiplier: {multiplier})"
    )
    return adjusted


def get_planting_window(season: str, region_id: Optional[str] = None) -> dict:
    """
    Get optimal planting window for a season.

    Data is loaded from data/reference/crop_knowledge.json (season_planting_windows),
    which is the single source of truth.

    Args:
        season: Season name (Kharif, Rabi, Zaid)
        region_id: Optional region identifier (reserved for future zone-specific windows)

    Returns:
        Dictionary with planting window information
    """
    if _PLANTING_WINDOWS and season in _PLANTING_WINDOWS:
        w = _PLANTING_WINDOWS[season]
        return {
            "start_month": w.get("start_month"),
            "start_day":   w.get("start_day"),
            "end_month":   w.get("end_month"),
            "end_day":     w.get("end_day"),
            "label":       w.get("label", season),
            "description": w.get("description", f"Planting window for {season} season")
        }

    # Fallback if crop_knowledge.json unavailable
    _fallback = {
        "Kharif": {"start_month": 6, "start_day": 1,  "end_month": 7,  "end_day": 15,
                   "description": "Plant with onset of monsoon (early June to mid-July)"},
        "Rabi":   {"start_month": 10, "start_day": 15, "end_month": 11, "end_day": 30,
                   "description": "Plant after monsoon withdrawal (mid-October to November)"},
        "Zaid":   {"start_month": 3,  "start_day": 1,  "end_month": 4,  "end_day": 15,
                   "description": "Plant in early summer (March to mid-April)"}
    }
    return _fallback.get(season, {})


def format_season_guidance(current_season: str, is_transition: bool, next_season: Optional[str]) -> str:
    """
    Format season guidance message for farmers.
    
    Args:
        current_season: Current season
        is_transition: Whether in transition period
        next_season: Next season if in transition
        
    Returns:
        Formatted guidance message
    """
    if is_transition and next_season:
        return (
            f"Currently in late {current_season} season, transitioning to {next_season}. "
            f"Consider crops suitable for both seasons or early {next_season} varieties."
        )
    else:
        season_info = get_season_info(current_season)
        return (
            f"Current season: {current_season}. "
            f"{season_info.get('description', '')}. "
            f"Optimal for: {', '.join(season_info.get('typical_crops', [])[:3])}."
        )
