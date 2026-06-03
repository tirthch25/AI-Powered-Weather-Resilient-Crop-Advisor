"""
Planting Calendar

Generates planting-to-harvest timelines with key milestones:
- Sowing date (based on season and region)
- Germination period
- Vegetative growth phase
- Flowering/fruiting
- Harvest date

Data Source:
    All growth phases, care tips, and planting windows are loaded from
    data/reference/crop_knowledge.json — the single source of truth.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

# Path to the central crop knowledge database
CROP_KNOWLEDGE_PATH = Path("data/reference/crop_knowledge.json")

# Default growth phases for any crop not in the database
DEFAULT_PHASES = {'germination': 0.10, 'vegetative': 0.30, 'flowering': 0.30, 'maturity': 0.30}


def _load_crop_knowledge() -> Dict:
    """Load crop knowledge database from JSON file."""
    if CROP_KNOWLEDGE_PATH.exists():
        try:
            with open(CROP_KNOWLEDGE_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded crop knowledge database from {CROP_KNOWLEDGE_PATH}")
            return data
        except Exception as e:
            logger.error(f"Failed to load crop knowledge database: {e}")
    else:
        logger.warning(f"Crop knowledge database not found at {CROP_KNOWLEDGE_PATH}. Using defaults.")
    return {}


# Load once at module import
_KNOWLEDGE = _load_crop_knowledge()
SEASON_PLANTING_WINDOWS: Dict = _KNOWLEDGE.get("season_planting_windows", {
    "Kharif": {"start_month": 6, "start_day": 15, "end_month": 7, "end_day": 31, "label": "Kharif (June–July)"},
    "Rabi":   {"start_month": 10, "start_day": 15, "end_month": 11, "end_day": 30, "label": "Rabi (October–November)"},
    "Zaid":   {"start_month": 3, "start_day": 1, "end_month": 4, "end_day": 15, "label": "Zaid (March–April)"}
})
GROWTH_PHASES: Dict = _KNOWLEDGE.get("growth_phases", {})
CARE_TIPS_DB: Dict = _KNOWLEDGE.get("care_tips", {})


class PlantingCalendar:
    """
    Generates planting calendars with milestone dates for crops.

    Growth phases and care tips are loaded from data/reference/crop_knowledge.json,
    covering all 50+ crops. Falls back to DEFAULT_PHASES for any unknown crop.

    Example:
        >>> cal = PlantingCalendar()
        >>> timeline = cal.get_calendar("BAJRA_01", "Kharif", duration_days=75)
    """

    def get_calendar(
        self,
        crop_id: str,
        season: str,
        duration_days: int = 90,
        crop_name: str = None,
        sowing_date: datetime = None,
        region_id: str = None
    ) -> Dict:
        """
        Generate planting calendar for a crop.

        Args:
            crop_id: Crop identifier (e.g., "BAJRA_01", "SPINACH_01")
            season: Agricultural season (Kharif, Rabi, Zaid)
            duration_days: Crop growth duration in days
            crop_name: Human-readable crop name
            sowing_date: Override automatic sowing date
            region_id: Region for localized timing

        Returns:
            Calendar dict with phases, milestones, and care tips
        """
        # Determine sowing date
        if sowing_date is None:
            sowing_date = self._get_optimal_sowing_date(season)

        # Get growth phases from database (fallback to default)
        phases = GROWTH_PHASES.get(crop_id, DEFAULT_PHASES)

        # Remove zero-fraction phases (e.g., microgreens have no flowering/maturity)
        phases = {k: v for k, v in phases.items() if v > 0}

        # Calculate milestone dates
        milestones = self._calculate_milestones(sowing_date, duration_days, phases)

        # Get care tips for each phase
        care_tips = self._get_care_tips(crop_id, phases)

        return {
            'crop_id': crop_id,
            'crop_name': crop_name or crop_id,
            'season': season,
            'total_duration_days': duration_days,
            'sowing_date': sowing_date.strftime('%Y-%m-%d'),
            'harvest_date': (sowing_date + timedelta(days=duration_days)).strftime('%Y-%m-%d'),
            'planting_window': SEASON_PLANTING_WINDOWS.get(season, {}).get('label', season),
            'milestones': milestones,
            'care_tips': care_tips,
            'phases': [
                {
                    'name': phase_name.replace('_', ' ').title(),
                    'start_date': milestones[phase_name]['start'],
                    'end_date': milestones[phase_name]['end'],
                    'duration_days': milestones[phase_name]['duration'],
                    'progress_pct': round(pct * 100, 1)
                }
                for phase_name, pct in phases.items()
            ]
        }

    def get_multiple_calendars(
        self,
        crops: List[Dict],
        season: str
    ) -> List[Dict]:
        """
        Generate calendars for multiple crops.

        Args:
            crops: List of dicts with crop_id, duration_days, common_name
            season: Agricultural season

        Returns:
            List of calendar dicts
        """
        calendars = []
        for crop in crops:
            cal = self.get_calendar(
                crop_id=crop.get('crop_id', ''),
                season=season,
                duration_days=crop.get('duration_days', crop.get('growth_duration_days', 90)),
                crop_name=crop.get('common_name', crop.get('crop', ''))
            )
            calendars.append(cal)
        return calendars

    def _get_optimal_sowing_date(self, season: str) -> datetime:
        """Get recommended sowing date for the season."""
        now = datetime.now()
        window = SEASON_PLANTING_WINDOWS.get(season, SEASON_PLANTING_WINDOWS.get('Kharif', {}))

        year = now.year

        start_month = window.get('start_month', 6)
        start_day   = window.get('start_day', 15)
        end_month   = window.get('end_month', 7)
        end_day     = window.get('end_day', 31)

        start = datetime(year, start_month, start_day)

        if now > datetime(year, end_month, end_day):
            # Past this season's window — use next year
            start = datetime(year + 1, start_month, start_day)
        elif now >= start:
            # Currently in the planting window — suggest 7 days from now
            start = now + timedelta(days=7)

        return start

    def _calculate_milestones(
        self, sowing_date: datetime, duration: int, phases: Dict
    ) -> Dict:
        """Calculate start/end dates for each growth phase."""
        milestones = {}
        current_day = 0

        for phase_name, fraction in phases.items():
            phase_days = max(1, int(duration * fraction))
            start_date = sowing_date + timedelta(days=current_day)
            end_date   = sowing_date + timedelta(days=current_day + phase_days)

            milestones[phase_name] = {
                'start': start_date.strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d'),
                'duration': phase_days,
                'day_start': current_day,
                'day_end': current_day + phase_days
            }
            current_day += phase_days

        return milestones

    def _get_care_tips(self, crop_id: str, phases: Dict) -> Dict:
        """
        Get phase-specific care tips from the crop knowledge database.
        Falls back to generic tips if crop_id not found.
        """
        # Check crop-specific tips first, then defaults
        tips_db = CARE_TIPS_DB.get(crop_id) or CARE_TIPS_DB.get("_default") or {}

        # If database not loaded, use hardcoded fallback
        if not tips_db:
            tips_db = {
                'germination': [
                    'Ensure adequate soil moisture for seed germination',
                    'Protect seeds from birds and rodents',
                    'Maintain soil temperature above 20°C for optimal germination'
                ],
                'vegetative': [
                    'Apply first dose of fertilizer (NPK) after 2 weeks',
                    'Regular weeding to reduce competition',
                    'Monitor for early pest attacks',
                    'Ensure consistent irrigation if rainfall is insufficient'
                ],
                'flowering': [
                    'Avoid water stress during this critical phase',
                    'Apply second dose of fertilizer',
                    'Monitor closely for pest and disease outbreaks',
                    'Avoid pesticide spraying during peak flowering (protect pollinators)'
                ],
                'maturity': [
                    'Reduce irrigation gradually as crop approaches maturity',
                    'Watch for signs of harvest readiness',
                    'Prepare storage and drying facilities',
                    'Plan harvest timing to avoid rain damage to mature crop'
                ]
            }

        # Only return tips for phases that exist in this crop's timeline
        return {phase: tips_db.get(phase, []) for phase in phases}
