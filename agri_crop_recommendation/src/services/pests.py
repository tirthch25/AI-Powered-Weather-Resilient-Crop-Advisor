"""
Pest and Disease Warning System

Provides crop-specific pest/disease alerts based on weather conditions.
Uses a knowledge base of pests/diseases mapped to triggering conditions
(temperature + humidity + rainfall thresholds).

Data Source:
    Pest/disease entries are loaded from data/reference/crop_knowledge.json
    (the "pest_diseases" key) — the single source of truth.
    Falls back to a hardcoded minimal set if the file is unavailable.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Path to the central crop knowledge database
CROP_KNOWLEDGE_PATH = Path("data/reference/crop_knowledge.json")


def _load_pest_db() -> List[Dict]:
    """Load pest/disease database from crop_knowledge.json."""
    if CROP_KNOWLEDGE_PATH.exists():
        try:
            with open(CROP_KNOWLEDGE_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            entries = data.get("pest_diseases", [])
            logger.info(f"Loaded {len(entries)} pest/disease entries from {CROP_KNOWLEDGE_PATH}")
            return entries
        except Exception as e:
            logger.error(f"Failed to load pest DB from crop_knowledge.json: {e}")
    else:
        logger.warning(f"Crop knowledge file not found at {CROP_KNOWLEDGE_PATH}. Using built-in fallback.")

    # Minimal hardcoded fallback — full list is in crop_knowledge.json
    return [
        {
            "id": "aphids", "name": "Aphids", "type": "pest",
            "affected_crops": ["MOONG_01","URAD_01","TOMATO_01","BRINJAL_01","OKRA_01"],
            "conditions": {"temp_min": 15, "temp_max": 30, "humidity_min": 60, "rainfall_max": 10},
            "severity_base": "Moderate",
            "description": "Sap-sucking insects that weaken plants and transmit viruses.",
            "prevention": "Use neem oil spray. Introduce ladybugs. Remove affected plant parts."
        },
        {
            "id": "pod_borer", "name": "Pod Borer (Helicoverpa)", "type": "pest",
            "affected_crops": ["MOONG_01","URAD_01","COWPEA_01","GUAR_01","SOYBEAN_01"],
            "conditions": {"temp_min": 20, "temp_max": 35, "humidity_min": 50},
            "severity_base": "High",
            "description": "Larvae bore into pods and feed on developing seeds.",
            "prevention": "Use pheromone traps. Apply NPV. Hand-pick larvae."
        }
    ]


# Load once at module import
PEST_DISEASE_DB: List[Dict] = _load_pest_db()


class PestWarningSystem:
    """
    Generates crop-specific pest and disease warnings based on weather conditions.

    All pest/disease data is loaded from data/reference/crop_knowledge.json,
    covering all 50+ crops. To add new pests/diseases, edit crop_knowledge.json —
    no Python code changes needed.

    Example:
        >>> pws = PestWarningSystem()
        >>> warnings = pws.get_warnings("TOMATO_01", weather_conditions)
    """

    def __init__(self, custom_db_path: str = None):
        """
        Args:
            custom_db_path: Optional path to override the default pest/disease JSON database.
                            If provided, loads from this file instead of crop_knowledge.json.
        """
        if custom_db_path:
            path = Path(custom_db_path)
            if path.exists():
                try:
                    with open(path, 'r') as f:
                        content = json.load(f)
                    # Support both a raw list and a dict with "pest_diseases" key
                    if isinstance(content, list):
                        self.pest_db = content
                    else:
                        self.pest_db = content.get("pest_diseases", PEST_DISEASE_DB)
                    logger.info(f"Loaded custom pest DB from {path} ({len(self.pest_db)} entries)")
                except Exception as e:
                    logger.warning(f"Failed to load custom pest DB: {e}. Using default.")
                    self.pest_db = PEST_DISEASE_DB
            else:
                logger.warning(f"Custom pest DB path not found: {path}. Using default.")
                self.pest_db = PEST_DISEASE_DB
        else:
            self.pest_db = PEST_DISEASE_DB

    def get_warnings(
        self,
        crop_id: str,
        weather_conditions: Dict,
        season: str = None
    ) -> List[Dict]:
        """
        Get pest/disease warnings for a specific crop based on weather.

        Args:
            crop_id: Crop identifier (e.g., "TOMATO_01", "SPINACH_01")
            weather_conditions: Dict with avg_temp_max, avg_temp_min,
                               avg_humidity, total_rainfall, etc.
            season: Current agricultural season (optional, reserved for future use)

        Returns:
            List of warning dicts sorted by severity (Critical → High → Moderate → Low)
        """
        warnings = []

        avg_temp = weather_conditions.get('avg_temp',
            (weather_conditions.get('avg_temp_max', 30) +
             weather_conditions.get('avg_temp_min', 20)) / 2)
        humidity = weather_conditions.get('avg_humidity', 65)
        daily_rainfall = weather_conditions.get('avg_daily_rainfall',
            weather_conditions.get('total_rainfall', 0) /
            max(weather_conditions.get('forecast_days', 7), 1))

        for entry in self.pest_db:
            # Check if this pest/disease affects the crop
            if crop_id not in entry.get('affected_crops', []):
                continue

            # Check if weather conditions trigger the warning
            conditions = entry.get('conditions', {})
            triggered = self._check_conditions(avg_temp, humidity, daily_rainfall, conditions)

            if triggered:
                severity = self._calculate_severity(
                    avg_temp, humidity, daily_rainfall, conditions, entry.get('severity_base', 'Moderate')
                )

                warnings.append({
                    'id': entry['id'],
                    'name': entry['name'],
                    'type': entry['type'],
                    'severity': severity,
                    'description': entry['description'],
                    'prevention': entry['prevention'],
                    'conditions_detail': self._format_conditions(conditions)
                })

        # Sort by severity (Critical > High > Moderate > Low)
        severity_order = {'Critical': 0, 'High': 1, 'Moderate': 2, 'Low': 3}
        warnings.sort(key=lambda w: severity_order.get(w['severity'], 4))

        return warnings

    def get_region_warnings(
        self,
        region_weather: Dict,
        crop_ids: List[str] = None
    ) -> Dict:
        """
        Get all pest/disease warnings for a region.

        Args:
            region_weather: Weather conditions for the region
            crop_ids: Optional list of specific crop IDs to check.
                      If None, checks all crops referenced in the database.

        Returns:
            Dictionary mapping crop_id to list of warnings
        """
        if crop_ids is None:
            crop_ids = list(set(
                crop_id
                for entry in self.pest_db
                for crop_id in entry.get('affected_crops', [])
            ))

        result = {}
        for crop_id in crop_ids:
            warnings = self.get_warnings(crop_id, region_weather)
            if warnings:
                result[crop_id] = warnings

        return result

    def get_all_covered_crops(self) -> List[str]:
        """Return list of all crop IDs that have at least one pest/disease entry."""
        return sorted(set(
            crop_id
            for entry in self.pest_db
            for crop_id in entry.get('affected_crops', [])
        ))

    def _check_conditions(
        self, temp: float, humidity: float, daily_rain: float, conditions: Dict
    ) -> bool:
        """Check if current weather triggers pest/disease conditions."""
        if 'temp_min' in conditions and temp < conditions['temp_min']:
            return False
        if 'temp_max' in conditions and temp > conditions['temp_max']:
            return False
        if 'humidity_min' in conditions and humidity < conditions['humidity_min']:
            return False
        if 'rainfall_min' in conditions and daily_rain < conditions['rainfall_min']:
            return False
        if 'rainfall_max' in conditions and daily_rain > conditions['rainfall_max']:
            return False
        return True

    def _calculate_severity(
        self, temp: float, humidity: float, daily_rain: float,
        conditions: Dict, base_severity: str
    ) -> str:
        """Calculate severity level based on how closely conditions match."""
        severity_levels = ['Low', 'Moderate', 'High', 'Critical']
        base_idx = severity_levels.index(base_severity) if base_severity in severity_levels else 1

        boost = 0

        if 'humidity_min' in conditions:
            if humidity > conditions['humidity_min'] + 15:
                boost += 1  # Very high humidity makes it worse

        if 'temp_min' in conditions and 'temp_max' in conditions:
            optimal_temp = (conditions['temp_min'] + conditions['temp_max']) / 2
            if abs(temp - optimal_temp) < 3:
                boost += 1  # Very close to optimal pest temperature

        final_idx = min(base_idx + boost, len(severity_levels) - 1)
        return severity_levels[final_idx]

    @staticmethod
    def _format_conditions(conditions: Dict) -> str:
        """Format conditions into human-readable string."""
        parts = []
        if 'temp_min' in conditions and 'temp_max' in conditions:
            parts.append(f"Temperature: {conditions['temp_min']}–{conditions['temp_max']}°C")
        if 'humidity_min' in conditions:
            parts.append(f"Humidity: >{conditions['humidity_min']}%")
        if 'rainfall_min' in conditions:
            parts.append(f"Rainfall: >{conditions['rainfall_min']}mm/day")
        if 'rainfall_max' in conditions:
            parts.append(f"Rainfall: <{conditions['rainfall_max']}mm/day")
        return "; ".join(parts)
