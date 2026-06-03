"""
Risk Assessment Engine

Evaluates agricultural risks based on weather forecasts and crop requirements:
- Drought risk: forecasted rainfall vs crop water needs
- Temperature stress: extreme heat/cold events vs crop tolerance
- Extreme weather events: heavy rainfall, heatwave likelihood
- Overall risk level: Low / Medium / High / Critical
"""

import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class RiskAssessmentEngine:
    """
    Comprehensive risk assessment for crop cultivation.
    
    Evaluates drought, temperature stress, and extreme weather risks
    based on weather forecasts and crop characteristics.
    """
    
    def assess_risk(
        self,
        crop_info: Dict,
        weather_forecast: Dict,
        season: str,
        irrigation_available: bool = True
    ) -> Dict:
        """
        Run full risk assessment for a crop-weather combination.
        
        Args:
            crop_info: Crop data (temp_min, temp_max, water_requirement_mm, drought_tolerance)
            weather_forecast: Forecast with predictions list and summary
            season: Agricultural season (Kharif, Rabi, Zaid)
            irrigation_available: Whether irrigation is available
            
        Returns:
            Dictionary with individual risk scores and overall assessment
        """
        predictions = weather_forecast.get('predictions', [])
        summary = weather_forecast.get('summary', {})
        
        # Calculate individual risk components
        drought = self._assess_drought_risk(crop_info, summary, season, irrigation_available)
        temp_stress = self._assess_temperature_stress(crop_info, predictions, summary)
        extreme = self._assess_extreme_events(predictions, summary)
        
        # Overall risk score (weighted average)
        overall_score = (
            drought['score'] * 0.4 +
            temp_stress['score'] * 0.35 +
            extreme['score'] * 0.25
        )
        
        overall_level = self._score_to_level(overall_score)
        
        # Compile warnings
        warnings = []
        if drought['score'] > 50:
            warnings.append(drought['warning'])
        if temp_stress['score'] > 50:
            warnings.append(temp_stress['warning'])
        if extreme['score'] > 50:
            warnings.append(extreme['warning'])
        
        return {
            'overall_risk_score': round(overall_score, 1),
            'overall_risk_level': overall_level,
            'drought_risk': drought,
            'temperature_stress': temp_stress,
            'extreme_weather': extreme,
            'warnings': warnings,
            'recommendation': self._get_recommendation(overall_level, warnings)
        }
    
    def _assess_drought_risk(
        self, crop_info: Dict, summary: Dict, season: str, irrigation: bool
    ) -> Dict:
        """Assess drought risk based on rainfall deficit."""
        water_needed = crop_info.get('water_requirement_mm', 500)
        expected_rain = summary.get('total_rainfall', 0)
        drought_tolerance = crop_info.get('drought_tolerance', 'Moderate')
        
        # Calculate water deficit
        deficit = max(0, water_needed - expected_rain)
        deficit_pct = (deficit / water_needed * 100) if water_needed > 0 else 0
        
        # Base score from deficit
        if deficit_pct > 70:
            score = 90
        elif deficit_pct > 50:
            score = 70
        elif deficit_pct > 30:
            score = 50
        elif deficit_pct > 15:
            score = 30
        else:
            score = 10
        
        # Adjust for drought tolerance
        tolerance_modifier = {'High': -20, 'Moderate': 0, 'Low': 15}
        score += tolerance_modifier.get(drought_tolerance, 0)
        
        # Adjust for irrigation
        if irrigation and deficit_pct > 20:
            score -= 25
        
        # Seasonal adjustment
        if season == 'Kharif':
            score -= 10  # Monsoon reduces drought risk
        elif season == 'Zaid':
            score += 10  # Summer increases it
        
        score = max(0, min(100, score))
        level = self._score_to_level(score)
        
        warning = ""
        if score > 50:
            warning = f"Drought risk: {deficit_pct:.0f}% water deficit expected ({deficit:.0f}mm short of {water_needed}mm needed)"
        
        return {
            'score': round(score, 1),
            'level': level,
            'water_deficit_mm': round(deficit, 1),
            'water_deficit_pct': round(deficit_pct, 1),
            'warning': warning
        }
    
    def _assess_temperature_stress(
        self, crop_info: Dict, predictions: List[Dict], summary: Dict
    ) -> Dict:
        """Assess temperature stress (heat and cold)."""
        crop_temp_min = crop_info.get('temp_min', 10)
        crop_temp_max = crop_info.get('temp_max', 40)
        crop_opt_min = crop_info.get('temp_optimal_min', crop_temp_min + 5)
        crop_opt_max = crop_info.get('temp_optimal_max', crop_temp_max - 5)
        
        avg_temp_max = summary.get('avg_temp_max', 30)
        avg_temp_min = summary.get('avg_temp_min', 20)
        
        # Count stress days
        heat_stress_days = 0
        cold_stress_days = 0
        
        for pred in predictions:
            if pred.get('temp_max', 0) > crop_temp_max:
                heat_stress_days += 1
            if pred.get('temp_min', 20) < crop_temp_min:
                cold_stress_days += 1
        
        total_days = max(len(predictions), 1)
        stress_pct = (heat_stress_days + cold_stress_days) / total_days * 100
        
        # Score based on stress percentage and severity
        score = 0
        
        # Heat stress
        if avg_temp_max > crop_temp_max:
            score += 40 + (avg_temp_max - crop_temp_max) * 5
        elif avg_temp_max > crop_opt_max:
            score += 20
        
        # Cold stress
        if avg_temp_min < crop_temp_min:
            score += 40 + (crop_temp_min - avg_temp_min) * 5
        elif avg_temp_min < crop_opt_min:
            score += 15
        
        # Adjust for stress days
        score += stress_pct * 0.3
        
        score = max(0, min(100, score))
        level = self._score_to_level(score)
        
        warning = ""
        if score > 50:
            issues = []
            if heat_stress_days > 0:
                issues.append(f"{heat_stress_days} heat stress days (>{crop_temp_max}°C)")
            if cold_stress_days > 0:
                issues.append(f"{cold_stress_days} cold stress days (<{crop_temp_min}°C)")
            warning = f"Temperature stress: {', '.join(issues)}"
        
        return {
            'score': round(score, 1),
            'level': level,
            'heat_stress_days': heat_stress_days,
            'cold_stress_days': cold_stress_days,
            'warning': warning
        }
    
    def _assess_extreme_events(
        self, predictions: List[Dict], summary: Dict
    ) -> Dict:
        """Assess risk of extreme weather events."""
        heavy_rain_days = 0
        heatwave_days = 0
        
        for pred in predictions:
            if pred.get('rainfall', 0) > 50:  # Heavy rainfall threshold
                heavy_rain_days += 1
            if pred.get('temp_max', 0) > 42:  # Heatwave threshold
                heatwave_days += 1
        
        total_rainfall = summary.get('total_rainfall', 0)
        
        score = 0
        
        # Heavy rainfall / flooding risk
        if heavy_rain_days > 3:
            score += 40
        elif heavy_rain_days > 1:
            score += 20
        
        # Very high total rainfall (potential waterlogging)
        if total_rainfall > 800:
            score += 25
        elif total_rainfall > 500:
            score += 10
        
        # Heatwave risk
        if heatwave_days > 2:
            score += 35
        elif heatwave_days > 0:
            score += 15
        
        score = max(0, min(100, score))
        level = self._score_to_level(score)
        
        warning = ""
        if score > 50:
            events = []
            if heavy_rain_days > 0:
                events.append(f"{heavy_rain_days} heavy rainfall days")
            if heatwave_days > 0:
                events.append(f"{heatwave_days} heatwave days")
            warning = f"Extreme weather risk: {', '.join(events)}"
        
        return {
            'score': round(score, 1),
            'level': level,
            'heavy_rain_days': heavy_rain_days,
            'heatwave_days': heatwave_days,
            'warning': warning
        }
    
    @staticmethod
    def _score_to_level(score: float) -> str:
        """Convert numeric risk score to categorical level."""
        if score >= 75:
            return 'Critical'
        elif score >= 50:
            return 'High'
        elif score >= 25:
            return 'Medium'
        else:
            return 'Low'
    
    @staticmethod
    def _get_recommendation(level: str, warnings: List[str]) -> str:
        """Get risk-level-based recommendation."""
        recommendations = {
            'Low': 'Conditions are favorable for planting. Proceed with standard cultivation practices.',
            'Medium': 'Some risks detected. Monitor conditions closely and prepare contingency measures.',
            'High': 'Significant risks present. Consider drought-resistant varieties or delay planting. Ensure irrigation backup.',
            'Critical': 'Severe risk conditions. Strongly recommend postponing planting or choosing highly resilient crop varieties.'
        }
        return recommendations.get(level, recommendations['Medium'])
