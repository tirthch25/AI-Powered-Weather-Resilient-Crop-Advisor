"""
Soil Data Models and Compatibility Scoring

Provides soil information models and functions to calculate soil-crop compatibility.
"""

from dataclasses import dataclass
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


@dataclass
class SoilInfo:
    """
    Soil characteristics information.
    
    Attributes:
        texture: Soil texture type
        ph: Soil pH value (0-14)
        organic_matter: Organic matter content level
        drainage: Drainage quality
    """
    texture: str  # "Clay", "Loam", "Sandy", "Clay-Loam", "Sandy-Loam"
    ph: float
    organic_matter: str  # "Low", "Medium", "High"
    drainage: Optional[str] = "Medium"  # "Poor", "Medium", "Good"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "texture": self.texture,
            "ph": self.ph,
            "organic_matter": self.organic_matter,
            "drainage": self.drainage
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SoilInfo':
        """Create SoilInfo from dictionary."""
        return cls(**data)


def calculate_soil_compatibility_score(crop, soil: SoilInfo) -> float:
    """
    Calculate soil-crop compatibility score (0-100).
    
    Scoring factors:
    - pH compatibility: 0-100 points
    - Texture compatibility: 0-20 bonus points
    - Drainage compatibility: 0-10 bonus points
    
    Args:
        crop: CropInfo object with soil requirements
        soil: SoilInfo object with soil characteristics
        
    Returns:
        Compatibility score (0-100)
    """
    score = 0.0
    
    # 1. pH Compatibility (0-100 points)
    ph_score = calculate_ph_score(crop, soil.ph)
    score += ph_score
    
    # 2. Texture Compatibility (bonus: 0-20 points)
    texture_bonus = calculate_texture_bonus(crop, soil.texture)
    score += texture_bonus
    
    # 3. Drainage Compatibility (bonus: 0-10 points)
    drainage_bonus = calculate_drainage_bonus(crop, soil)
    score += drainage_bonus
    
    # Cap at 100
    score = min(score, 100.0)
    
    logger.debug(
        f"Soil compatibility for {crop.common_name}: "
        f"pH={ph_score:.1f}, texture={texture_bonus:.1f}, "
        f"drainage={drainage_bonus:.1f}, total={score:.1f}"
    )
    
    return score


def calculate_ph_score(crop, ph: float) -> float:
    """
    Calculate pH compatibility score (0-100).
    
    Scoring:
    - Within optimal range: 100 points
    - Within acceptable range: 70 points
    - Outside acceptable: 0 points
    
    Args:
        crop: CropInfo object
        ph: Soil pH value
        
    Returns:
        pH score (0-100)
    """
    # Define optimal range (middle 60% of acceptable range)
    ph_range = crop.soil_ph_max - crop.soil_ph_min
    optimal_margin = ph_range * 0.2
    optimal_min = crop.soil_ph_min + optimal_margin
    optimal_max = crop.soil_ph_max - optimal_margin
    
    if optimal_min <= ph <= optimal_max:
        # Optimal range
        return 100.0
    elif crop.soil_ph_min <= ph <= crop.soil_ph_max:
        # Acceptable but not optimal
        return 70.0
    else:
        # Outside acceptable range
        return 0.0


def calculate_texture_bonus(crop, texture: str) -> float:
    """
    Calculate texture compatibility bonus (0-20).
    
    Scoring:
    - Perfect match: +20 points
    - No match but manageable: 0 points
    - Incompatible: -50 points (penalty)
    
    Args:
        crop: CropInfo object
        texture: Soil texture type
        
    Returns:
        Texture bonus (-50 to +20)
    """
    if texture in crop.suitable_soil_textures:
        # Perfect match
        return 20.0
    else:
        # Check if it's a related texture (e.g., "Loam" is related to "Clay-Loam")
        if is_related_texture(texture, crop.suitable_soil_textures):
            # Manageable but not ideal
            return 0.0
        else:
            # Incompatible texture
            return -50.0


def is_related_texture(texture: str, suitable_textures: list) -> bool:
    """
    Check if a texture is related to suitable textures.
    
    Related textures share a common component:
    - "Clay" is related to "Clay-Loam"
    - "Sandy" is related to "Sandy-Loam"
    - "Loam" is related to "Clay-Loam" and "Sandy-Loam"
    
    Args:
        texture: Soil texture to check
        suitable_textures: List of suitable textures
        
    Returns:
        True if related, False otherwise
    """
    # Extract base components
    texture_parts = set(texture.split('-'))
    
    for suitable in suitable_textures:
        suitable_parts = set(suitable.split('-'))
        # Check if there's any overlap
        if texture_parts & suitable_parts:
            return True
    
    return False


def calculate_drainage_bonus(crop, soil: SoilInfo) -> float:
    """
    Calculate drainage compatibility bonus (0-10).
    
    Considers crop waterlogging tolerance and soil drainage.
    
    Args:
        crop: CropInfo object
        soil: SoilInfo object
        
    Returns:
        Drainage bonus (0-10)
    """
    if soil.drainage is None:
        return 5.0  # Neutral if drainage not specified
    
    # Map drainage quality to numeric value
    drainage_map = {"Poor": 1, "Medium": 2, "Good": 3}
    drainage_value = drainage_map.get(soil.drainage, 2)
    
    # Map waterlogging tolerance to preferred drainage
    if crop.waterlogging_tolerance == "High":
        # Can handle poor drainage
        if drainage_value == 1:
            return 10.0
        elif drainage_value == 2:
            return 8.0
        else:
            return 5.0
    elif crop.waterlogging_tolerance == "Moderate":
        # Prefers medium drainage
        if drainage_value == 2:
            return 10.0
        else:
            return 5.0
    else:  # Low tolerance
        # Needs good drainage
        if drainage_value == 3:
            return 10.0
        elif drainage_value == 2:
            return 5.0
        else:
            return 0.0


def get_soil_amendment_suggestions(crop, soil: SoilInfo) -> list:
    """
    Generate soil amendment suggestions to improve compatibility.
    
    Args:
        crop: CropInfo object
        soil: SoilInfo object
        
    Returns:
        List of amendment suggestions
    """
    suggestions = []
    
    # pH amendments
    if soil.ph < crop.soil_ph_min:
        deficit = crop.soil_ph_min - soil.ph
        if deficit > 1.0:
            suggestions.append(
                f"Add lime to increase pH from {soil.ph:.1f} to at least {crop.soil_ph_min:.1f} "
                f"(apply 2-3 tons/ha of agricultural lime)"
            )
        else:
            suggestions.append(
                f"Add lime to slightly increase pH from {soil.ph:.1f} to {crop.soil_ph_min:.1f} "
                f"(apply 1-2 tons/ha of agricultural lime)"
            )
    elif soil.ph > crop.soil_ph_max:
        excess = soil.ph - crop.soil_ph_max
        if excess > 1.0:
            suggestions.append(
                f"Add sulfur or organic matter to decrease pH from {soil.ph:.1f} to {crop.soil_ph_max:.1f} "
                f"(apply 200-300 kg/ha of elemental sulfur)"
            )
        else:
            suggestions.append(
                f"Add organic matter to slightly decrease pH from {soil.ph:.1f} to {crop.soil_ph_max:.1f}"
            )
    
    # Texture amendments
    if soil.texture not in crop.suitable_soil_textures:
        if soil.texture == "Clay":
            suggestions.append(
                "Add sand and organic matter to improve clay soil structure and drainage "
                "(apply 5-10 tons/ha of well-decomposed compost)"
            )
        elif soil.texture == "Sandy":
            suggestions.append(
                "Add organic matter and clay to improve sandy soil water retention "
                "(apply 10-15 tons/ha of compost or farmyard manure)"
            )
    
    # Drainage amendments
    if soil.drainage == "Poor" and crop.waterlogging_tolerance == "Low":
        suggestions.append(
            "Improve drainage by creating raised beds or installing subsurface drainage systems"
        )
    
    # Organic matter
    if soil.organic_matter == "Low":
        suggestions.append(
            "Increase organic matter content by adding compost or farmyard manure "
            "(apply 10-15 tons/ha annually)"
        )
    
    return suggestions
