"""
Crop Data Models

Comprehensive data models for crop information including temperature requirements,
water needs, soil compatibility, and regional suitability.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

# SoilInfo is defined in src.crops.soil (single source of truth). Re-export here
# for backward compatibility with any code that imports from src.crops.models.
from src.crops.soil import SoilInfo  # noqa: F401


@dataclass
class CropInfo:
    """
    Comprehensive crop information model.
    
    Attributes:
        crop_id: Unique identifier (e.g., "BAJRA_01")
        common_name: Common name (e.g., "Bajra (Pearl Millet)")
        scientific_name: Scientific name
        duration_days: Typical duration in days
        duration_range: Min and max duration (min_days, max_days)
        
        # Temperature requirements (°C)
        temp_min: Minimum temperature tolerance
        temp_optimal_min: Optimal range minimum
        temp_optimal_max: Optimal range maximum
        temp_max: Maximum temperature tolerance
        
        # Water requirements
        water_requirement_mm: Total water needed (mm)
        drought_tolerance: "Low", "Moderate", "High"
        waterlogging_tolerance: "Low", "Moderate", "High"
        
        # Soil requirements
        soil_ph_min: Minimum pH tolerance
        soil_ph_max: Maximum pH tolerance
        suitable_soil_textures: List of suitable textures
        nutrient_requirements: NPK requirements (Low/Medium/High)
        
        # Regional data
        regional_suitability: Dict mapping region_id to suitability score (0-1)
        successful_regions: List of regions where crop is proven successful
        
        # Seasonal data
        seasons: List of suitable seasons (Kharif, Rabi, Zaid)
        
        # Additional metadata
        varieties: List of available varieties
        typical_yield_kg_per_ha: Expected yield
        market_demand: "Low", "Moderate", "High"
    """
    crop_id: str
    common_name: str
    scientific_name: str
    duration_days: int
    duration_range: Tuple[int, int]
    
    # Temperature requirements
    temp_min: float
    temp_optimal_min: float
    temp_optimal_max: float
    temp_max: float
    
    # Water requirements
    water_requirement_mm: int
    drought_tolerance: str
    waterlogging_tolerance: str
    
    # Soil requirements
    soil_ph_min: float
    soil_ph_max: float
    suitable_soil_textures: List[str]
    nutrient_requirements: Dict[str, str]
    
    # Regional data
    regional_suitability: Dict[str, float]
    successful_regions: List[str]
    
    # Seasonal data
    seasons: List[str]
    
    # Additional metadata
    varieties: List[str] = field(default_factory=list)
    typical_yield_kg_per_ha: int = 0
    market_demand: str = "Moderate"
    growing_tip: str = ""
    
    def is_suitable_for_region(self, region_id: str, threshold: float = 0.3) -> bool:
        """
        Check if crop is suitable for a region.
        
        Args:
            region_id: Region identifier
            threshold: Minimum suitability score (0-1)
            
        Returns:
            True if suitable, False otherwise
        """
        return (region_id in self.successful_regions or 
                self.regional_suitability.get(region_id, 0) >= threshold)
    
    def is_suitable_for_season(self, season: str) -> bool:
        """
        Check if crop is suitable for a season.
        
        Args:
            season: Season name (Kharif, Rabi, Zaid)
            
        Returns:
            True if suitable, False otherwise
        """
        return season in self.seasons
    
    def is_suitable_for_soil_ph(self, ph: float) -> bool:
        """
        Check if crop is suitable for given soil pH.
        
        Args:
            ph: Soil pH value
            
        Returns:
            True if suitable, False otherwise
        """
        return self.soil_ph_min <= ph <= self.soil_ph_max
    
    def is_suitable_for_soil_texture(self, texture: str) -> bool:
        """
        Check if crop is suitable for given soil texture.
        
        Args:
            texture: Soil texture type
            
        Returns:
            True if suitable, False otherwise
        """
        return texture in self.suitable_soil_textures
    
    def is_short_duration(self, min_days: int = 70, max_days: int = 90) -> bool:
        """
        Check if crop is short-duration (70-90 days).
        
        Args:
            min_days: Minimum duration threshold
            max_days: Maximum duration threshold
            
        Returns:
            True if short-duration, False otherwise
        """
        return min_days <= self.duration_days <= max_days
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "crop_id": self.crop_id,
            "common_name": self.common_name,
            "scientific_name": self.scientific_name,
            "duration_days": self.duration_days,
            "duration_range": list(self.duration_range),
            "temp_min": self.temp_min,
            "temp_optimal_min": self.temp_optimal_min,
            "temp_optimal_max": self.temp_optimal_max,
            "temp_max": self.temp_max,
            "water_requirement_mm": self.water_requirement_mm,
            "drought_tolerance": self.drought_tolerance,
            "waterlogging_tolerance": self.waterlogging_tolerance,
            "soil_ph_min": self.soil_ph_min,
            "soil_ph_max": self.soil_ph_max,
            "suitable_soil_textures": self.suitable_soil_textures,
            "nutrient_requirements": self.nutrient_requirements,
            "regional_suitability": self.regional_suitability,
            "successful_regions": self.successful_regions,
            "seasons": self.seasons,
            "varieties": self.varieties,
            "typical_yield_kg_per_ha": self.typical_yield_kg_per_ha,
            "market_demand": self.market_demand,
            "growing_tip": self.growing_tip
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CropInfo':
        """Create CropInfo from dictionary."""
        # Convert duration_range from list to tuple
        if 'duration_range' in data and isinstance(data['duration_range'], list):
            data['duration_range'] = tuple(data['duration_range'])
        return cls(**data)


