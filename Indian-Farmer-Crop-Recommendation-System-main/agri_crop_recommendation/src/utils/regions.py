"""
Region Profile Management

Manages region profiles for Indian agricultural regions including
location data, climate zones, and typical soil characteristics.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
import json
from pathlib import Path
import math
import logging

from src.crops.soil import SoilInfo

logger = logging.getLogger(__name__)


@dataclass
class RegionProfile:
    """
    Profile for an Indian agricultural region.
    
    Attributes:
        region_id: Unique identifier (e.g., "PUNE", "SOLAPUR")
        name: Human-readable name
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees
        elevation: Elevation in meters
        climate_zone: Climate classification (e.g., "Semi-Arid", "Sub-Humid")
        typical_soil_types: List of common soil types in the region
        supported_seasons: List of agricultural seasons (Kharif, Rabi, Zaid)
        default_soil: Default soil profile for the region
    """
    region_id: str
    name: str
    state: str
    latitude: float
    longitude: float
    elevation: int
    climate_zone: str
    typical_soil_types: List[str]
    supported_seasons: List[str]
    default_soil: Optional[Dict] = None
    
    def get_default_soil(self) -> Optional[SoilInfo]:
        """Get default soil profile as SoilInfo object."""
        if self.default_soil:
            return SoilInfo.from_dict(self.default_soil)
        return None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "region_id": self.region_id,
            "name": self.name,
            "state": self.state,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "elevation": self.elevation,
            "climate_zone": self.climate_zone,
            "typical_soil_types": self.typical_soil_types,
            "supported_seasons": self.supported_seasons,
            "default_soil": self.default_soil
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'RegionProfile':
        """Create RegionProfile from dictionary."""
        return cls(**data)


class RegionManager:
    """
    Manages region profiles and provides region lookup functionality.
    """
    
    def __init__(self, regions_file: str = "data/reference/regions.json"):
        """
        Initialize the region manager.
        
        Args:
            regions_file: Path to JSON file containing region profiles
        """
        self.regions_file = Path(regions_file)
        self.regions: Dict[str, RegionProfile] = {}
        self._load_regions()
    
    def _load_regions(self) -> None:
        """Load region profiles from JSON file."""
        if not self.regions_file.exists():
            logger.warning(f"Regions file not found: {self.regions_file}")
            self._create_default_regions()
            return
        
        try:
            with open(self.regions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for region_data in data.get('regions', []):
                profile = RegionProfile.from_dict(region_data)
                self.regions[profile.region_id] = profile
            
            logger.info(f"Loaded {len(self.regions)} region profiles")
        except Exception as e:
            logger.error(f"Error loading regions file: {e}")
            self._create_default_regions()
    
    def _create_default_regions(self) -> None:
        """Create default region profiles for major Indian agricultural regions."""
        default_regions = [
            RegionProfile(
                region_id="MH_PUNE", name="Pune", state="Maharashtra",
                latitude=18.5204, longitude=73.8567, elevation=560,
                climate_zone="Semi-Arid", typical_soil_types=["Clay-Loam", "Sandy-Loam"],
                supported_seasons=["Kharif", "Rabi"],
                default_soil={"texture": "Clay-Loam", "ph": 7.2, "organic_matter": "Medium", "drainage": "Medium"}
            ),
            RegionProfile(
                region_id="UP_LUCKNOW", name="Lucknow", state="Uttar Pradesh",
                latitude=26.8467, longitude=80.9462, elevation=123,
                climate_zone="Humid Sub-Tropical", typical_soil_types=["Loam", "Sandy-Loam"],
                supported_seasons=["Kharif", "Rabi", "Zaid"],
                default_soil={"texture": "Loam", "ph": 7.4, "organic_matter": "Medium", "drainage": "Good"}
            ),
            RegionProfile(
                region_id="PB_LUDHIANA", name="Ludhiana", state="Punjab",
                latitude=30.9010, longitude=75.8573, elevation=244,
                climate_zone="Humid Sub-Tropical", typical_soil_types=["Sandy-Loam", "Loam"],
                supported_seasons=["Kharif", "Rabi"],
                default_soil={"texture": "Sandy-Loam", "ph": 7.6, "organic_matter": "Medium", "drainage": "Good"}
            ),
        ]
        
        for region in default_regions:
            self.regions[region.region_id] = region
        
        # Save to file
        self.save_regions()
        logger.info(f"Created {len(default_regions)} default region profiles")
    
    def save_regions(self) -> None:
        """Save region profiles to JSON file."""
        self.regions_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "regions": [region.to_dict() for region in self.regions.values()]
        }
        
        with open(self.regions_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(self.regions)} region profiles to {self.regions_file}")
    
    def get_region_profile(self, region_id: str) -> Optional[RegionProfile]:
        """
        Get region profile by ID.
        
        Args:
            region_id: Region identifier
            
        Returns:
            RegionProfile or None if not found
        """
        return self.regions.get(region_id)
    
    def get_all_regions(self) -> List[RegionProfile]:
        """
        Get all region profiles.
        
        Returns:
            List of all RegionProfile objects
        """
        return list(self.regions.values())
    
    def find_nearest_region(
        self,
        latitude: float,
        longitude: float,
        max_distance_km: float = 50.0
    ) -> Optional[RegionProfile]:
        """
        Find the nearest region to given coordinates using Haversine formula.
        
        Args:
            latitude: Target latitude
            longitude: Target longitude
            max_distance_km: Maximum distance in kilometers (default 50km)
            
        Returns:
            Nearest RegionProfile within max_distance_km, or None if none found
        """
        if not self.regions:
            logger.warning("No regions available for nearest region search")
            return None
        
        nearest_region = None
        min_distance = float('inf')
        
        for region in self.regions.values():
            distance = self._haversine_distance(
                latitude, longitude,
                region.latitude, region.longitude
            )
            
            if distance < min_distance:
                min_distance = distance
                nearest_region = region
        
        if min_distance <= max_distance_km:
            logger.info(
                f"Found nearest region {nearest_region.region_id} "
                f"at {min_distance:.2f} km from ({latitude}, {longitude})"
            )
            return nearest_region
        else:
            logger.warning(
                f"Nearest region {nearest_region.region_id} is {min_distance:.2f} km away, "
                f"exceeds max distance of {max_distance_km} km"
            )
            return None
    
    @staticmethod
    def _haversine_distance(
        lat1: float, lon1: float,
        lat2: float, lon2: float
    ) -> float:
        """
        Calculate distance between two points using Haversine formula.
        
        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates
            
        Returns:
            Distance in kilometers
        """
        # Earth radius in kilometers
        R = 6371.0
        
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        distance = R * c
        return distance
    
    def add_region(self, region: RegionProfile) -> None:
        """
        Add a new region profile.
        
        Args:
            region: RegionProfile to add
        """
        self.regions[region.region_id] = region
        logger.info(f"Added region {region.region_id}")
    
    def region_exists(self, region_id: str) -> bool:
        """
        Check if a region exists.
        
        Args:
            region_id: Region identifier
            
        Returns:
            True if region exists, False otherwise
        """
        return region_id in self.regions
