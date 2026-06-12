"""
User-facing API request models.

These schemas describe the inputs accepted by the FastAPI routes in
src.api.app. They intentionally cover both the legacy India-region workflow
and the current global agent workflow used by the web UI.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


Season = Literal["Kharif", "Rabi", "Zaid"]
IrrigationLevel = Literal["None", "Limited", "Full"]
SoilTexture = Literal["Clay", "Loam", "Sandy", "Clay-Loam", "Sandy-Loam"]
SoilLevel = Literal["Low", "Medium", "High"]
DrainageLevel = Literal["Poor", "Medium", "Good"]


class SoilRequest(BaseModel):
    """Optional soil override supplied by a user."""

    model_config = ConfigDict(str_strip_whitespace=True)

    texture: SoilTexture = Field(..., description="Soil texture")
    ph: float = Field(..., ge=0, le=14, description="Soil pH (0-14)")
    organic_matter: SoilLevel = Field(..., description="Organic matter level")
    drainage: Optional[DrainageLevel] = Field("Medium", description="Drainage quality")


class RegionRequest(BaseModel):
    """Legacy India-region crop recommendation request."""

    model_config = ConfigDict(str_strip_whitespace=True)

    region_id: Optional[str] = Field(None, description="Region ID, for example MH_PUNE")
    latitude: Optional[float] = Field(None, ge=-90, le=90, description="Latitude")
    longitude: Optional[float] = Field(None, ge=-180, le=180, description="Longitude")
    season: Optional[Season] = Field(None, description="Season; auto-detected if omitted")
    soil: Optional[SoilRequest] = Field(None, description="Soil information override")
    irrigation: IrrigationLevel = Field("Limited", description="Irrigation availability")
    planning_days: int = Field(90, ge=15, le=365, description="Planning horizon in days")

    @field_validator("region_id")
    @classmethod
    def normalize_region_id(cls, value: Optional[str]) -> Optional[str]:
        return value.strip().upper() if value else value


class RiskRequest(BaseModel):
    """Crop risk assessment request for a known region and crop."""

    model_config = ConfigDict(str_strip_whitespace=True)

    region_id: str = Field(..., description="Region ID, for example MH_PUNE")
    crop_id: str = Field(..., description="Crop ID, for example BAJRA_01")
    season: Optional[Season] = Field(None, description="Season; auto-detected if omitted")
    irrigation: IrrigationLevel = Field("Limited", description="Irrigation availability")

    @field_validator("region_id", "crop_id")
    @classmethod
    def normalize_ids(cls, value: str) -> str:
        return value.strip().upper()


class ChatRequest(BaseModel):
    """Non-streaming farmer chat request."""

    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(..., min_length=1, description="Farmer's question in English")
    region_id: Optional[str] = Field("", description="Region or generated location key for context")
    season: Optional[str] = Field("", description="Current season for context")
    history: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Conversation history")
    crop_context: Optional[str] = Field("", description="Top recommended crops from the last recommendation")

    @field_validator("question")
    @classmethod
    def require_non_empty_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question cannot be empty")
        return value

    @field_validator("region_id")
    @classmethod
    def normalize_optional_region(cls, value: Optional[str]) -> Optional[str]:
        return value.strip().upper() if value else value


class StreamChatRequest(ChatRequest):
    """Streaming farmer chat request."""


class AnalyzeRequest(BaseModel):
    """Global agent analysis request used by the web UI."""

    model_config = ConfigDict(str_strip_whitespace=True)

    country_code: str = Field(..., min_length=2, description="ISO country code, for example IN")
    country_name: str = Field(..., min_length=1, description="Country name")
    state_code: str = Field(..., min_length=1, description="State/province code")
    state_name: str = Field(..., min_length=1, description="State/province name")
    district: str = Field(..., min_length=1, description="District or locality name")
    lat: Optional[float] = Field(None, ge=-90, le=90, description="Latitude; auto-resolved if omitted")
    lon: Optional[float] = Field(None, ge=-180, le=180, description="Longitude; auto-resolved if omitted")
    irrigation: IrrigationLevel = Field("Limited", description="Irrigation availability")
    planning_days: int = Field(90, ge=15, le=365, description="Planning horizon in days")
    soil_texture: Optional[SoilTexture] = Field(None, description="Optional soil texture override")
    soil_ph: Optional[float] = Field(None, ge=0, le=14, description="Optional soil pH override")
    soil_organic: Optional[SoilLevel] = Field(None, description="Optional organic matter override")
    soil_drainage: Optional[DrainageLevel] = Field(None, description="Optional drainage override")

    @field_validator("country_code", "state_code")
    @classmethod
    def normalize_codes(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("country_name", "state_name", "district")
    @classmethod
    def strip_names(cls, value: str) -> str:
        return value.strip()
