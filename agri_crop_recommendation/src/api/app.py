import os
import logging
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from starlette.requests import Request
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

from src.weather.fetcher import fetch_weather
from src.ml.pipeline import add_agri_features
from src.weather.forecast import forecast_days_17_90
from src.services.recommender import recommend_crops
from src.utils.regions import RegionManager
from src.crops.soil import SoilInfo
from src.utils.seasons import detect_season, is_season_transition, format_season_guidance
from src.services.risk import RiskAssessmentEngine
from src.services.pests import PestWarningSystem
from src.services.calendar import PlantingCalendar

# LLM Explainer (optional — graceful fallback if unavailable)
try:
    from src.services.llm_explainer import generate_bulk_explanations
    _LLM_EXPLAINER_AVAILABLE = True
except ImportError:
    _LLM_EXPLAINER_AVAILABLE = False
    generate_bulk_explanations = None

# LLM Chat (optional — graceful fallback if unavailable)
try:
    from src.services.llm_chat import answer_farmer_question, stream_farmer_answer, set_weather_cache
    _LLM_CHAT_AVAILABLE = True
except ImportError:
    _LLM_CHAT_AVAILABLE = False
    answer_farmer_question = None
    stream_farmer_answer = None
    set_weather_cache = None

# Agents (LLaMA + Ollama web search — global data gathering)
try:
    from src.agents.location_agent import get_countries, get_states, get_districts, resolve_coordinates
    from src.agents.data_gathering_agent import gather_location_data
    from src.agents.crop_agent import recommend_crops_agent
    _AGENTS_AVAILABLE = True
except ImportError:
    _AGENTS_AVAILABLE = False
    get_countries = get_states = get_districts = resolve_coordinates = None
    gather_location_data = recommend_crops_agent = None

app = FastAPI(
    title="AI-Powered Weather Resilient Crop Advisor v3.0",
    description="Global crop advisor powered by LLaMA + Ollama web search agents. "
                "Supports 50+ countries, 250+ states, 170+ districts. "
                "Agent gathers real-time weather, soil, forecast & market data.",
    version="3.0"
)


@app.on_event("startup")
async def _warmup_llm():
    """
    Warm up the LLM on startup so the first real user request is instant.
    LLaMA 3.2 takes ~60s to load into memory on first call; running a tiny
    prompt at startup eliminates that cold-start delay for users.
    """
    import asyncio
    import threading

    def _do_warmup():
        try:
            from src.services.llm_chat import _resolve_client, OLLAMA_MODEL
            provider, client = _resolve_client()
            if provider == "ollama" and client:
                client.chat(
                    model=OLLAMA_MODEL,
                    messages=[{"role": "user", "content": "hi"}],
                )
                import logging
                logging.getLogger(__name__).info(
                    f"LLaMA model warmed up ({OLLAMA_MODEL}) — chat will be instant"
                )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"LLM warmup skipped: {e}")

    # Run in a background thread so startup doesn't block
    threading.Thread(target=_do_warmup, daemon=True).start()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates — cache_size=0 disables Jinja2's LRUCache which crashes on
# Python 3.14 due to unhashable dict inside tuple cache keys (upstream bug)
from jinja2 import Environment, FileSystemLoader
_jinja_env = Environment(loader=FileSystemLoader("templates"), cache_size=0)
from starlette.templating import Jinja2Templates as _J2T
templates = _J2T(env=_jinja_env)

# Initialize managers
region_manager = RegionManager()
risk_engine = RiskAssessmentEngine()
pest_system = PestWarningSystem()
planting_calendar = PlantingCalendar()


# ----------- Request Schemas -----------

class SoilRequest(BaseModel):
    texture: str = Field(..., description="Soil texture: Clay, Loam, Sandy, Clay-Loam, Sandy-Loam")
    ph: float = Field(..., ge=0, le=14, description="Soil pH (0-14)")
    organic_matter: str = Field(..., description="Organic matter: Low, Medium, High")
    drainage: Optional[str] = Field("Medium", description="Drainage: Poor, Medium, Good")


class RegionRequest(BaseModel):
    region_id: Optional[str] = Field(None, description="Region ID (e.g., PUNE, SOLAPUR)")
    latitude: Optional[float] = Field(None, description="Latitude (if region_id not provided)")
    longitude: Optional[float] = Field(None, description="Longitude (if region_id not provided)")
    season: Optional[str] = Field(None, description="Season: Kharif, Rabi, Zaid (auto-detected if not provided)")
    soil: Optional[SoilRequest] = Field(None, description="Soil information (uses region default if not provided)")
    irrigation: str = Field("Limited", description="Irrigation: None, Limited, Full")
    planning_days: int = Field(90, ge=15, le=365, description="Planning horizon in days (15-365)")


class RiskRequest(BaseModel):
    region_id: str = Field(..., description="Region ID")
    crop_id: str = Field(..., description="Crop ID (e.g., BAJRA_01)")
    season: Optional[str] = Field(None, description="Season (auto-detected if not provided)")
    irrigation: str = Field("Limited", description="Irrigation: None, Limited, Full")


class ChatRequest(BaseModel):
    question: str = Field(..., description="Farmer's question in English")
    region_id: Optional[str] = Field("", description="Region ID for context (e.g., MH_PUNE)")
    season: Optional[str] = Field("", description="Current season for context")
    history: Optional[List[dict]] = Field(default_factory=list, description="Conversation history for multi-turn support")
    crop_context: Optional[str] = Field("", description="Top recommended crops from last recommendation")


# ----------- Helper Functions -----------

def _resolve_region(region_id=None, latitude=None, longitude=None):
    """Resolve region from ID or coordinates."""
    if region_id:
        region = region_manager.get_region_profile(region_id)
        if not region:
            raise HTTPException(status_code=404, detail=f"Region {region_id} not found")
        return region, region.latitude, region.longitude
    elif latitude and longitude:
        region = region_manager.find_nearest_region(latitude, longitude, max_distance_km=150)
        if not region:
            raise HTTPException(status_code=404, detail="No region found within 100km")
        return region, latitude, longitude
    else:
        raise HTTPException(status_code=400, detail="Either region_id or coordinates required")


def _get_weather_and_season(region, latitude, longitude, season=None):
    """Fetch weather, detect season, create forecast."""
    current_date = datetime.now()
    
    if season:
        if season not in ["Kharif", "Rabi", "Zaid"]:
            raise HTTPException(status_code=400, detail="Invalid season. Must be Kharif, Rabi, or Zaid")
    else:
        season = detect_season(current_date, region.region_id)
    
    is_transition, next_season = is_season_transition(current_date)
    
    # Pass region_id and season so fetch_weather can enrich with historical humidity
    weather = fetch_weather(latitude, longitude, region_id=region.region_id, season=season)
    weather = add_agri_features(weather)
    
    return weather, season, is_transition, next_season


# ----------- API Endpoints -----------

@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    """Serve the main web interface."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Serve the real favicon from static/favicon.ico."""
    from fastapi.responses import FileResponse, Response
    from pathlib import Path
    ico_path = Path("static/favicon.ico")
    if ico_path.exists():
        return FileResponse(ico_path, media_type="image/x-icon")
    # Fallback: 1×1 transparent pixel so the browser never gets a 404
    return Response(
        content=b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82',
        media_type="image/x-icon"
    )


@app.get("/health")
def health_check():
    """Health check endpoint."""
    ml_status = {}
    try:
        from src.ml.predictor import CropSuitabilityRF
        rf_model = CropSuitabilityRF.load()
        ml_status['crop_suitability_rf'] = 'loaded' if rf_model else 'not_trained'
    except Exception:
        ml_status['crop_suitability_rf'] = 'not_available'

    # Check LLM availability — prefer Ollama, fall back to Gemini
    llm_provider = os.getenv("LLM_PROVIDER", "ollama")
    ollama_available = False
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")
    try:
        import ollama as _ollama
        _c = _ollama.Client(host=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
        _c.list()
        ollama_available = True
    except Exception:
        pass
    gemini_available = bool(os.getenv("GEMINI_API_KEY"))
    llm_available = ollama_available or gemini_available
    if ollama_available:
        active_llm = f"ollama/{ollama_model}"
    elif gemini_available:
        active_llm = "gemini-2.0-flash-lite (fallback)"
    else:
        active_llm = None

    return {
        "status": "healthy",
        "version": "2.6",
        "regions_loaded": len(region_manager.get_all_regions()),
        "ml_models": ml_status,
        "llm_available": llm_available,
        "llm_provider": llm_provider,
        "llm_model": active_llm,
        "ollama_running": ollama_available,
        "gemini_fallback": gemini_available,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/regions")
def get_regions():
    """Get list of all supported regions."""
    regions = region_manager.get_all_regions()
    return {
        "regions": [
            {
                "region_id": r.region_id,
                "name": r.name,
                "state": getattr(r, 'state', 'Unknown'),
                "latitude": r.latitude,
                "longitude": r.longitude,
                "climate_zone": r.climate_zone,
                "typical_soil_types": r.typical_soil_types
            }
            for r in regions
        ]
    }


@app.post("/recommend")
def recommend(request: RegionRequest):
    """
    Generate ML-enhanced crop recommendations.
    
    Returns recommendations with suitability scores (ML-blended when available),
    risk assessments, pest warnings, and planting calendar.
    """
    try:
        # 1. Resolve region
        region, latitude, longitude = _resolve_region(
            request.region_id, request.latitude, request.longitude
        )
        
        # 2. Get weather and season
        weather, season, is_transition, next_season = _get_weather_and_season(
            region, latitude, longitude, request.season
        )
        season_guidance = format_season_guidance(season, is_transition, next_season)
        
        # 3. Determine soil
        if request.soil:
            soil = SoilInfo(
                texture=request.soil.texture,
                ph=request.soil.ph,
                organic_matter=request.soil.organic_matter,
                drainage=request.soil.drainage
            )
        else:
            soil = region.get_default_soil()
            if not soil:
                soil = SoilInfo(texture="Loam", ph=7.0, organic_matter="Medium", drainage="Medium")
        
        # 4. Medium-range forecast (ML-enhanced)
        forecast = forecast_days_17_90(weather, request.planning_days, region_id=region.region_id)

        # 5. Determine irrigation
        irrigation_map = {"None": False, "Limited": True, "Full": True}
        irrigation_available = irrigation_map.get(request.irrigation, True)
        
        # 6. Generate crop recommendations (ML-blended scoring)
        crops = recommend_crops(
            weather_df=weather,
            season=season,
            region_id=region.region_id,
            soil=soil,
            irrigation_available=irrigation_available,
            planning_days=request.planning_days,
        )
        
        # 7. Add risk assessment and pest warnings to each crop
        weather_conditions = {
            'avg_temp_max': float(weather['temp_max'].mean()),
            'avg_temp_min': float(weather['temp_min'].mean()),
            'avg_temp': float(weather['temp_avg'].mean()) if 'temp_avg' in weather.columns else float((weather['temp_max'].mean() + weather['temp_min'].mean()) / 2),
            'total_rainfall': float(forecast.get('expected_rainfall_mm', 0)),
            'avg_humidity': float(weather['humidity'].mean()) if 'humidity' in weather.columns else 65,
            'forecast_days': request.planning_days
        }
        
        for crop_rec in crops[:15]:
            # Risk assessment
            crop_info = {
                'water_requirement_mm': crop_rec['water_required_mm'],
                'drought_tolerance': crop_rec['drought_tolerance'],
                'temp_min': 15,  # General defaults
                'temp_max': 40,
            }
            
            # Try to get actual crop temp limits
            try:
                from src.crops.database import crop_db
                crop_detail = crop_db.get_crop(crop_rec['crop_id'])
                if crop_detail:
                    crop_info['temp_min'] = crop_detail.temp_min
                    crop_info['temp_max'] = crop_detail.temp_max
                    crop_info['temp_optimal_min'] = crop_detail.temp_optimal_min
                    crop_info['temp_optimal_max'] = crop_detail.temp_optimal_max
                    crop_rec['growing_tip'] = getattr(crop_detail, 'growing_tip', '')
                    crop_rec['duration_range'] = list(crop_detail.duration_range)
            except Exception:
                logger.warning("[crop detail lookup failed]\n" + traceback.format_exc())
                crop_rec.setdefault('growing_tip', '')
                crop_rec.setdefault('duration_range', [])
            
            risk = risk_engine.assess_risk(
                crop_info=crop_info,
                weather_forecast=forecast,
                season=season,
                irrigation_available=irrigation_available
            )
            crop_rec['risk_assessment'] = risk
            
            # Pest warnings
            pest_warnings = pest_system.get_warnings(
                crop_rec['crop_id'], weather_conditions, season
            )
            crop_rec['pest_warnings'] = pest_warnings[:3]  # Top 3
        
        # 8. Generate planting calendars for top crops
        calendars = planting_calendar.get_multiple_calendars(crops[:15], season)

        # 8b. LLM Explanation — enrich top 3 crops with farmer-friendly reasoning
        llm_powered = False
        if _LLM_EXPLAINER_AVAILABLE and crops:
            try:
                avg_temp_val = float(weather['temp_avg'].mean()) if 'temp_avg' in weather.columns \
                    else float((weather['temp_max'].mean() + weather['temp_min'].mean()) / 2)
                crops = generate_bulk_explanations(
                    crops=crops,
                    region_name=region.name,
                    region_id=region.region_id,
                    season=season,
                    avg_temp=avg_temp_val,
                    expected_rainfall=float(forecast.get('expected_rainfall_mm', 0)),
                    soil_texture=soil.texture,
                    soil_ph=soil.ph,
                    top_n=3,
                )
                llm_powered = True
            except Exception as _llm_e:
                logger.warning(f"[LLM explainer failed — skipping explanations]\n" + traceback.format_exc())
        
        # 9. Build month-wise forecast (Jan-Dec) for the climate chart
        #    Temperature: live API anchor for current month + zone seasonal shape offset
        #    so each district shows its actual temperature range, not the zone average.
        #    Humidity + rainfall remain zone-based (open-meteo free tier omits them).
        MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        monthly_forecast = []
        try:
            from src.weather.history import get_zone_for_region, get_monthly_climate
            import datetime as _dt

            zone          = get_zone_for_region(region.region_id)
            current_month = _dt.datetime.now().month

            # Live API 16-day mean temperature for this exact district (lat/lon accurate)
            live_anchor = float(weather["temp_avg"].mean()) if "temp_avg" in weather.columns \
                else float((weather["temp_max"].mean() + weather["temp_min"].mean()) / 2)

            # Zone temps give the seasonal *shape* (warmer summer, cooler winter)
            zone_temps  = {m: get_monthly_climate(zone, m)["temperature"] for m in range(1, 13)}
            # Offset = how much this district's real temp differs from its zone average
            temp_offset = live_anchor - zone_temps[current_month]

            for m in range(1, 13):
                clim     = get_monthly_climate(zone, m)
                # Apply same offset to every month -- preserves seasonal curve shape
                # but anchors the whole curve to the district's actual temperature
                adj_temp     = round(zone_temps[m] + temp_offset, 1)
                # temp_max / temp_min: apply same district offset to the zone range
                # so "35–43°C for Thane May" displays correctly on the chart
                adj_temp_max = round(clim.get("temp_max", adj_temp + 7) + temp_offset, 1)
                adj_temp_min = round(clim.get("temp_min", adj_temp - 7) + temp_offset, 1)
                monthly_forecast.append({
                    "month":       MONTH_NAMES[m - 1],
                    "month_num":   m,
                    "temperature": adj_temp,
                    "temp_max":    adj_temp_max,
                    "temp_min":    adj_temp_min,
                    "rainfall":    clim["rainfall"],
                    "humidity":    clim["humidity"],
                })
        except Exception:
            logger.warning("[monthly forecast build failed]\n" + traceback.format_exc())
        forecast["monthly_forecast"] = monthly_forecast
        
        # 10. Build response
        return {
            "region": {
                "region_id": region.region_id,
                "name": region.name,
                "latitude": latitude,
                "longitude": longitude,
                "climate_zone": region.climate_zone
            },
            "season": {
                "current": season,
                "is_transition": is_transition,
                "next_season": next_season,
                "guidance": season_guidance
            },
            "soil": {
                "texture": soil.texture,
                "ph": soil.ph,
                "organic_matter": soil.organic_matter,
                "drainage": soil.drainage,
                "source": "user_provided" if request.soil else "region_default"
            },
            "irrigation": request.irrigation,
            "medium_range_forecast": forecast,
            "recommended_crops": crops[:15],
            "planting_calendars": calendars,
            "total_crops_analyzed": len(crops),
            "llm_powered": llm_powered,
            "llm_note": "Top 3 crops include AI-generated explanations" if llm_powered else "Rule-based scoring only"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("[/recommend endpoint error]\n" + traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# ----------- New ML-Enhanced Endpoints -----------

@app.get("/forecast/{region_id}")
def get_forecast(region_id: str, days: int = 7):
    """
    Get ML-powered weather forecast for a region.
    
    Uses ensemble of LSTM + XGBoost models when available,
    falls back to climatology-based estimation.
    """
    try:
        region = region_manager.get_region_profile(region_id.upper())
        if not region:
            raise HTTPException(status_code=404, detail=f"Region {region_id} not found")
        
        # Fetch current weather
        weather = fetch_weather(region.latitude, region.longitude)
        weather = add_agri_features(weather)
        
        # Generate ML forecast
        forecast = forecast_days_17_90(weather, planning_days=days, region_id=region.region_id)
        
        return {
            "region_id": region.region_id,
            "region_name": region.name,
            "forecast_days": days,
            "current_weather": {
                "avg_temp_max": round(float(weather['temp_max'].mean()), 2),
                "avg_temp_min": round(float(weather['temp_min'].mean()), 2),
                "total_rainfall_recent": round(float(weather['rainfall'].sum()), 2)
            },
            "forecast": forecast
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("[/forecast endpoint error]\n" + traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Forecast error: {str(e)}")


@app.post("/risk-assessment")
def assess_risk(request: RiskRequest):
    """
    Get comprehensive risk assessment for a crop in a region.
    
    Evaluates drought risk, temperature stress, and extreme weather events.
    """
    try:
        region = region_manager.get_region_profile(request.region_id.upper())
        if not region:
            raise HTTPException(status_code=404, detail=f"Region {request.region_id} not found")
        
        # Get crop info
        from src.crops.database import crop_db
        crop = crop_db.get_crop(request.crop_id)
        if not crop:
            raise HTTPException(status_code=404, detail=f"Crop {request.crop_id} not found")
        
        # Fetch weather and forecast
        weather = fetch_weather(region.latitude, region.longitude)
        weather = add_agri_features(weather)
        
        season = request.season or detect_season(datetime.now(), region.region_id)
        forecast = forecast_days_17_90(weather, planning_days=90, region_id=region.region_id)
        
        irrigation_map = {"None": False, "Limited": True, "Full": True}
        
        # Run risk assessment
        crop_info = {
            'water_requirement_mm': crop.water_requirement_mm,
            'drought_tolerance': crop.drought_tolerance,
            'temp_min': crop.temp_min,
            'temp_max': crop.temp_max,
            'temp_optimal_min': crop.temp_optimal_min,
            'temp_optimal_max': crop.temp_optimal_max
        }
        
        risk = risk_engine.assess_risk(
            crop_info=crop_info,
            weather_forecast=forecast,
            season=season,
            irrigation_available=irrigation_map.get(request.irrigation, True)
        )
        
        return {
            "region_id": region.region_id,
            "crop": crop.common_name,
            "crop_id": crop.crop_id,
            "season": season,
            "risk_assessment": risk
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("[/risk-assessment endpoint error]\n" + traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Risk assessment error: {str(e)}")


@app.get("/pest-warnings/{region_id}")
def get_pest_warnings(region_id: str, crop_id: Optional[str] = None):
    """
    Get pest and disease warnings for a region based on current weather conditions.
    
    Optionally filter by crop_id.
    """
    try:
        region = region_manager.get_region_profile(region_id.upper())
        if not region:
            raise HTTPException(status_code=404, detail=f"Region {region_id} not found")
        
        # Get current weather
        weather = fetch_weather(region.latitude, region.longitude)
        
        weather_conditions = {
            'avg_temp_max': float(weather['temp_max'].mean()),
            'avg_temp_min': float(weather['temp_min'].mean()),
            'avg_temp': float((weather['temp_max'].mean() + weather['temp_min'].mean()) / 2),
            'total_rainfall': float(weather['rainfall'].sum()),
            'avg_humidity': float(weather['humidity'].mean()) if 'humidity' in weather.columns else 65,
            'forecast_days': len(weather)
        }
        
        if crop_id:
            warnings = pest_system.get_warnings(crop_id, weather_conditions)
            return {
                "region_id": region.region_id,
                "crop_id": crop_id,
                "weather_conditions": weather_conditions,
                "warnings": warnings
            }
        else:
            all_warnings = pest_system.get_region_warnings(weather_conditions)
            return {
                "region_id": region.region_id,
                "weather_conditions": weather_conditions,
                "warnings_by_crop": all_warnings,
                "total_warnings": sum(len(w) for w in all_warnings.values())
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("[/pest-warnings endpoint error]\n" + traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Pest warning error: {str(e)}")


@app.get("/planting-calendar/{crop_id}")
def get_planting_calendar_endpoint(
    crop_id: str,
    season: Optional[str] = None,
    region_id: Optional[str] = None
):
    """
    Get planting calendar with milestone dates for a crop.
    
    Returns sowing date, growth phases, and harvest date with care tips.
    """
    try:
        from src.crops.database import crop_db
        crop = crop_db.get_crop(crop_id)
        if not crop:
            raise HTTPException(status_code=404, detail=f"Crop {crop_id} not found")
        
        if not season:
            season = detect_season(datetime.now())
        
        calendar = planting_calendar.get_calendar(
            crop_id=crop.crop_id,
            season=season,
            duration_days=crop.duration_days,
            crop_name=crop.common_name,
            region_id=region_id
        )
        
        return {
            "crop": crop.common_name,
            "calendar": calendar
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error("[/planting-calendar endpoint error]\n" + traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Calendar error: {str(e)}")


# ----------- Live Weather Endpoint -----------

@app.get("/weather/now/{region_id}")
def get_current_weather(region_id: str):
    """
    Fetch today's real-time weather for a region using Open-Meteo (free, no API key).

    Returns temperature high/low/avg for today, recent 30-day rainfall total,
    humidity estimate, and a human-readable summary string.
    This endpoint is used by the chat widget to answer temperature questions accurately.
    """
    try:
        region = region_manager.get_region_profile(region_id.upper())
        if not region:
            raise HTTPException(status_code=404, detail=f"Region {region_id} not found")

        from src.weather.fetcher import fetch_weather
        from src.ml.pipeline import add_agri_features
        from datetime import datetime as _dt

        wx = fetch_weather(region.latitude, region.longitude,
                           region_id=region.region_id, season="")
        wx = add_agri_features(wx)

        # Row 14 = today (14 past days included by Open-Meteo, index 14 = today)
        today_idx = min(14, len(wx) - 1)
        today = wx.iloc[today_idx]

        t_max  = round(float(today["temp_max"]), 1)
        t_min  = round(float(today["temp_min"]), 1)
        t_avg  = round(float(today.get("temp_avg", (today["temp_max"] + today["temp_min"]) / 2)), 1)
        humidity = round(float(today.get("humidity", 60)), 1)
        total_rain = round(float(wx["rainfall"].sum()), 1)
        today_date = _dt.now().strftime("%d %b %Y")

        summary = (
            f"Today ({today_date}): {t_max}\u00b0C high / {t_min}\u00b0C low (avg {t_avg}\u00b0C), "
            f"recent 30-day total rainfall {total_rain} mm"
        )

        # Warm the weather cache so the streaming chat picks it up
        if set_weather_cache and _LLM_CHAT_AVAILABLE:
            set_weather_cache(region.region_id, summary)

        return {
            "region_id": region.region_id,
            "region_name": region.name,
            "date": today_date,
            "temp_max": t_max,
            "temp_min": t_min,
            "temp_avg": t_avg,
            "humidity_pct": humidity,
            "rainfall_30d_mm": total_rain,
            "summary": summary,
            "source": "Open-Meteo live API"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("[/weather/now endpoint error]\n" + traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Weather fetch error: {str(e)}")


# ----------- LLM Chat Endpoint -----------

@app.post("/chat")
def farmer_chat(request: ChatRequest):
    """
    Answer a farmer's free-form question using Gemini LLM with multi-turn memory.

    Accepts any farming-related question and returns a concise,
    region-aware answer with updated conversation history.
    Falls back gracefully if LLM unavailable.
    """
    try:
        if not _LLM_CHAT_AVAILABLE or answer_farmer_question is None:
            return {
                "answer": (
                    "AI chat is unavailable. Start Ollama locally (ollama serve) or "
                    "add GEMINI_API_KEY to your .env file as a fallback. "
                    "All crop recommendation features work without AI chat."
                ),
                "llm_available": False,
                "history": []
            }

        # Resolve full region profile for richer context
        region_name = ""
        state_name = ""
        climate_zone = ""
        soil_info = ""
        weather_summary = ""

        if request.region_id:
            try:
                robj = region_manager.get_region_profile(request.region_id.upper())
                if robj:
                    region_name  = robj.name
                    state_name   = getattr(robj, 'state', '')
                    climate_zone = getattr(robj, 'climate_zone', '')
                    # Default soil summary
                    default_soil = robj.get_default_soil() if hasattr(robj, 'get_default_soil') else None
                    if default_soil:
                        soil_info = f"{default_soil.texture}, pH {default_soil.ph}, {default_soil.organic_matter} organic matter"
                    # Live weather summary — use TODAY's row (index 14 = current day after 14 past days)
                    try:
                        from src.weather.fetcher import fetch_weather
                        from src.ml.pipeline import add_agri_features
                        wx = fetch_weather(robj.latitude, robj.longitude,
                                           region_id=robj.region_id,
                                           season=request.season or "")
                        wx = add_agri_features(wx)
                        # Row 14 is today (14 past days + today)
                        today_idx = min(14, len(wx) - 1)
                        today = wx.iloc[today_idx]
                        t_max  = round(float(today['temp_max']), 1)
                        t_min  = round(float(today['temp_min']), 1)
                        t_avg  = round(float(today.get('temp_avg', (today['temp_max'] + today['temp_min']) / 2)), 1)
                        total_rain = round(float(wx['rainfall'].sum()), 1)
                        from datetime import datetime as _dt
                        today_date = _dt.now().strftime('%d %b %Y')
                        weather_summary = (
                            f"Today ({today_date}): {t_max}°C high / {t_min}°C low (avg {t_avg}°C), "
                            f"recent 30-day total rainfall {total_rain} mm"
                        )
                    except Exception:
                        pass
                else:
                    region_name = request.region_id
            except Exception:
                region_name = request.region_id or ""

        # Populate weather cache so the streaming endpoint (and next chat turn)
        # can reuse it without re-fetching live data
        if weather_summary and set_weather_cache and request.region_id:
            set_weather_cache(request.region_id.upper(), weather_summary)

        answer, updated_history = answer_farmer_question(
            question=request.question,
            region_id=request.region_id or "",
            region_name=region_name,
            season=request.season or "",
            history=request.history or [],
            crop_context=request.crop_context or "",
            state_name=state_name,
            climate_zone=climate_zone,
            soil_info=soil_info,
            weather_summary=weather_summary,
        )

        return {
            "answer": answer,
            "llm_available": True,
            "region_context": region_name or request.region_id or "General",
            "season_context": request.season or "General",
            "history": updated_history
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


# ----------- LLM Streaming Chat Endpoint -----------

class StreamChatRequest(BaseModel):
    question: str = Field(..., description="Farmer's question in English")
    region_id: Optional[str] = Field("", description="Region ID for context")
    season: Optional[str] = Field("", description="Current season")
    history: Optional[List[dict]] = Field(default_factory=list, description="Conversation history")
    crop_context: Optional[str] = Field("", description="Top crops from last recommendation")


@app.post("/chat/stream")
def farmer_chat_stream(request: StreamChatRequest):
    """
    Stream Gemini chat responses using Server-Sent Events (SSE).

    Yields text chunks as they arrive, giving instant perceived response.
    Region context (name, state, soil, climate zone) is resolved from fast
    in-memory lookups only. Weather is used only if already cached — we never
    block on a live weather API call here, keeping first-token latency low.
    The final SSE event is 'data: [DONE]{json-encoded-history}'.
    """
    if not _LLM_CHAT_AVAILABLE or stream_farmer_answer is None:
        def _no_llm():
            yield "data: AI chat is unavailable. Start Ollama (ollama serve) or add GEMINI_API_KEY to .env.\n\n"
            yield "data: [DONE][]\n\n"
        return StreamingResponse(_no_llm(), media_type="text/event-stream")

    # --- Fast in-memory context resolution (no I/O) ---
    region_name = state_name = climate_zone = soil_info = weather_summary = ""
    robj_resolved = None
    if request.region_id:
        try:
            robj = region_manager.get_region_profile(request.region_id.upper())
            if robj:
                robj_resolved = robj
                region_name  = robj.name
                state_name   = getattr(robj, "state", "")
                climate_zone = getattr(robj, "climate_zone", "")
                default_soil = robj.get_default_soil() if hasattr(robj, "get_default_soil") else None
                if default_soil:
                    soil_info = (
                        f"{default_soil.texture}, pH {default_soil.ph}, "
                        f"{default_soil.organic_matter} organic matter"
                    )
        except Exception:
            region_name = request.region_id or ""

    # Try cached weather first (fast); if missing, fetch live (slower but accurate)
    if _LLM_CHAT_AVAILABLE and request.region_id:
        try:
            from src.services.llm_chat import _get_cached_weather
            weather_summary = _get_cached_weather(request.region_id.upper())
        except Exception:
            pass

    # If cache miss and we have a region object, fetch live weather now so the
    # chatbot can answer temperature questions even before a /recommend call.
    if not weather_summary and robj_resolved is not None:
        try:
            from src.weather.fetcher import fetch_weather
            from src.ml.pipeline import add_agri_features
            from datetime import datetime as _dt
            wx = fetch_weather(robj_resolved.latitude, robj_resolved.longitude,
                               region_id=robj_resolved.region_id, season="")
            wx = add_agri_features(wx)
            today_idx = min(14, len(wx) - 1)
            today_row = wx.iloc[today_idx]
            t_max  = round(float(today_row["temp_max"]), 1)
            t_min  = round(float(today_row["temp_min"]), 1)
            t_avg  = round(float(today_row.get("temp_avg",
                           (today_row["temp_max"] + today_row["temp_min"]) / 2)), 1)
            total_rain = round(float(wx["rainfall"].sum()), 1)
            today_date = _dt.now().strftime("%d %b %Y")
            weather_summary = (
                f"Today ({today_date}): {t_max}\u00b0C high / {t_min}\u00b0C low "
                f"(avg {t_avg}\u00b0C), recent 30-day total rainfall {total_rain} mm"
            )
            # Warm the cache for future turns
            if set_weather_cache:
                set_weather_cache(robj_resolved.region_id, weather_summary)
        except Exception:
            pass

    gen = stream_farmer_answer(
        question=request.question,
        region_id=request.region_id or "",
        region_name=region_name,
        season=request.season or "",
        history=request.history or [],
        crop_context=request.crop_context or "",
        state_name=state_name,
        climate_zone=climate_zone,
        soil_info=soil_info,
        weather_summary=weather_summary,
    )
    return StreamingResponse(gen, media_type="text/event-stream")


# ===============================================================================
# Global Agent Endpoints -- LLaMA + Ollama Web Search
# ===============================================================================

@app.get("/api/countries")
def api_get_countries():
    """Return all supported countries for the global location selector."""
    if not _AGENTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Location agent not available")
    return {"countries": get_countries()}


@app.get("/api/states/{country_code}")
def api_get_states(country_code: str):
    """Return states/provinces for a given country code."""
    if not _AGENTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Location agent not available")
    states = get_states(country_code.upper())
    return {"states": states}


@app.get("/api/districts/{country_code}/{state_code}")
def api_get_districts(country_code: str, state_code: str):
    """Return districts for a given country and state code."""
    if not _AGENTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Location agent not available")
    districts = get_districts(country_code.upper(), state_code.upper())
    return {"districts": districts}


class AnalyzeRequest(BaseModel):
    country_code: str  = Field(..., description="ISO country code e.g. IN, US, BR")
    country_name: str  = Field(..., description="Country name e.g. India")
    state_code:   str  = Field(..., description="State code e.g. MH")
    state_name:   str  = Field(..., description="State name e.g. Maharashtra")
    district:     str  = Field(..., description="District name e.g. Nashik")
    lat:          Optional[float] = Field(None, description="Latitude (auto-resolved if not given)")
    lon:          Optional[float] = Field(None, description="Longitude (auto-resolved if not given)")
    irrigation:   str  = Field("Limited", description="None / Limited / Full")
    planning_days:int  = Field(90, ge=15, le=365)
    soil_texture: Optional[str]  = Field(None)
    soil_ph:      Optional[float]= Field(None)
    soil_organic: Optional[str]  = Field(None)
    soil_drainage:Optional[str]  = Field(None)


@app.post("/api/analyze")
def api_analyze(request: AnalyzeRequest):
    """
    Main global analysis endpoint.

    1. Resolves coordinates for the district
    2. Runs LLaMA agent with Ollama web search to gather real data
    3. Runs crop recommendation agent
    4. Returns dashboard data: current weather, 6-month forecast, crops, market prices
    """
    if not _AGENTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Agent system not available. Check OLLAMA_API_KEY in .env.")

    try:
        # Resolve coordinates
        lat = request.lat
        lon = request.lon
        if lat is None or lon is None:
            lat, lon = resolve_coordinates(request.country_code, request.state_code, request.district)

        # Step 1: Data gathering agent — real weather + climatology + LLM enrichment
        gathered = gather_location_data(
            country=request.country_name,
            state=request.state_name,
            district=request.district,
            lat=lat,
            lon=lon,
            state_code=request.state_code,  # passed for India zone mapping
        )

        # Step 2: Soil override if user provided
        soil_override = None
        if request.soil_texture:
            soil_override = {
                "type":          request.soil_texture,
                "ph":            request.soil_ph or 7.0,
                "organic_matter":request.soil_organic or "Medium",
                "drainage":      request.soil_drainage or "Medium",
            }

        # Step 3: Crop recommendation agent
        crops = recommend_crops_agent(
            country=request.country_name,
            state=request.state_name,
            district=request.district,
            gathered_data=gathered,
            irrigation=request.irrigation,
            planning_days=request.planning_days,
            soil_override=soil_override,
        )

        # Step 4: Update LLM chat weather cache so chatbot knows live conditions
        if set_weather_cache and _LLM_CHAT_AVAILABLE and gathered.get("current"):
            cur = gathered["current"]
            summary = (
                f"Current conditions in {request.district}, {request.state_name}: "
                f"{cur.get('temperature_c','?')} C, "
                f"humidity {cur.get('humidity_pct','?')}%, "
                f"rainfall last 7 days: {cur.get('rainfall_7d_mm','?')} mm"
            )
            cache_key = f"{request.country_code}_{request.state_code}_{request.district}".upper()
            set_weather_cache(cache_key, summary)

        return {
            "location": {
                "country_code": request.country_code,
                "country_name": request.country_name,
                "state_code":   request.state_code,
                "state_name":   request.state_name,
                "district":     request.district,
                "lat": lat,
                "lon": lon,
            },
            "gathered_data": gathered,
            "recommended_crops": crops,
            "agent": "llama3.2 + ollama_web_search",
            "version": "3.0",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[/api/analyze error]\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

# -- SSE Streaming Analyze Endpoint -------------------------------------------
# Emits real-time progress. Weather + LLM enrichment run IN PARALLEL for speed.

@app.post('/api/analyze/stream')
def api_analyze_stream(request: AnalyzeRequest):
    if not _AGENTS_AVAILABLE:
        raise HTTPException(status_code=503, detail='Agent system not available.')
    import json as _json
    import concurrent.futures as _cf
    import datetime as _dt

    def _evt(step, pct, msg, data=None):
        payload = {'step': step, 'pct': pct, 'msg': msg}
        if data is not None:
            payload['data'] = data
        return 'data: ' + _json.dumps(payload) + '\n\n'

    def generate():
        try:
            from src.agents.data_gathering_agent import (
                _fetch_openmeteo_current, _build_forecast_6month,
                _llm_enrich, _get_world_zone, _ZONE_CLIMATE,
                _ZONE_SOIL_DEFAULTS, _default_market_prices, _guess_season,
            )
            current_month = _dt.datetime.now().month
            loc_str = request.district + ', ' + request.state_name + ', ' + request.country_name

            # Step 1: Resolve coordinates
            yield _evt(1, 12, 'Resolving location: ' + request.district + ', ' + request.state_name + '...')
            lat = request.lat
            lon = request.lon
            if lat is None or lon is None:
                lat, lon = resolve_coordinates(request.country_code, request.state_code, request.district)
            yield _evt(1, 22, 'Location resolved: ' + str(round(lat,3)) + 'N, ' + str(round(lon,3)) + 'E')

            # Steps 2+4: Fire weather + LLM IN PARALLEL
            yield _evt(2, 30, 'Fetching live weather & running AI analysis in parallel...')
            executor = _cf.ThreadPoolExecutor(max_workers=2)
            wx_future  = executor.submit(_fetch_openmeteo_current, lat, lon)
            llm_future = executor.submit(_llm_enrich, loc_str, request.country_name, 25.0, current_month)

            try:
                live_wx = wx_future.result(timeout=10)
            except Exception:
                live_wx = None
            executor.shutdown(wait=False)

            if live_wx:
                current = live_wx
                yield _evt(2, 42, 'Live weather: ' + str(current['temperature_c']) + 'C  |  Humidity: ' + str(current['humidity_pct']) + '%  |  Rain 7d: ' + str(current['rainfall_7d_mm']) + ' mm')
            else:
                wz_t = _get_world_zone(request.country_name, lat)
                zc = (_ZONE_CLIMATE.get(wz_t) or _ZONE_CLIMATE['Subtropical']).get(current_month, {})
                ta = zc.get('temp', 25)
                current = {'temperature_c': ta, 'temp_max_c': ta+6, 'temp_min_c': ta-6,
                    'humidity_pct': zc.get('hum',65), 'soil_temp_c': round(ta-2,1),
                    'rainfall_7d_mm': round(zc.get('rain',60)/4,1),
                    'wind_kmh': 12.0, 'uv_index': 6.0, 'feels_like_c': round(ta+2,1)}
                yield _evt(2, 42, 'Using zone-based weather estimate (API unavailable)')

            # Step 3: Forecast - pure computation, instant
            yield _evt(3, 55, 'Building 6-month seasonal forecast...')
            forecast_6month = _build_forecast_6month(
                country=request.country_name, state_code=request.state_code,
                lat=lat, live_temp_avg=current['temperature_c'], current_month=current_month)
            season = _guess_season(current_month, request.country_name)
            yield _evt(3, 65, 'Forecast ready: ' + str(len(forecast_6month)) + ' months  |  Season: ' + season)

            # Step 4: Collect LLM result (already running in background)
            yield _evt(4, 70, 'Collecting AI soil and market analysis...')
            try:
                llm_data = llm_future.result(timeout=3)
            except Exception:
                llm_data = None

            world_zone = _get_world_zone(request.country_name, lat)
            if llm_data and llm_data.get('soil') and isinstance(llm_data['soil'].get('ph'), (int, float)):
                soil = llm_data['soil']
                soil.setdefault('type', _ZONE_SOIL_DEFAULTS.get(world_zone, {}).get('type', 'Loam'))
                soil.setdefault('organic_matter', 'Medium')
                soil.setdefault('drainage', 'Medium')
            else:
                soil = dict(_ZONE_SOIL_DEFAULTS.get(world_zone,
                    {'type': 'Loam', 'ph': 7.0, 'organic_matter': 'Medium', 'drainage': 'Good'}))

            if request.soil_texture:
                soil = {'type': request.soil_texture, 'ph': request.soil_ph or soil.get('ph', 7.0),
                    'organic_matter': request.soil_organic or 'Medium',
                    'drainage': request.soil_drainage or 'Medium'}

            market_prices = (
                llm_data.get('market_prices')
                if llm_data and len(llm_data.get('market_prices', {})) >= 2
                else _default_market_prices(request.country_name, world_zone)
            )
            zlm = {'Tropical':'Tropical','Subtropical':'Subtropical','Arid':'Arid',
                   'Mediterranean':'Mediterranean','Temperate':'Temperate','Continental':'Continental',
                   'Temperate_Americas':'Temperate','Tropical_Americas':'Tropical',
                   'Subtropical_S':'Subtropical','Arid_Oceania':'Semi-Arid'}
            climate_zone = (llm_data.get('climate_zone') if llm_data and llm_data.get('climate_zone')
                else zlm.get(world_zone, 'Subtropical'))
            district_summary = (llm_data.get('district_summary') if llm_data and llm_data.get('district_summary')
                else (request.district + ' is an agricultural district in ' + request.state_name +
                      ', ' + request.country_name + '. ' + climate_zone + ' climate, ' +
                      str(current['temperature_c']) + 'C current temperature.'))

            gathered = {'current': current, 'forecast_6month': forecast_6month, 'soil': soil,
                'season': season, 'climate_zone': climate_zone,
                'market_prices': market_prices, 'district_summary': district_summary}
            yield _evt(4, 82, 'Soil: ' + str(soil.get('type','?')) + ' pH ' + str(soil.get('ph','?')) + '  |  ' + str(len(market_prices)) + ' crop prices ready')

            # Step 5: Crop recommendations
            yield _evt(5, 88, 'Running crop suitability ranking for ' + request.district + '...')
            crops = recommend_crops_agent(
                country=request.country_name, state=request.state_name,
                district=request.district, gathered_data=gathered,
                irrigation=request.irrigation, planning_days=request.planning_days, soil_override=None)
            yield _evt(5, 96, str(len(crops)) + ' crops ranked by suitability')

            if set_weather_cache and _LLM_CHAT_AVAILABLE:
                cur = gathered['current']
                wx_sum = 'Temp ' + str(cur.get('temperature_c','?')) + 'C, humidity ' + str(cur.get('humidity_pct','?')) + '%, rain7d ' + str(cur.get('rainfall_7d_mm','?')) + 'mm'
                ck = (request.country_code+'_'+request.state_code+'_'+request.district).upper()
                set_weather_cache(ck, wx_sum)

            result = {'location': {'country_code': request.country_code, 'country_name': request.country_name,
                'state_code': request.state_code, 'state_name': request.state_name,
                'district': request.district, 'lat': lat, 'lon': lon},
                'gathered_data': gathered, 'recommended_crops': crops,
                'agent': 'open-meteo+zone+llm(parallel)', 'version': '3.1'}
            yield _evt('done', 100, 'Analysis complete!', result)

        except Exception as exc:
            import traceback as _tb
            logger.error('[/api/analyze/stream error]\n' + _tb.format_exc())
            yield 'data: ' + _json.dumps({'step': 'error', 'pct': 0, 'msg': str(exc)}) + '\n\n'

    return StreamingResponse(generate(), media_type='text/event-stream')
