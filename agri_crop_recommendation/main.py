"""
Quick sanity test — runs a single recommendation against MH_PUNE.
Run from the agri_crop_recommendation/ directory: python main.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from datetime import datetime
from src.weather.fetcher import fetch_weather
from src.ml.pipeline import add_agri_features
from src.weather.forecast import forecast_days_17_90
from src.services.recommender import recommend_crops
from src.utils.seasons import detect_season
from src.crops.soil import SoilInfo

LAT, LON = 18.5204, 73.8567   # Pune
REGION_ID = "MH_PUNE"

weather = fetch_weather(LAT, LON, region_id=REGION_ID)
weather = add_agri_features(weather)
season  = detect_season(datetime.now(), REGION_ID)

forecast = forecast_days_17_90(weather, region_id=REGION_ID)
print("\n[Weather] Days 17-90 Outlook")
print(f"  Avg temp : {forecast.get('expected_avg_temp')}C")
print(f"  Rainfall : {forecast.get('expected_rainfall_mm')} mm")
print(f"  Source   : {forecast.get('forecast_source')}")

soil = SoilInfo(texture="Clay-Loam", ph=7.0, organic_matter="Medium", drainage="Medium")
results = recommend_crops(
    weather_df=weather,
    season=season,
    region_id=REGION_ID,
    soil=soil,
    irrigation_available=True,
    planning_days=90
)
print(f"\n[Crops] Top Recommendations ({season} season):\n")
for i, r in enumerate(results[:5], 1):
    src = r.get('score_source', '')
    print(f"  {i}. {r.get('crop', ''):<30} score: {r.get('suitability_score', 'N/A')}  [{src}]")

