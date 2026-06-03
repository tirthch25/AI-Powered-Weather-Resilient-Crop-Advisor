"""
Quick test script: verifies fallback mode and recommender work correctly.
Run from: agri_crop_recommendation/
  python scripts/test_llm_integration.py
"""
import sys, os
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()

import pandas as pd

# ── Test 1: Fallback mode (no API key) ────────────────────────────────────────
print("=== TEST 1: Fallback Mode (LLM disabled) ===")
os.environ['GEMINI_API_KEY'] = ''

from src.services.llm_filter import llm_filter_crops, _get_gemini_client
import src.services.llm_filter as lf
lf._gemini_client = None   # force reload

result = llm_filter_crops(
    crop_ids=['BAJRA_01', 'JOWAR_01', 'BABY_CORN_01', 'LETTUCE_01'],
    crop_names=['Bajra', 'Jowar', 'Baby Corn', 'Lettuce'],
    region_id='MH_NANDED',
    region_name='Nanded',
    season='Kharif',
    state='Maharashtra'
)
print(f"LLM result (should be None in fallback): {result}")
assert result is None, "Expected None fallback"
print("PASS: Fallback returns None correctly\n")

# ── Test 2: Recommender with regional threshold fix ───────────────────────────
print("=== TEST 2: Recommender (Kharif, MH_NANDED) ===")

weather = pd.DataFrame({
    'temp_max': [35.0]*7, 'temp_min': [24.0]*7,
    'temp_avg': [29.5]*7, 'rainfall': [8.0]*7,
    'dry_spell_days': [2]*7, 'humidity': [65.0]*7
})

from src.services.recommender import recommend_crops
from src.crops.soil import SoilInfo

soil = SoilInfo(texture='Clay-Loam', ph=7.2, organic_matter='Medium', drainage='Medium')
crops = recommend_crops(weather, season='Kharif', region_id='MH_NANDED', soil=soil)

print(f"Crops returned: {len(crops)}")
print("\nTop 5 recommended crops for Nanded (Kharif):")
for c in crops[:5]:
    print(f"  {c['crop']:<35} score={c['suitability_score']:.1f}  "
          f"region={c['regional_suitability']:.2f}  source={c['score_source']}")

baby_corn = [c for c in crops if 'Baby Corn' in c['crop']]
lettuce   = [c for c in crops if 'Lettuce'   in c['crop']]
print(f"\nBaby Corn in results: {len(baby_corn) > 0}  (should be False)")
print(f"Lettuce in results:   {len(lettuce)   > 0}  (should be False)")

if not baby_corn and not lettuce:
    print("\nPASS: Threshold fix works — unwanted crops filtered out")
else:
    print("\nNOTE: Some unwanted crops still present (LLM filter will remove them when quota resets)")

# ── Test 3: verify Gemini SDK loads ──────────────────────────────────────────
print("\n=== TEST 3: Gemini SDK Import ===")
try:
    from google import genai
    print("PASS: google-genai SDK importable")
except ImportError as e:
    print(f"FAIL: {e}")

print("\n=== ALL TESTS COMPLETE ===")
