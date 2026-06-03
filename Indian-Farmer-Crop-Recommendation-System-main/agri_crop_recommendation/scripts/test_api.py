"""Test ML forecast integration."""
import urllib.request
import json

BASE = "http://127.0.0.1:8000"

# Test Forecast endpoint
print("=" * 50)
print("  Testing /forecast/PUNE")
print("=" * 50)
try:
    r = urllib.request.urlopen(f"{BASE}/forecast/MH_PUNE?days=7")
    data = json.loads(r.read())
    print(f"  Region: {data['region_name']}")
    print(f"  Source: {data['forecast'].get('forecast_source', 'N/A')}")
    print(f"  Confidence: {data['forecast'].get('confidence', 'N/A')}")
    if data['forecast'].get('daily_predictions'):
        preds = data['forecast']['daily_predictions']
        print(f"  Daily predictions: {len(preds)} days")
        for p in preds[:3]:
            print(f"    Day {p['day']}: temp_max={p.get('temp_max', 'N/A')}, temp_min={p.get('temp_min', 'N/A')}, rain={p.get('rainfall', 'N/A')}")
    else:
        print("  [!] No daily predictions returned")
    if data['forecast'].get('ml_summary'):
        s = data['forecast']['ml_summary']
        print(f"  ML Summary: avg_temp={s.get('avg_temp')}, total_rain={s.get('total_rainfall')}")
except Exception as e:
    print(f"  [FAIL] {e}")

# Test recommendation with ML
print("\n" + "=" * 50)
print("  Testing /recommend with ML scoring")
print("=" * 50)
try:
    body = json.dumps({"region_id": "MH_PUNE", "irrigation": "Limited", "planning_days": 90}).encode()
    req = urllib.request.Request(f"{BASE}/recommend", data=body, headers={"Content-Type": "application/json"})
    r = urllib.request.urlopen(req)
    data = json.loads(r.read())
    src = data['medium_range_forecast'].get('forecast_source', 'N/A')
    print(f"  Forecast source: {src}")
    top = data['recommended_crops'][0]
    print(f"  Top crop: {top['crop']} -- Score: {top['suitability_score']} [{top['score_source']}]")
    if data['medium_range_forecast'].get('daily_predictions'):
        preds = data['medium_range_forecast']['daily_predictions']
        print(f"  Daily predictions: {len(preds)} days")
        temps = [p['temp_max'] for p in preds]
        unique_temps = len(set(temps))
        print(f"  Unique temp_max values: {unique_temps} (should be > 1 for ML)")
        for p in preds[:5]:
            print(f"    Day {p['day']}: max={p['temp_max']}, min={p['temp_min']}, rain={p['rainfall']}")
    else:
        print("  No daily predictions")
except Exception as e:
    print(f"  [FAIL] {e}")

print("\n  Done!")
