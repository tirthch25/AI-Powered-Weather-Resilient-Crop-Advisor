import requests, json

body = {'region_id': 'MH_PUNE', 'irrigation': 'Limited', 'planning_days': 90}
r = requests.post('http://localhost:8000/recommend', json=body, timeout=30)
d = r.json()
print('STATUS:', r.status_code)

if r.status_code != 200:
    print('ERROR:', json.dumps(d, indent=2))
else:
    print('Region:', d.get('region', {}).get('name'))
    print('Season:', d.get('season', {}).get('current'))
    crops = d.get('recommended_crops', [])
    print('Crops count:', len(crops))
    for c in crops[:5]:
        print(f"  {c['crop']:30s} score={c['suitability_score']}")
    forecast = d.get('medium_range_forecast', {})
    print('Forecast keys:', list(forecast.keys()))
    risk = d.get('risk_assessment', {})
    print('Risk assessment:', list(risk.keys()))

# Also test a non-Maharashtra region
print()
body2 = {'region_id': 'UP_LUCKNOW', 'irrigation': 'Full', 'planning_days': 90}
r2 = requests.post('http://localhost:8000/recommend', json=body2, timeout=30)
d2 = r2.json()
print('UP Lucknow - STATUS:', r2.status_code)
print('UP Lucknow - Crops:', len(d2.get('recommended_crops', [])))
