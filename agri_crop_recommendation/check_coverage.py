import json, re

data = json.loads(open('data/reference/world_locations.json', encoding='utf-8').read())
world_countries = [(c['code'], c['name'].lower()) for c in data.get('countries', [])]

crop_agent = open('src/agents/crop_agent.py', encoding='utf-8').read()

# Extract hints section
hints_section = crop_agent.split('_COUNTRY_CROP_HINTS: Dict[str, str] = {')[1].split('\n}\n')[0]
hint_keys = re.findall(r'"([a-z][a-z\s]+)":\s+"', hints_section)

print(f'Countries in world_locations.json: {len(world_countries)}')
print(f'Countries in _COUNTRY_CROP_HINTS: {len(hint_keys)}')
print()
print('=== MISSING: In world_locations but NOT in crop hints ===')
missing = []
for code, name in world_countries:
    found = any(name in h or h in name for h in hint_keys)
    if not found:
        missing.append((code, name))
        print(f'  [{code}] {name}')

print()
print('=== PRESENT: countries found in both ===')
for code, name in world_countries:
    found = any(name in h or h in name for h in hint_keys)
    if found:
        match = [h for h in hint_keys if name in h or h in name][0]
        print(f'  [{code}] {name} -> hint key: "{match}"')

print()
# Also check _COUNTRY_TO_ZONE (forecast zone)
zone_section = crop_agent.split('_COUNTRY_TO_ZONE: Dict[str, str] = {')[1].split('\n}\n')[0]
zone_keys = re.findall(r'"([a-z][a-z\s\'"]+)":', zone_section)
print(f'Countries in _COUNTRY_TO_ZONE (forecast): {len(zone_keys)}')
print()
print('=== NOT in _COUNTRY_TO_ZONE forecast zone map ===')
for code, name in world_countries:
    found = any(name in z or z in name for z in zone_keys)
    if not found:
        print(f'  [{code}] {name} (uses lat-based fallback)')

print()
# Also check seasons.py - India-only
print('=== SEASONS.PY - India-centric check ===')
seasons_content = open('src/utils/seasons.py', encoding='utf-8').read()
if 'Kharif' in seasons_content and 'Spring' not in seasons_content:
    print('  WARNING: seasons.py is India-only (Kharif/Rabi/Zaid). No Spring/Summer/Autumn/Winter for global!')
else:
    print('  OK: seasons.py has both India and global season detection')
    
# Check data_gathering_agent season detection
dga = open('src/agents/data_gathering_agent.py', encoding='utf-8').read()
print()
print('=== DATA GATHERING AGENT - _guess_season coverage ===')
if 'Spring' in dga and 'Summer' in dga and 'Autumn' in dga and 'Winter' in dga:
    print('  OK: _guess_season covers Spring/Summer/Autumn/Winter for non-India countries')
    if 'Kharif' in dga and 'Rabi' in dga:
        print('  OK: also covers Kharif/Rabi/Zaid for India/South Asia')
else:
    print('  ISSUE: _guess_season missing global seasons')

# Check database.py - India only?
db_content = open('src/crops/database.py', encoding='utf-8').read()
print()
print('=== DATABASE.PY COVERAGE ===')
if 'ZONE_REGIONS' in db_content:
    if 'PB_JALANDHAR' in db_content:
        print('  OK: Punjab/Jalandhar region IDs included')
    if 'UP_LUCKNOW' in db_content:
        print('  OK: Uttar Pradesh included')
    if 'GJ_AHMEDABAD' in db_content:
        print('  OK: Gujarat included')
    
# Check if there are global (non-India) crop entries in CROPS_DATA
if 'CROPS_DATA' in db_content:
    global_hint = 'India' in db_content and 'Maharashtra' in db_content
    if global_hint and 'Tomato' in db_content:
        print('  NOTE: database.py is India-centric with regional suitability for Indian districts')
        print('  NOTE: For non-India countries, crop_agent.py LLM pipeline handles recommendations')
