import json
with open('data/reference/world_locations.json', encoding='utf-8') as f:
    data = json.load(f)

countries = data.get('countries', [])
states_map = data.get('states', {})
districts = data.get('districts', {})

print('Countries:', len(countries))
print('State entries:', len(states_map), 'country codes with states')
print('District keys:', len(districts))
print()

# Find states that have NO districts
missing = []
for cc, state_list in states_map.items():
    for s in state_list:
        key = cc + '_' + s['code']
        if key not in districts or len(districts[key]) == 0:
            missing.append(key)

print('States with NO districts:', len(missing))
print('First 20:', missing[:20])
print()
print('Sample district keys:', list(districts.keys())[:10])
