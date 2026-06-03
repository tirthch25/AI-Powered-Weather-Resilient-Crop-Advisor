"""Final verification: planning_days filter using min_duration_days from API."""
import urllib.request, json

def test(days):
    body = json.dumps({'region_id': 'MH_PUNE', 'irrigation': 'Limited', 'planning_days': days}).encode()
    req = urllib.request.Request(
        'http://localhost:8000/recommend',
        data=body,
        headers={'Content-Type': 'application/json'}
    )
    r = urllib.request.urlopen(req)
    data = json.loads(r.read())
    crops = data['recommended_crops']
    limit = int(days * 1.2)
    print("planning_days=" + str(days) + " (max_allowed=" + str(limit) + ")")
    print("Crops: " + str(len(crops)))
    over = 0
    for c in crops:
        min_dur = c.get('min_duration_days', c['growth_duration_days'])
        dur_range = c.get('duration_range', [c['growth_duration_days'], c['growth_duration_days']])
        status = "OK" if min_dur <= limit else "OVER"
        if status == "OVER":
            over += 1
        print("  " + status + " | " + c['crop'] + " | min=" + str(min_dur) + "d typ=" + str(c['growth_duration_days']) + "d range=" + str(dur_range))
    if over == 0:
        print("PASS - all crops within planning limit")
    else:
        print("FAIL - " + str(over) + " crops exceeded limit")
    print("")

test(30)
test(60)
test(90)
