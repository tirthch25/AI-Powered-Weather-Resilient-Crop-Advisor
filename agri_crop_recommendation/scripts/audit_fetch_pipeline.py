"""
Full audit of the district weather data fetching pipeline.
Run from: agri_crop_recommendation/
    python scripts/audit_fetch_pipeline.py
"""
import json
import sys
import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE          = Path(".")
REGIONS_JSON  = BASE / "data/reference/regions.json"
DISTRICT_DIR  = BASE / "data/weather/district"
FETCH_SCRIPT  = BASE / "scripts/fetch_missing_districts.py"
RETRAIN_SCRIPT= BASE / "scripts/retrain_models.py"
PIPELINE_SRC  = BASE / "src"

YEARS = {str(y) for y in range(2014, 2025)}  # 11 years

SEP = "=" * 65

def check(label, ok, detail=""):
    status = "OK " if ok else "ERR"
    msg = f"  [{status}] {label}"
    if detail:
        msg += f"  ->  {detail}"
    print(msg)
    return ok


# ═══════════════════════════════════════════════════════════════════
print(SEP)
print("FETCH PIPELINE AUDIT")
print(SEP)

# ── 1. File existence checks ──────────────────────────────────────
print("\n[1] KEY FILE & FOLDER EXISTENCE")
check("regions.json present",      REGIONS_JSON.exists(),  str(REGIONS_JSON))
check("data/weather/district/ present", DISTRICT_DIR.exists(), str(DISTRICT_DIR))
check("fetch_missing_districts.py", FETCH_SCRIPT.exists(), str(FETCH_SCRIPT))
check("retrain_models.py",          RETRAIN_SCRIPT.exists(), str(RETRAIN_SCRIPT))
check("fetch_district_weather.py",  (BASE/"scripts/fetch_district_weather.py").exists())
check(".env file present",          (BASE/".env").exists(), str(BASE/".env"))

# ── 2. regions.json integrity ─────────────────────────────────────
print("\n[2] regions.json INTEGRITY")
r = json.load(open(REGIONS_JSON))
regions = r["regions"]
json_ids = {reg["region_id"] for reg in regions}

no_coords = [reg["region_id"] for reg in regions
             if not reg.get("latitude") or not reg.get("longitude")]
check("Total regions",             len(json_ids) > 0,           f"{len(json_ids)} regions")
check("All have lat/lon coords",   len(no_coords) == 0,
      f"{len(no_coords)} missing coords" if no_coords else "all present")

# ── 3. fetch_missing_districts.py path check ─────────────────────
print("\n[3] fetch_missing_districts.py PATH CHECK")
src = FETCH_SCRIPT.read_text(encoding="utf-8")

output_dir_line = [l.strip() for l in src.splitlines() if "OUTPUT_DIR" in l and "=" in l and "#" not in l.split("OUTPUT_DIR")[0]]
regions_line    = [l.strip() for l in src.splitlines() if "regions.json" in l]
check("OUTPUT_DIR defined",        len(output_dir_line) > 0,  output_dir_line[0] if output_dir_line else "NOT FOUND")
check("regions.json path in script", len(regions_line) > 0,  regions_line[0] if regions_line else "NOT FOUND")

# Verify the OUTPUT_DIR resolves to the right place
import re
m = re.search(r'OUTPUT_DIR\s*=\s*Path\("([^"]+)"\)', src)
if m:
    declared_path = Path(m.group(1))
    resolved = (BASE / declared_path).resolve()
    actual   = DISTRICT_DIR.resolve()
    check("OUTPUT_DIR matches actual district dir", resolved == actual,
          f"declared={declared_path}  resolved={resolved}")
else:
    check("OUTPUT_DIR parseable", False, "could not parse")

# ── 4. Disk vs JSON completeness ──────────────────────────────────
print("\n[4] DISTRICT DATA COMPLETENESS")
disk_ids  = {p.name for p in DISTRICT_DIR.iterdir() if p.is_dir()} if DISTRICT_DIR.exists() else set()
matched   = disk_ids & json_ids
orphaned  = disk_ids - json_ids

complete, partial, no_folder = [], [], []
for rid in sorted(json_ids):
    d = DISTRICT_DIR / rid
    if not d.exists():
        no_folder.append(rid)
    else:
        existing    = {f.stem for f in d.glob("*.parquet")}
        missing_yrs = YEARS - existing
        if not missing_yrs:
            complete.append(rid)
        else:
            partial.append((rid, sorted(missing_yrs)))

total_parquets = sum(
    len(list((DISTRICT_DIR/rid).glob("*.parquet")))
    for rid in matched
)

print(f"  Total JSON regions            : {len(json_ids)}")
print(f"  Disk folders total            : {len(disk_ids)}")
print(f"  Matched (JSON key == folder)  : {len(matched)}")
print(f"  Orphaned old-key folders      : {len(orphaned)}")
print(f"  [OK]  Complete (11/11 years)  : {len(complete)}")
print(f"  [!!]  Partial (some yrs miss) : {len(partial)}")
print(f"  [XX]  No folder at all        : {len(no_folder)}")
print(f"  Total .parquet files          : {total_parquets}")
print(f"  Script will fetch             : {len(partial) + len(no_folder)} districts")

# ── 5. Parquet content spot-check ────────────────────────────────
print("\n[5] PARQUET CONTENT SPOT-CHECK (3 random files)")
import random
random.seed(42)
try:
    import pandas as pd
    samples = random.sample(complete, min(3, len(complete)))
    for rid in samples:
        f = list((DISTRICT_DIR/rid).glob("*.parquet"))[0]
        df = pd.read_parquet(f)
        expected_cols = {"date","temp_max","temp_min","rainfall","humidity","wind_speed","region_id"}
        has_cols = expected_cols.issubset(set(df.columns))
        has_rows = len(df) >= 365
        check(f"{rid}/{f.name}  cols+rows",
              has_cols and has_rows,
              f"{len(df)} rows, cols={list(df.columns)}")
except Exception as e:
    print(f"  [ERR] Could not check parquet: {e}")

# ── 6. Partial districts detail ───────────────────────────────────
print("\n[6] PARTIAL DISTRICTS (have folder, missing some years)")
if not partial:
    print("  None — all folders with data are complete!")
else:
    for rid, yrs in partial:
        print(f"  {rid:45s} missing: {yrs}")

# ── 7. Orphaned folders ───────────────────────────────────────────
print(f"\n[7] ORPHANED OLD-KEY FOLDERS ({len(orphaned)} total)")
print("  (These are harmless — data exists but key not in regions.json)")
for o in sorted(orphaned)[:25]:
    fc = len(list((DISTRICT_DIR/o).glob("*.parquet")))
    print(f"  {o:40s} {fc} files")
if len(orphaned) > 25:
    print(f"  ... and {len(orphaned)-25} more")

# ── 8. fetch_missing_districts.py logic check ────────────────────
print("\n[8] fetch_missing_districts.py LOGIC SIMULATION")
print("  (Simulating exactly what the script would do if run now)")
would_fetch = []
for rid in sorted(json_ids):
    d = DISTRICT_DIR / rid
    if not d.exists():
        would_fetch.append((rid, list(YEARS), "NO_FOLDER"))
    else:
        existing    = {f.stem for f in d.glob("*.parquet")}
        missing_yrs = sorted(YEARS - existing)
        if missing_yrs:
            would_fetch.append((rid, missing_yrs, "PARTIAL"))

total_api_calls = sum(len(yrs) for _, yrs, _ in would_fetch)
print(f"  Districts script would target  : {len(would_fetch)}")
print(f"  Total API calls needed         : {total_api_calls}")
print(f"  Est. time (0.5s/req, no limit) : ~{total_api_calls//120} min")
print(f"  Daily limit ~600 req => days   : ~{total_api_calls//600 + 1} days")

# ── 9. Summary ───────────────────────────────────────────────────
print(f"\n{SEP}")
print("SUMMARY")
print(SEP)
all_ok = len(partial) == 0 and len(no_folder) == 0
if all_ok:
    print("  ALL DISTRICTS COMPLETE! Ready to retrain models.")
else:
    print(f"  {len(complete)}/{len(json_ids)} districts fully complete ({100*len(complete)//len(json_ids)}%)")
    print(f"  Still need: {len(partial)} partial + {len(no_folder)} missing = {len(partial)+len(no_folder)} districts")
    print(f"  Action: run fetch_missing_districts.py (~{total_api_calls//600+1} more daily runs)")
print(SEP)
