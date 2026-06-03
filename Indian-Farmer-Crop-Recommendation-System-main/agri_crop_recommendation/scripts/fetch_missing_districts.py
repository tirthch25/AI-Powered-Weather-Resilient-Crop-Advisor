"""
Fetch missing district weather data using exact IDs from regions.json.
Uses coordinates already stored in regions.json — no hardcoded lookup needed.

Run from agri_crop_recommendation/:
    python scripts/fetch_missing_districts.py
"""

import json
import sys
import time
import requests
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("fetch_missing.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)

HISTORICAL_API  = "https://archive-api.open-meteo.com/v1/archive"
REQUEST_TIMEOUT = 30
RETRY_LIMIT     = 3
RETRY_DELAY     = 5
START_YEAR      = 2014
END_YEAR        = 2024
OUTPUT_DIR      = Path("data/weather/district")


class RateLimitError(Exception):
    pass


def fetch_year(region_id: str, lat: float, lon: float, year: int) -> bool:
    out = OUTPUT_DIR / region_id / f"{year}.parquet"
    if out.exists():
        return True  # already downloaded

    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": f"{year}-01-01",
        "end_date":   f"{year}-12-31",
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "relative_humidity_2m_max",
            "wind_speed_10m_max",
        ],
        "timezone": "Asia/Kolkata",
    }

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            resp = requests.get(HISTORICAL_API, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                raise RateLimitError(
                    "Open-Meteo daily API limit reached. "
                    "Re-run tomorrow — already-downloaded files will be skipped."
                )
            resp.raise_for_status()
            data = resp.json()["daily"]

            df = pd.DataFrame({
                "date":       data["time"],
                "temp_max":   data["temperature_2m_max"],
                "temp_min":   data["temperature_2m_min"],
                "rainfall":   data["precipitation_sum"],
                "humidity":   data["relative_humidity_2m_max"],
                "wind_speed": data["wind_speed_10m_max"],
            })
            df["date"]       = pd.to_datetime(df["date"])
            df["rainfall"]   = df["rainfall"].fillna(0.0)
            df["temp_max"]   = df["temp_max"].interpolate().bfill().fillna(30.0)
            df["temp_min"]   = df["temp_min"].interpolate().bfill().fillna(18.0)
            df["humidity"]   = df["humidity"].interpolate().bfill().fillna(60.0)
            df["wind_speed"] = df["wind_speed"].interpolate().bfill().fillna(10.0)
            df["region_id"]  = region_id

            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out, index=False)
            return True

        except RateLimitError:
            raise
        except Exception as e:
            logger.warning(f"  Attempt {attempt}/{RETRY_LIMIT} failed for {region_id}/{year}: {e}")
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY)

    logger.error(f"  FAILED: {region_id}/{year} after {RETRY_LIMIT} attempts")
    return False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    regions = json.load(open("data/reference/regions.json"))
    years = list(range(START_YEAR, END_YEAR + 1))

    # Include districts with no folder AND those with incomplete year files
    all_targets = []
    for r in regions["regions"]:
        if not r.get("latitude") or not r.get("longitude"):
            continue
        rid = r["region_id"]
        dist_dir = OUTPUT_DIR / rid
        if not dist_dir.exists():
            all_targets.append(r)  # completely missing
        else:
            existing = {f.stem for f in dist_dir.glob("*.parquet")}
            needed = {str(y) for y in years}
            if not needed.issubset(existing):
                all_targets.append(r)  # partial — some years missing

    missing = all_targets

    total = len(missing) * len(years)
    logger.info(f"Districts to fetch : {len(missing)}")
    logger.info(f"Years per district : {len(years)}  ({START_YEAR}-{END_YEAR})")
    logger.info(f"Total API requests : {total}")
    logger.info(f"Est. time          : ~{total // 120} min (0.5s/req)")
    logger.info("-" * 60)

    done = 0
    failed = []
    start = datetime.now()

    for r in missing:
        rid = r["region_id"]
        lat = r["latitude"]
        lon = r["longitude"]

        for year in years:
            try:
                ok = fetch_year(rid, lat, lon, year)
            except RateLimitError as e:
                logger.error(f"\n[STOP] {e}")
                logger.info(f"Processed {done}/{total}  |  saved {done - len(failed)}")
                sys.exit(0)

            done += 1
            if not ok:
                failed.append(f"{rid}/{year}")

            time.sleep(0.5)  # ~2 req/sec, well within free tier

            if done % 100 == 0:
                elapsed  = (datetime.now() - start).total_seconds()
                rate     = done / elapsed if elapsed > 0 else 1
                eta_secs = (total - done) / rate
                logger.info(
                    f"Progress: {done}/{total} ({100*done//total}%)  "
                    f"| failed={len(failed)}  "
                    f"| ETA: {int(eta_secs//3600)}h {int((eta_secs%3600)//60)}m"
                )

    elapsed = (datetime.now() - start).total_seconds()
    logger.info(f"\n[DONE] {done - len(failed)}/{total} files saved in {int(elapsed//60)}m")
    if failed:
        logger.warning(f"[WARN] {len(failed)} failures: {failed[:10]}{'...' if len(failed) > 10 else ''}")


if __name__ == "__main__":
    main()
