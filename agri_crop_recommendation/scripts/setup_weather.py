"""
Historical Weather Data Generator for Indian Agro-Climatic Zones
Generates 10+ years of realistic monthly weather data based on Indian
climatological patterns (IMD reference data).

Usage:
    python scripts/setup_weather.py

Output:
    data/weather/zone/historical_weather.csv  -- 792 rows (6 zones x 11 years x 12 months)
"""

import os
import csv
import random
import math

# ── Base climatological patterns per zone (monthly averages) ──
# Format: { zone: { month: (avg_temp_C, rainfall_mm, humidity_%) } }
# Based on IMD (India Meteorological Department) long-term normals

ZONE_CLIMATE = {
    "North": {
        # Delhi, UP, Punjab, Haryana, Uttarakhand — continental monsoon
        1:  (12.5,  18,  62),   # January  — cold, dry
        2:  (15.5,  20,  55),   # February — warming up
        3:  (21.5,  15,  45),   # March    — spring
        4:  (36.5,  10,  28),   # April    — hot, dry
        5:  (40.0,  18,  25),   # May      — peak heat
        6:  (42.0,  75,  48),   # June     — scorching before monsoon
        7:  (31.0, 210,  75),   # July     — peak monsoon
        8:  (30.0, 235,  78),   # August   — heavy monsoon
        9:  (28.5, 120,  68),   # September— retreating monsoon
        10: (25.0,  20,  52),   # October  — post-monsoon
        11: (19.0,   5,  55),   # November — cooling
        12: (13.5,  10,  62),   # December — cold
    },
    "South": {
        # Tamil Nadu, Kerala, Karnataka, AP, Telangana — tropical
        1:  (25.0,  15,  65),
        2:  (26.5,  10,  60),
        3:  (28.5,  12,  58),
        4:  (33.5,  45,  58),
        5:  (37.0,  65,  55),
        6:  (35.5, 140,  68),   # SW monsoon arrives
        7:  (28.0, 180,  78),
        8:  (27.5, 165,  78),
        9:  (27.0, 195,  80),
        10: (27.0, 220,  82),   # NE monsoon (heavy for TN)
        11: (26.0, 175,  78),
        12: (25.0,  60,  70),
    },
    "East": {
        # Bihar, Jharkhand, Odisha, West Bengal — sub-humid
        1:  (17.0,  12,  65),
        2:  (20.5,  18,  58),
        3:  (26.0,  20,  48),
        4:  (37.0,  38,  45),
        5:  (40.5,  55,  48),
        6:  (38.0, 195,  70),
        7:  (29.5, 290,  82),   # Very heavy monsoon
        8:  (29.0, 310,  85),
        9:  (28.5, 215,  80),
        10: (26.5,  85,  72),
        11: (22.0,  15,  65),
        12: (17.5,   8,  65),
    },
    "West": {
        # Gujarat + Maharashtra (coastal to semi-arid) — IMD mean daily temperatures
        # MEAN = (daily_max + daily_min) / 2  — NOT daily maximums.
        # Calibrated from real IMD data:
        #   Thane May: highs 35-43°C, lows 24-27°C → mean ~34°C
        #   Pune  May: highs 38-42°C, lows 22-25°C → mean ~33°C
        #   Ahmedabad May: highs 41-45°C, lows 27-30°C → mean ~36°C
        #   Mumbai May: highs 32-36°C, lows 24-27°C → mean ~30°C
        # Weighted zone mean for May ≈ 34-35°C
        1:  (20.0,   3,  55),   # January  — cool, dry
        2:  (22.5,   3,  50),   # February — warming
        3:  (27.0,   2,  42),   # March    — hot start
        4:  (33.0,   5,  32),   # April    — pre-monsoon heat build-up
        5:  (35.0,   8,  38),   # May      — peak heat (mean ~35°C; highs 35-43°C)
        6:  (30.5, 110,  65),   # June     — monsoon onset, temps drop sharply
        7:  (27.0, 260,  82),   # July     — peak monsoon
        8:  (26.5, 220,  83),   # August   — heavy monsoon
        9:  (27.0, 145,  79),   # September— retreating monsoon
        10: (27.5,  30,  63),   # October  — post-monsoon
        11: (23.5,   8,  54),   # November — cooling
        12: (20.5,   3,  53),   # December — cool, dry
    },
    "Central": {
        # MP, Chhattisgarh — sub-tropical continental
        1:  (16.5,  12,  55),
        2:  (19.5,  10,  45),
        3:  (25.0,   8,  32),
        4:  (39.0,   5,  20),
        5:  (43.0,  10,  18),
        6:  (41.0, 130,  52),
        7:  (28.5, 310,  80),   # Very heavy monsoon
        8:  (27.5, 340,  84),
        9:  (27.0, 195,  78),
        10: (25.0,  35,  58),
        11: (20.5,  10,  50),
        12: (17.0,   8,  55),
    },
    "Northeast": {
        # Assam, Meghalaya, Arunachal, etc. — heavy rainfall region
        1:  (15.0,  12,  72),
        2:  (17.0,  25,  65),
        3:  (21.0,  65,  60),
        4:  (23.5, 155,  70),   # Pre-monsoon showers
        5:  (25.0, 265,  78),
        6:  (27.0, 340,  85),   # Peak rainfall (Cherrapunji!)
        7:  (28.0, 380,  88),
        8:  (28.0, 320,  87),
        9:  (27.0, 240,  84),
        10: (24.0, 120,  78),
        11: (19.5,  20,  72),
        12: (15.5,   5,  72),
    },
}

ZONES = list(ZONE_CLIMATE.keys())
YEARS = list(range(2014, 2025))  # 2014–2024 = 11 years


def add_natural_variation(base_temp, base_rain, base_humidity, year, month):
    """
    Add realistic inter-annual variation and climate trends.
    - Temperature: ±2°C variation + slight warming trend (+0.02°C/year from 2014)
    - Rainfall: ±30% variation + El Niño/La Niña effect
    - Humidity: ±8% variation correlated with rainfall
    """
    random.seed(year * 100 + month)  # reproducible per year-month

    # Slight warming trend (+0.02°C/year from 2014 baseline)
    warming = (year - 2014) * 0.02

    # El Niño approximation: deficit monsoon years
    el_nino_years = {2015, 2018, 2023}
    la_nina_years = {2016, 2020, 2021}

    rain_factor = 1.0
    if year in el_nino_years and month in (6, 7, 8, 9):
        rain_factor = random.uniform(0.7, 0.88)
    elif year in la_nina_years and month in (6, 7, 8, 9):
        rain_factor = random.uniform(1.10, 1.30)

    temp = base_temp + warming + random.gauss(0, 1.2)
    rain = max(0, base_rain * rain_factor + random.gauss(0, base_rain * 0.18))
    humidity = base_humidity + random.gauss(0, 4.0)
    humidity = max(15, min(98, humidity))

    return round(temp, 1), round(rain, 1), round(humidity, 1)


def generate_dataset(output_dir="data/weather/zone"):
    """Generate the full historical weather dataset and save to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "historical_weather.csv")

    rows = []
    for zone in ZONES:
        for year in YEARS:
            for month in range(1, 13):
                base_temp, base_rain, base_hum = ZONE_CLIMATE[zone][month]
                temp, rain, hum = add_natural_variation(
                    base_temp, base_rain, base_hum, year, month
                )
                rows.append({
                    "zone":        zone,
                    "year":        year,
                    "month":       month,
                    "temperature": temp,
                    "rainfall":    rain,
                    "humidity":    hum,
                })

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "zone", "year", "month", "temperature", "rainfall", "humidity"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Generated {len(rows)} records -> {filepath}")
    print(f"     Zones: {len(ZONES)}, Years: {len(YEARS)}, Months: 12")
    return filepath


if __name__ == "__main__":
    generate_dataset()
