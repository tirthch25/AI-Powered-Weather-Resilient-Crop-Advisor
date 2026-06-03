import sys, os, json, logging
sys.path.insert(0, os.getcwd())
logging.disable(logging.CRITICAL)

import pandas as pd
import numpy as np

# Metadata check
lstm_meta = json.load(open("models/weather_lstm/metadata.json"))
xgb_meta  = json.load(open("models/weather_xgboost/metadata.json"))
print(f"LSTM: {len(lstm_meta['district_encoder'])} districts, final_rmse={lstm_meta['final_rmse']:.4f}, epochs={lstm_meta['epochs']}")
print(f"XGB:  {xgb_meta['n_districts']} districts trained")

new_districts = ["RJ_JAIPUR", "UP_LUCK", "GJ_KATCH", "RJ_JODHPUR", "UP_VARANA"]
print("\nNew district coverage:")
for d in new_districts:
    in_lstm = d in lstm_meta["district_encoder"]
    in_xgb  = d in xgb_meta["district_encoder"]
    print(f"  {d}: lstm={in_lstm}  xgb={in_xgb}")

# LSTM inference on new districts
print("\nLSTM inference tests:")
from src.ml.lstm_weather import LSTMWeatherForecaster
lstm = LSTMWeatherForecaster.load()

dates = pd.date_range("2026-02-20", periods=30)
df = pd.DataFrame({
    "date":       dates,
    "temp_max":   [32.0] * 30,
    "temp_min":   [18.0] * 30,
    "rainfall":   [0.0] * 30,
    "humidity":   [55.0] * 30,
    "wind_speed": [12.0] * 30,
})

for district in ["RJ_JAIPUR", "UP_LUCK", "GJ_KATCH"]:
    r = lstm.predict(df, district_id=district, horizon=7)
    p = r["predictions"][0]
    print(f"  {district}: Day-1 temp_max={p['temp_max']}, temp_min={p['temp_min']}, rain={p['rainfall']}, confidence={r['confidence']}")

print("\nPASSED - LSTM inference working for new districts")
