"""
Retrain Weather Models on Current District Data
=================================================
Retrains the XGBoost and LSTM weather forecasting models using all
district parquet files currently available in data/weather/district/.

Designed to be run incrementally — safe to re-run any time more
district data has been downloaded.

Run from agri_crop_recommendation/:
    python scripts/retrain_models.py                  # both models
    python scripts/retrain_models.py --model xgboost  # XGBoost only
    python scripts/retrain_models.py --model lstm     # LSTM only
    python scripts/retrain_models.py --model xgboost --estimators 400
    python scripts/retrain_models.py --model lstm --epochs 25 --device cpu
"""

import sys
import os
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime

# Allow running from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("retrain.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

DISTRICT_DATA_DIR = Path("data/weather/district")
DIVIDER = "=" * 60


def count_districts() -> tuple[int, int]:
    """Return (total_district_folders, fully_complete_districts_11yr)."""
    if not DISTRICT_DATA_DIR.exists():
        return 0, 0
    total = 0
    complete = 0
    for d in DISTRICT_DATA_DIR.iterdir():
        if not d.is_dir():
            continue
        total += 1
        years = {f.stem for f in d.glob("*.parquet")}
        if len(years) >= 11:   # 2014-2024
            complete += 1
    return total, complete


def train_xgboost(args):
    """Train XGBoost weather models and save to models/weather_xgboost/."""
    from src.ml.xgboost_weather import XGBoostWeatherForecaster

    print(f"\n{DIVIDER}")
    print("  XGBoost — Weather Forecasting (temp_max, temp_min, rainfall)")
    print(DIVIDER)

    t0 = time.time()
    forecaster = XGBoostWeatherForecaster()
    forecaster.train(
        data_dir=str(DISTRICT_DATA_DIR),
        n_estimators=args.estimators,
        max_depth=args.max_depth,
        learning_rate=args.lr,
    )
    forecaster.save()
    elapsed = time.time() - t0

    n = forecaster.metadata.get("n_districts", "?")
    print(f"\n  [OK] XGBoost retrained on {n} districts in {elapsed/60:.1f} min")
    print(f"  [SAVED] -> models/weather_xgboost/\n")
    logger.info(f"XGBoost retrain complete: {n} districts, {elapsed/60:.1f} min")


def train_lstm(args):
    """Train LSTM weather model and save to models/weather_lstm/."""
    from src.ml.lstm_weather import LSTMWeatherForecaster

    print(f"\n{DIVIDER}")
    print("  LSTM — Weather Forecasting (PyTorch, 2-layer, hidden=128)")
    print(DIVIDER)

    t0 = time.time()
    forecaster = LSTMWeatherForecaster()
    forecaster.train(
        data_dir=str(DISTRICT_DATA_DIR),
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )
    forecaster.save()
    elapsed = time.time() - t0

    rmse = forecaster.metadata.get("final_rmse", "?")
    print(f"\n  [OK] LSTM retrained in {elapsed/60:.1f} min  |  Final RMSE: {rmse}")
    print(f"  [SAVED] -> models/weather_lstm/\n")
    logger.info(f"LSTM retrain complete: RMSE={rmse}, {elapsed/60:.1f} min")


def main():
    parser = argparse.ArgumentParser(
        description="Retrain weather forecast models on current district data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        choices=["xgboost", "lstm", "both"],
        default="both",
        help="Which model(s) to retrain (default: both)",
    )

    # XGBoost options
    parser.add_argument("--estimators", type=int, default=300,
                        help="[XGBoost] Number of boosting trees (default: 300)")
    parser.add_argument("--max-depth", type=int, default=6,
                        help="[XGBoost] Max tree depth (default: 6)")
    parser.add_argument("--lr", type=float, default=0.05,
                        help="[XGBoost] Learning rate (default: 0.05)")

    # LSTM options
    parser.add_argument("--epochs", type=int, default=20,
                        help="[LSTM] Training epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="[LSTM] Batch size (default: 512)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="[LSTM] Device: cpu or cuda (default: cpu)")

    args = parser.parse_args()

    # ── Pre-flight check ──────────────────────────────────────────────────────
    total, complete = count_districts()
    if total == 0:
        print("[ERROR] No district data found at data/weather/district/")
        print("    Run scripts/fetch_missing_districts.py first.")
        sys.exit(1)

    print(f"\n{DIVIDER}")
    print(f"  Indian Farmer Crop Recommendation - Model Retrainer")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(DIVIDER)
    print(f"  Districts available : {total}")
    print(f"  Fully complete (11yr): {complete}")
    print(f"  Model target        : {args.model.upper()}")
    if args.model in ("xgboost", "both"):
        print(f"  XGBoost estimators  : {args.estimators}, depth={args.max_depth}, lr={args.lr}")
    if args.model in ("lstm", "both"):
        print(f"  LSTM epochs         : {args.epochs}, batch={args.batch_size}, device={args.device}")
    print(DIVIDER)

    logger.info(f"Starting retrain — {total} districts ({complete} complete), model={args.model}")

    # ── Train ─────────────────────────────────────────────────────────────────
    overall_start = time.time()

    if args.model in ("xgboost", "both"):
        train_xgboost(args)

    if args.model in ("lstm", "both"):
        train_lstm(args)

    total_time = time.time() - overall_start
    print(f"\n{DIVIDER}")
    print(f"  ALL DONE [OK]  Total time: {total_time/60:.1f} min")
    print(f"  Models saved to models/weather_xgboost/ and models/weather_lstm/")
    print(DIVIDER)
    logger.info(f"All retraining done in {total_time/60:.1f} min")


if __name__ == "__main__":
    main()
