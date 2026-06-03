"""
Unified Model Training Script
================================
Trains one or more models for the crop recommendation system.

Models:
    rf               — Random Forest crop suitability model (default)
    xgboost_weather  — XGBoost 7-day weather forecast model
    lstm_weather     — LSTM 7-day weather forecast model (PyTorch)
    all              — Train all three in sequence

Usage:
    cd agri_crop_recommendation

    # Crop suitability model (no district data needed)
    python scripts/train_model.py --model rf

    # Weather models (requires district data — run fetch_district_weather.py first)
    python scripts/train_model.py --model xgboost_weather --sample 10
    python scripts/train_model.py --model lstm_weather --sample 10 --epochs 5
    python scripts/train_model.py --model all

    # Full training (all districts, production quality)
    python scripts/train_model.py --model all --epochs 20
"""

import sys
import os
import io
import argparse
import logging

# Force UTF-8 stdout/stderr on Windows to avoid UnicodeEncodeError
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

DIVIDER = "=" * 60


# ── Individual trainers ───────────────────────────────────────────────────────

def train_rf(args):
    """Train the Random Forest crop suitability model."""
    from src.ml.pipeline import CropTrainingDataGenerator
    from src.ml.predictor import CropSuitabilityRF, plot_feature_importance

    training_data_path = Path("data/ml/training/crop_suitability/crop_suitability_data.csv")

    print(f"\n{DIVIDER}\n  RANDOM FOREST — Crop Suitability\n{DIVIDER}")

    if not training_data_path.exists() or args.regenerate:
        print(f"  Generating training data ({args.scenarios} scenarios/combo) — streaming to disk...")
        training_data_path.parent.mkdir(parents=True, exist_ok=True)
        generator = CropTrainingDataGenerator()
        data = generator.generate_training_data(
            num_weather_scenarios=args.scenarios,
            random_seed=42,
            output_path=str(training_data_path),  # stream records → CSV to avoid OOM
        )
        print(f"  [OK] Saved {len(data):,} records -> {training_data_path}")
    else:
        print(f"  Loading existing training data from {training_data_path}")
        if args.sample_rows:
            # Fast path: read only a sample (useful for testing / low-RAM machines)
            data = pd.read_csv(training_data_path, nrows=args.sample_rows)
            print(f"  [OK] Loaded {len(data):,} records (sampled, limit={args.sample_rows:,})")
        else:
            # Full load — may require 8-12 GB RAM for the 4 GB CSV
            print("  Reading full CSV in chunks to reduce peak RAM...")
            chunks = []
            total = 0
            for chunk in pd.read_csv(training_data_path, chunksize=500_000):
                chunks.append(chunk)
                total += len(chunk)
                print(f"    ... {total:,} rows loaded", end='\r', flush=True)
            data = pd.concat(chunks, ignore_index=True)
            print(f"\n  [OK] Loaded {len(data):,} records")

    model = CropSuitabilityRF()
    results = model.train(
        data,
        target_col="suitability_score",
        n_estimators=args.estimators,
        max_depth=args.max_depth,
    )

    top_features = model.get_feature_importance(top_n=15)
    print(f"\n  Top features:")
    for feat in top_features:
        bar = "█" * int(feat["importance"] * 100)
        print(f"    {feat['feature']:<25} {feat['importance']:.4f} {bar}")

    if model.feature_names and model.feature_importances is not None:
        plot_feature_importance(
            model.feature_names,
            model.feature_importances,
            title="Crop Suitability — Feature Importance",
            save_path="models/crop_suitability/feature_importance.png",
        )

    model.save()
    print(f"  [OK] Model saved -> models/crop_suitability/")
    print(f"  Train R2: {results['train_metrics']['r2']:.4f} | "
          f"Test R2: {results['test_metrics']['r2']:.4f} | "
          f"RMSE: {results['test_metrics']['rmse']:.4f}\n")


def train_xgboost_weather(args):
    """Train the XGBoost weather forecasting model."""
    from src.ml.xgboost_weather import XGBoostWeatherForecaster

    print(f"\n{DIVIDER}\n  XGBOOST — Weather Forecasting\n{DIVIDER}")
    data_dir = args.data_dir or "data/weather/district"

    if not Path(data_dir).exists():
        print(f"  [ERROR] District data not found at '{data_dir}'")
        print(f"     Run first: python scripts/fetch_district_weather.py")
        return

    forecaster = XGBoostWeatherForecaster()
    forecaster.train(
        data_dir=data_dir,
        n_estimators=args.estimators,
        sample_districts=args.sample,
    )
    forecaster.save()
    print(f"  [OK] XGBoost weather model saved -> models/weather_xgboost/")


def train_lstm_weather(args):
    """Train the LSTM weather forecasting model."""
    from src.ml.lstm_weather import LSTMWeatherForecaster

    print(f"\n{DIVIDER}\n  LSTM — Weather Forecasting (PyTorch)\n{DIVIDER}")
    data_dir = args.data_dir or "data/weather/district"

    if not Path(data_dir).exists():
        print(f"  [ERROR] District data not found at '{data_dir}'")
        print(f"     Run first: python scripts/fetch_district_weather.py")
        return

    forecaster = LSTMWeatherForecaster()
    forecaster.train(
        data_dir=data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sample_districts=args.sample,
        device=args.device,
    )
    forecaster.save()
    print(f"  [OK] LSTM weather model saved -> models/weather_lstm/")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train crop recommendation models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Which model(s) to train
    parser.add_argument(
        "--model",
        choices=["rf", "xgboost_weather", "lstm_weather", "all"],
        default="rf",
        help="Model to train (default: rf)",
    )

    # RF options
    parser.add_argument("--scenarios", type=int, default=50,
                        help="[RF] Weather scenarios per combination (default: 50)")
    parser.add_argument("--estimators", type=int, default=200,
                        help="[RF/XGB] Number of trees/estimators (default: 200)")
    parser.add_argument("--max-depth", type=int, default=15,
                        help="[RF] Max tree depth (default: 15)")
    parser.add_argument("--regenerate", action="store_true",
                        help="[RF] Force regenerate training data")
    parser.add_argument("--sample-rows", type=int, default=None,
                        help="[RF] Only load this many rows from existing CSV (e.g. 2000000 for a quick test)")

    # LSTM options
    parser.add_argument("--epochs", type=int, default=20,
                        help="[LSTM] Training epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="[LSTM] Batch size (default: 512)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="[LSTM] Device: cpu or cuda (default: cpu)")

    # Shared weather model options
    parser.add_argument("--data-dir", type=str, default=None,
                        help="[Weather] Path to district data (default: data/weather/district)")
    parser.add_argument("--sample", type=int, default=None,
                        help="[Weather] Only use this many districts (for fast testing)")

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Crop Recommendation System — Model Trainer")
    print(f"  Model target: {args.model.upper()}")
    print(f"{'='*60}\n")

    if args.model == "rf":
        train_rf(args)
    elif args.model == "xgboost_weather":
        train_xgboost_weather(args)
    elif args.model == "lstm_weather":
        train_lstm_weather(args)
    elif args.model == "all":
        train_rf(args)
        train_xgboost_weather(args)
        train_lstm_weather(args)

    print(f"\n{'='*60}")
    print(f"  ALL DONE [OK]")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
