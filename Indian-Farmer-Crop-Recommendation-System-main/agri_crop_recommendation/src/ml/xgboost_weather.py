"""
XGBoost Weather Forecasting Model
===================================
Trains a gradient-boosted tree ensemble to predict short-term weather
(temperature and rainfall) from historical district data.

Three separate XGBoost regressors are trained:
  - temp_max predictor
  - temp_min predictor
  - rainfall predictor

Each model predicts the NEXT DAY's value from lag/rolling engineered features.
For multi-day forecasting, predictions are made iteratively (day-by-day).

Usage:
    # Train (from scripts/train_model.py or directly):
    forecaster = XGBoostWeatherForecaster()
    forecaster.train("data/weather/district")
    forecaster.save()

    # Predict 7-day forecast for a district:
    forecaster = XGBoostWeatherForecaster.load()
    forecast = forecaster.predict(last_30_days_df, district_id="MH_PUNE")
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models/weather_xgboost")
TARGETS = ["temp_max", "temp_min", "rainfall"]


class XGBoostWeatherForecaster:
    """
    XGBoost-based weather forecaster trained on district-level daily data.

    Predicts 7-day ahead temperature and rainfall using lag & rolling features.
    A district-ID label encoding is used so one global model covers all districts.
    """

    def __init__(self):
        self.models: Dict[str, object] = {}   # one XGBRegressor per target
        self.scaler = None
        self.district_encoder: Dict[str, int] = {}
        self.feature_cols: List[str] = []
        self.metadata: Dict = {}
        self._loaded = False

    # ── Training ─────────────────────────────────────────────────────────────

    def train(
        self,
        data_dir: str = "data/weather/district",
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        forecast_horizon: int = 7,
        sample_districts: Optional[int] = None,
    ):
        """
        Train XGBoost regressors on all district parquet data.

        Args:
            data_dir:         Path to data/weather/district/
            n_estimators:     Number of boosting trees
            max_depth:        Max tree depth
            learning_rate:    XGBoost learning rate (eta)
            forecast_horizon: Days ahead to predict (default 7)
            sample_districts: If set, only use this many districts (fast testing)
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost not installed. Run: pip install xgboost")

        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(
                f"District data directory not found: {data_path}\n"
                "Run: python scripts/fetch_district_weather.py"
            )

        logger.info("Loading district weather data...")
        all_dfs = self._load_all_districts(data_path, sample_districts)

        if not all_dfs:
            raise ValueError("No district data found. Run fetch_district_weather.py first.")

        logger.info(f"Loaded data for {len(all_dfs)} districts. Engineering features...")

        combined = self._build_training_frame(all_dfs, forecast_horizon)
        target_cols = [f"target_{t}" for t in TARGETS]
        logger.info(f"Training frame: {len(combined):,} rows × {len(combined.columns)} cols")

        # Train one model per target variable
        X = combined.drop(columns=target_cols, errors="ignore")
        self.feature_cols = list(X.columns)

        for target in TARGETS:
            col = f"target_{target}"
            y = combined[col]
            valid_mask = y.notna() & X.notna().all(axis=1)
            X_clean = X[valid_mask]
            y_clean = y[valid_mask]

            logger.info(f"Training XGBoost for '{target}' — {len(X_clean):,} samples...")
            model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                eval_metric="rmse",
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
            model.fit(X_clean, y_clean, verbose=False)
            self.models[target] = model
            n_eval = min(1000, len(X_clean))
            test_pred = model.predict(X_clean[:n_eval])
            rmse = np.sqrt(np.mean((test_pred - y_clean[:n_eval].values) ** 2))
            logger.info(f"  {target} training RMSE: {rmse:.3f}")

        self.metadata = {
            "n_districts": len(all_dfs),
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "forecast_horizon": forecast_horizon,
            "feature_cols": self.feature_cols,
            "district_encoder": self.district_encoder,
        }
        self._loaded = True
        logger.info("[OK] XGBoost weather models trained successfully.")

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(
        self,
        recent_df: pd.DataFrame,
        district_id: str = None,
        horizon: int = 7
    ) -> Dict:
        """
        Predict 7-day weather forecast from recent weather observations.

        Args:
            recent_df:   DataFrame with last 30+ days of weather
                         (columns: date, temp_max, temp_min, rainfall, humidity, wind_speed)
            district_id: Region ID for district encoding (optional)
            horizon:     Number of days to forecast (default 7)

        Returns:
            Dict with keys:
              - predictions: list of {date, temp_max, temp_min, rainfall}
              - summary:     {avg_temp, total_rainfall, ...}
              - model_used:  "xgboost"
              - confidence:  "high" / "medium"
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call train() or load() first.")

        df = recent_df.copy().sort_values("date").reset_index(drop=True)

        predictions = []
        current_df = df.copy()

        for day_offset in range(1, horizon + 1):
            row = self._engineer_features(current_df, district_id)
            if row is None:
                break

            pred_row = {}
            for target in TARGETS:
                if target in self.models:
                    feat = row[self.feature_cols]
                    val = float(self.models[target].predict(feat)[0])
                    if target == "rainfall":
                        val = max(0.0, val)
                    pred_row[target] = round(val, 1)

            next_date = pd.Timestamp(current_df["date"].iloc[-1]) + pd.Timedelta(days=1)
            pred_row["date"] = str(next_date.date())
            predictions.append(pred_row)

            # Append predicted row so autoregressive forecast keeps going
            new_row = pd.DataFrame([{
                "date":       next_date,
                "temp_max":   pred_row.get("temp_max", 30.0),
                "temp_min":   pred_row.get("temp_min", 18.0),
                "rainfall":   pred_row.get("rainfall", 0.0),
                "humidity":   current_df["humidity"].mean() if "humidity" in current_df else 60.0,
                "wind_speed": current_df["wind_speed"].mean() if "wind_speed" in current_df else 10.0,
                "region_id":  district_id or "",
            }])
            current_df = pd.concat([current_df, new_row], ignore_index=True)

        avg_temp = np.mean([
            (p["temp_max"] + p["temp_min"]) / 2 for p in predictions
        ]) if predictions else 25.0
        total_rain = sum(p["rainfall"] for p in predictions)

        return {
            "predictions": predictions,
            "summary": {
                "avg_temp":       round(avg_temp, 1),
                "total_rainfall": round(total_rain, 1),
            },
            "model_used":  "xgboost",
            "confidence":  "high" if len(predictions) >= horizon else "medium",
        }

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, model_dir: str = None):
        """Save trained models to disk."""
        save_path = Path(model_dir) if model_dir else MODELS_DIR
        save_path.mkdir(parents=True, exist_ok=True)

        for target, model in self.models.items():
            joblib.dump(model, save_path / f"{target}_model.joblib")

        with open(save_path / "metadata.json", "w") as f:
            meta = self.metadata.copy()
            json.dump(meta, f, indent=2)

        logger.info(f"Saved XGBoost models to {save_path}")

    @classmethod
    def load(cls, model_dir: str = None) -> "XGBoostWeatherForecaster":
        """Load trained models from disk."""
        try:
            import xgboost  # noqa: F401
        except ImportError:
            raise ImportError("xgboost not installed. Run: pip install xgboost")

        load_path = Path(model_dir) if model_dir else MODELS_DIR
        if not load_path.exists():
            raise FileNotFoundError(f"XGBoost model directory not found: {load_path}")

        instance = cls()

        with open(load_path / "metadata.json") as f:
            instance.metadata = json.load(f)

        instance.feature_cols = instance.metadata.get("feature_cols", [])
        instance.district_encoder = instance.metadata.get("district_encoder", {})

        for target in TARGETS:
            model_path = load_path / f"{target}_model.joblib"
            if model_path.exists():
                instance.models[target] = joblib.load(model_path)

        instance._loaded = len(instance.models) > 0
        logger.info(f"Loaded XGBoost models from {load_path} ({len(instance.models)} targets)")
        return instance

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_all_districts(
        self, data_path: Path, sample: Optional[int]
    ) -> Dict[str, pd.DataFrame]:
        """Load all district parquet files into memory."""
        from src.ml.pipeline import WeatherDataPipeline
        pipeline = WeatherDataPipeline(str(data_path))

        district_dirs = sorted(d for d in data_path.iterdir() if d.is_dir())
        if sample:
            district_dirs = district_dirs[:sample]

        all_dfs = {}
        for district_dir in district_dirs:
            region_id = district_dir.name
            # Build encoder incrementally
            if region_id not in self.district_encoder:
                self.district_encoder[region_id] = len(self.district_encoder)
            try:
                df = pipeline.load_region_data(region_id)
                df["region_id"] = region_id
                all_dfs[region_id] = df
            except Exception as e:
                logger.warning(f"Skipping {region_id}: {e}")

        return all_dfs

    def _build_training_frame(
        self, all_dfs: Dict[str, pd.DataFrame], forecast_horizon: int
    ) -> pd.DataFrame:
        """
        Build combined feature+target DataFrame across all districts.

        Strategy: compute future targets on the SORTED raw df first (before
        feature engineering drops lag-NaN rows), then inner-join on date so
        the alignment is always correct.
        """
        from src.ml.pipeline import WeatherDataPipeline
        pipeline = WeatherDataPipeline()

        frames = []
        for region_id, df in all_dfs.items():
            raw = df.copy().sort_values("date").reset_index(drop=True)

            # ── Future targets (shifted on raw df, indexed by date) ──
            target_df = pd.DataFrame({"date": raw["date"]})
            for target in TARGETS:
                target_df[f"target_{target}"] = raw[target].shift(-forecast_horizon).values

            # ── Engineered features (rows with NaN lags dropped internally) ──
            feat_df = pipeline.create_xgboost_features(raw)
            feat_df["district_enc"] = self.district_encoder.get(region_id, -1)

            # ── Join on date — only keeps rows present in both ──
            merged = feat_df.merge(target_df, on="date", how="inner")
            frames.append(merged)

        combined = pd.concat(frames, ignore_index=True)
        target_cols = [f"target_{t}" for t in TARGETS]
        combined = combined.dropna(subset=target_cols).reset_index(drop=True)

        # Drop non-feature columns (but keep target_ columns for training)
        drop_cols = ["date", "region_id"] + TARGETS
        combined = combined.drop(columns=[c for c in drop_cols if c in combined.columns])
        return combined

    def _engineer_features(
        self, df: pd.DataFrame, district_id: Optional[str]
    ) -> Optional[pd.DataFrame]:
        """Engineer features from a recent-history DataFrame for prediction."""
        from src.ml.pipeline import WeatherDataPipeline
        pipeline = WeatherDataPipeline()

        try:
            feat_df = pipeline.create_xgboost_features(df)
            if feat_df.empty:
                return None

            # Take the last row (most recent features)
            row = feat_df.iloc[[-1]].copy()
            row["district_enc"] = self.district_encoder.get(district_id or "", -1)

            # Align to training feature columns; fill any missing with 0
            for col in self.feature_cols:
                if col not in row.columns:
                    row[col] = 0.0
            row = row[self.feature_cols]
            return row

        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            return None
