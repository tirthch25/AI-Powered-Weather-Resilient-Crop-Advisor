"""
LSTM Weather Forecasting Model (PyTorch)
==========================================
A 2-layer LSTM network that predicts 7-day weather forecasts
from a 30-day lookback window of daily weather observations.

Architecture:
    Input:  (batch, 30, num_features)    — 30 days of weather features
    LSTM:   2 layers, hidden_size=128
    Output: (batch, 7, 3)                — 7-day forecast of temp_max, temp_min, rainfall

Usage:
    # Train:
    forecaster = LSTMWeatherForecaster()
    forecaster.train("data/weather/district", epochs=20)
    forecaster.save()

    # Predict:
    forecaster = LSTMWeatherForecaster.load()
    forecast = forecaster.predict(last_30_days_df, district_id="MH_PUNE")
"""

import json
import math
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib

logger = logging.getLogger(__name__)

MODELS_DIR  = Path("models/weather_lstm")
LOOKBACK    = 30      # days of history used as input
HORIZON     = 7       # days to predict
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
TARGETS     = ["temp_max", "temp_min", "rainfall"]
FEATURES    = ["temp_max", "temp_min", "rainfall", "humidity", "wind_speed",
               "month_sin", "month_cos", "day_sin", "day_cos"]


# ── PyTorch model definition ─────────────────────────────────────────────────

def _build_model(num_features: int, hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS, horizon: int = HORIZON,
                 num_targets: int = 3):
    """Construct the LSTM model (lazy import to avoid hard torch dependency at import time)."""
    import torch.nn as nn

    class LSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=num_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.2,
            )
            self.fc = nn.Linear(hidden_size, horizon * num_targets)
            self.horizon = horizon
            self.num_targets = num_targets

        def forward(self, x):
            # x: (batch, seq_len, features)
            out, _ = self.lstm(x)
            # Use last time step
            last = out[:, -1, :]           # (batch, hidden)
            pred = self.fc(last)           # (batch, horizon * targets)
            return pred.view(-1, self.horizon, self.num_targets)

    return LSTMModel()


# ── Forecaster wrapper ────────────────────────────────────────────────────────

class LSTMWeatherForecaster:
    """
    PyTorch LSTM weather forecaster.

    Trains on multi-district daily weather data.
    Uses normalization stats computed at training time (no data leakage).
    """

    def __init__(self):
        self.model = None
        self.norm_params: Dict = {}
        self.district_encoder: Dict[str, int] = {}
        self.metadata: Dict = {}
        self._num_features = len(FEATURES) + 1  # +1 for district_enc
        self._loaded = False

    # ── Training ─────────────────────────────────────────────────────────────

    def train(
        self,
        data_dir: str = "data/weather/district",
        epochs: int = 20,
        batch_size: int = 512,
        lr: float = 1e-3,
        lookback: int = LOOKBACK,
        horizon: int = HORIZON,
        sample_districts: Optional[int] = None,
        device: str = "cpu",
    ):
        """
        Train the LSTM model on all district daily weather data.

        Args:
            data_dir:          Path to district parquet files
            epochs:            Training epochs
            batch_size:        Mini-batch size
            lr:                Learning rate (Adam)
            lookback:          Sequence lookback window (days)
            horizon:           Forecast horizon (days)
            sample_districts:  If set, train on only this many districts (testing)
            device:            "cpu" or "cuda"
        """
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError("PyTorch not installed. Run: pip install torch")

        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(
                f"District data directory not found: {data_path}\n"
                "Run: python scripts/fetch_district_weather.py first."
            )

        logger.info("Loading and preparing data for LSTM training...")
        X_all, y_all = self._build_sequences(data_path, lookback, horizon, sample_districts)

        if X_all is None or len(X_all) == 0:
            raise ValueError("No training sequences built. Check data directory.")

        self._num_features = X_all.shape[2]
        logger.info(
            f"Training sequences: {len(X_all):,} x shape {X_all.shape} "
            f"-> targets {y_all.shape}"
        )

        dev = torch.device(device)
        self.model = _build_model(
            num_features=self._num_features,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            horizon=horizon,
            num_targets=len(TARGETS),
        ).to(dev)

        X_t = torch.tensor(X_all, dtype=torch.float32)
        y_t = torch.tensor(y_all, dtype=torch.float32)

        dataset = TensorDataset(X_t, y_t)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
        criterion = nn.MSELoss()

        best_loss = math.inf
        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(dev), yb.to(dev)
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            scheduler.step()
            avg_loss = epoch_loss / len(X_all)
            if avg_loss < best_loss:
                best_loss = avg_loss
            rmse = math.sqrt(avg_loss)
            logger.info(f"Epoch {epoch:3d}/{epochs}  RMSE={rmse:.4f}  (best={math.sqrt(best_loss):.4f})")

        self.metadata = {
            "lookback":          lookback,
            "horizon":           horizon,
            "num_features":      self._num_features,
            "features":          FEATURES + ["district_enc"],
            "targets":           TARGETS,
            "norm_params":       self.norm_params,
            "district_encoder":  self.district_encoder,
            "epochs":            epochs,
            "final_rmse":        math.sqrt(best_loss),
        }
        self._loaded = True
        logger.info(f"[OK] LSTM training complete. Best RMSE: {math.sqrt(best_loss):.4f}")

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(
        self,
        recent_df: pd.DataFrame,
        district_id: str = None,
        horizon: int = HORIZON,
    ) -> Dict:
        """
        Generate a 7-day weather forecast from recent history.

        Args:
            recent_df:   DataFrame with at least 30 days of weather
                         Columns: date, temp_max, temp_min, rainfall, humidity, wind_speed
            district_id: Region ID for district encoding
            horizon:     Forecast days (default 7)

        Returns:
            Dict with predictions, summary, model_used, confidence
        """
        if not self._loaded or self.model is None:
            raise RuntimeError("Model not loaded. Call train() or load() first.")

        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch not installed.")

        self.model.eval()
        df = recent_df.copy().sort_values("date").tail(LOOKBACK).reset_index(drop=True)

        if len(df) < LOOKBACK:
            logger.warning(f"Only {len(df)} days of history; need {LOOKBACK}. Padding with means.")
            pad = LOOKBACK - len(df)
            pad_df = pd.concat([df.iloc[:1]] * pad, ignore_index=True)
            df = pd.concat([pad_df, df], ignore_index=True)

        seq = self._df_to_sequence(df, district_id)   # (1, LOOKBACK, F)
        x_t = torch.tensor(seq, dtype=torch.float32)

        with torch.no_grad():
            pred = self.model(x_t).squeeze(0).numpy()  # (HORIZON, 3)

        # Denormalize
        predictions = []
        base_date = pd.Timestamp(df["date"].iloc[-1])
        for i in range(min(horizon, len(pred))):
            row = {}
            for j, target in enumerate(TARGETS):
                val = float(pred[i, j])
                if target in self.norm_params:
                    val = val * self.norm_params[target]["std"] + self.norm_params[target]["mean"]
                if target == "rainfall":
                    val = max(0.0, val)
                row[target] = round(val, 1)
            row["date"] = str((base_date + pd.Timedelta(days=i + 1)).date())
            predictions.append(row)

        avg_temp = np.mean([(p["temp_max"] + p["temp_min"]) / 2 for p in predictions])
        total_rain = sum(p["rainfall"] for p in predictions)

        return {
            "predictions": predictions,
            "summary": {
                "avg_temp":       round(float(avg_temp), 1),
                "total_rainfall": round(total_rain, 1),
            },
            "model_used":  "lstm",
            "confidence":  "high" if len(df) >= LOOKBACK else "medium",
        }

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, model_dir: str = None):
        """Save model weights and metadata to disk."""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch not installed.")

        save_path = Path(model_dir) if model_dir else MODELS_DIR
        save_path.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), save_path / "lstm_weights.pt")
        with open(save_path / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)

        logger.info(f"Saved LSTM model to {save_path}")

    @classmethod
    def load(cls, model_dir: str = None) -> "LSTMWeatherForecaster":
        """Load model from disk."""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch not installed.")

        load_path = Path(model_dir) if model_dir else MODELS_DIR
        if not load_path.exists():
            raise FileNotFoundError(f"LSTM model directory not found: {load_path}")

        instance = cls()

        with open(load_path / "metadata.json") as f:
            instance.metadata = json.load(f)

        instance.norm_params       = instance.metadata.get("norm_params", {})
        instance.district_encoder  = instance.metadata.get("district_encoder", {})
        instance._num_features     = instance.metadata.get("num_features", len(FEATURES) + 1)
        lookback    = instance.metadata.get("lookback", LOOKBACK)
        horizon     = instance.metadata.get("horizon", HORIZON)

        instance.model = _build_model(
            num_features=instance._num_features,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            horizon=horizon,
            num_targets=len(TARGETS),
        )
        instance.model.load_state_dict(
            torch.load(load_path / "lstm_weights.pt", map_location="cpu")
        )
        instance.model.eval()
        instance._loaded = True
        logger.info(f"Loaded LSTM model from {load_path}")
        return instance

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_sequences(
        self,
        data_path: Path,
        lookback: int,
        horizon: int,
        sample: Optional[int],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load all districts and build (X, y) sequence arrays."""
        from src.ml.pipeline import WeatherDataPipeline
        pipeline = WeatherDataPipeline(str(data_path))

        district_dirs = sorted(d for d in data_path.iterdir() if d.is_dir())
        if sample:
            district_dirs = district_dirs[:sample]

        # First pass: compute global norm stats across all data
        all_dfs = {}
        for d in district_dirs:
            region_id = d.name
            if region_id not in self.district_encoder:
                self.district_encoder[region_id] = len(self.district_encoder)
            try:
                df = pipeline.load_region_data(region_id)
                df["region_id"] = region_id
                all_dfs[region_id] = df
            except Exception as e:
                logger.warning(f"Skipping {region_id}: {e}")

        if not all_dfs:
            return None, None

        combined_for_norm = pd.concat(list(all_dfs.values()), ignore_index=True)
        feat_cols = ["temp_max", "temp_min", "rainfall", "humidity", "wind_speed"]
        for col in feat_cols:
            if col in combined_for_norm.columns:
                mean = float(combined_for_norm[col].mean())
                std  = float(combined_for_norm[col].std() or 1.0)
                self.norm_params[col] = {"mean": mean, "std": std}

        # Second pass: build sequences
        X_list, y_list = [], []
        for region_id, df in all_dfs.items():
            seqs = self._df_to_sequences_training(df, region_id, lookback, horizon)
            if seqs is not None:
                X_list.append(seqs[0])
                y_list.append(seqs[1])

        if not X_list:
            return None, None

        return np.concatenate(X_list), np.concatenate(y_list)

    def _add_temporal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add sin/cos encoded temporal features."""
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)
        df["day_sin"]   = np.sin(2 * np.pi * df["date"].dt.dayofyear / 365)
        df["day_cos"]   = np.cos(2 * np.pi * df["date"].dt.dayofyear / 365)
        return df

    def _normalize_col(self, series: pd.Series, col: str) -> np.ndarray:
        if col in self.norm_params:
            return ((series - self.norm_params[col]["mean"]) / self.norm_params[col]["std"]).values
        return series.values

    def _df_to_sequences_training(
        self, df: pd.DataFrame, region_id: str, lookback: int, horizon: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Convert a district DataFrame into (X, y) LSTM sequences."""
        df = self._add_temporal(df.copy())
        feat_cols = ["temp_max", "temp_min", "rainfall", "humidity", "wind_speed",
                     "month_sin", "month_cos", "day_sin", "day_cos"]

        # Fill missing feature columns with 0
        for col in feat_cols:
            if col not in df.columns:
                df[col] = 0.0

        data = np.column_stack([
            self._normalize_col(df[c], c) for c in feat_cols
        ] + [np.full(len(df), self.district_encoder.get(region_id, 0), dtype=float)])

        target_data = np.column_stack([
            self._normalize_col(df[t], t) for t in TARGETS
        ])

        if len(data) < lookback + horizon:
            return None

        X, y = [], []
        for i in range(lookback, len(data) - horizon + 1):
            X.append(data[i - lookback:i])
            y.append(target_data[i:i + horizon])

        if not X:
            return None

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def _df_to_sequence(self, df: pd.DataFrame, district_id: Optional[str]) -> np.ndarray:
        """Convert recent history DataFrame to a single input sequence for inference."""
        df = self._add_temporal(df.copy())
        feat_cols = ["temp_max", "temp_min", "rainfall", "humidity", "wind_speed",
                     "month_sin", "month_cos", "day_sin", "day_cos"]

        for col in feat_cols:
            if col not in df.columns:
                df[col] = 0.0

        data = np.column_stack([
            self._normalize_col(df[c], c) for c in feat_cols
        ] + [np.full(len(df), self.district_encoder.get(district_id or "", 0), dtype=float)])

        # Return shape: (1, LOOKBACK, num_features)
        seq = data[-LOOKBACK:]
        return seq[np.newaxis, :]
