"""
ML Data Pipeline

Prepares historical weather data for ML model training:
- Sliding window sequences for LSTM
- Tabular features with lag/rolling stats for XGBoost
- Train/validation/test splits
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agricultural Feature Engineering
# (previously in src/preprocessing/features.py — merged here to reduce files)
# ---------------------------------------------------------------------------

def add_agri_features(df: pd.DataFrame, base_temp: float = 10.0) -> pd.DataFrame:
    """
    Add agriculture-specific derived features to a weather DataFrame.

    Args:
        df: Raw weather DataFrame with columns: temp_max, temp_min, rainfall
        base_temp: Base temperature (°C) for GDD calculation (default 10°C)

    Returns:
        DataFrame with added columns:
            temp_avg       — daily average temperature
            gdd            — Growing Degree Days above base_temp
            rainfall_7d    — 7-day rolling rainfall sum
            dry_day        — True if rainfall < 2mm
            dry_spell_days — Consecutive dry days count
    """
    df = df.copy()
    df["temp_avg"] = (df["temp_max"] + df["temp_min"]) / 2
    df["gdd"] = np.maximum(df["temp_avg"] - base_temp, 0)
    df["rainfall_7d"] = df["rainfall"].rolling(window=7, min_periods=1).sum()
    df["dry_day"] = df["rainfall"] < 2
    df["dry_spell_days"] = (
        df["dry_day"]
        .astype(int)
        .groupby((df["dry_day"] != df["dry_day"].shift()).cumsum())
        .cumsum()
    )
    return df


class WeatherDataPipeline:
    """
    Prepares historical weather data for ML model training.
    
    Converts raw Parquet historical data into formats suitable for
    LSTM (sequences) and XGBoost (tabular features).
    """
    
    def __init__(self, data_dir: str = "data/weather/district"):
        self.data_dir = Path(data_dir)
        logger.info(f"Initialized WeatherDataPipeline with data_dir={self.data_dir}")
    
    def load_region_data(self, region_id: str) -> pd.DataFrame:
        """
        Load all historical data for a region and combine into one DataFrame.
        
        Args:
            region_id: Region identifier (e.g., "PUNE")
            
        Returns:
            DataFrame with columns: date, temp_max, temp_min, rainfall, humidity, wind_speed
        """
        region_dir = self.data_dir / region_id
        if not region_dir.exists():
            raise FileNotFoundError(f"No data directory found for region {region_id}")
        
        dfs = []
        for parquet_file in sorted(region_dir.glob("*.parquet")):
            df = pd.read_parquet(parquet_file)
            dfs.append(df)
        
        if not dfs:
            raise FileNotFoundError(f"No parquet files found for region {region_id}")
        
        combined = pd.concat(dfs, ignore_index=True)
        combined['date'] = pd.to_datetime(combined['date'])
        combined = combined.sort_values('date').reset_index(drop=True)
        
        logger.info(f"Loaded {len(combined)} records for region {region_id}")
        return combined
    
    def load_all_regions(self) -> Dict[str, pd.DataFrame]:
        """Load data for all available regions."""
        regions = {}
        for region_dir in sorted(self.data_dir.iterdir()):
            if region_dir.is_dir():
                try:
                    regions[region_dir.name] = self.load_region_data(region_dir.name)
                except FileNotFoundError:
                    logger.warning(f"Skipping region {region_dir.name}: no data")
        
        logger.info(f"Loaded data for {len(regions)} regions")
        return regions
    
    # ---- Feature Engineering for XGBoost ----
    
    def create_xgboost_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create tabular features for XGBoost from weather time series.
        
        Features created:
        - Lag features (1, 3, 7, 14, 30 days) for temp_max, temp_min, rainfall
        - Rolling mean/std (7, 14, 30 days) for each weather variable
        - Month, day-of-year, season encoding
        - Cumulative rainfall (7, 30 day windows)
        
        Args:
            df: Raw weather DataFrame with date, temp_max, temp_min, rainfall, etc.
            
        Returns:
            DataFrame with engineered features (rows with NaN from lagging are dropped)
        """
        df = df.copy()
        df = df.sort_values('date').reset_index(drop=True)
        
        weather_cols = ['temp_max', 'temp_min', 'rainfall']
        
        # Add humidity and wind_speed if available
        if 'humidity' in df.columns:
            weather_cols.append('humidity')
        if 'wind_speed' in df.columns:
            weather_cols.append('wind_speed')
        
        # --- Lag features ---
        lag_days = [1, 3, 7, 14, 30]
        for col in weather_cols:
            for lag in lag_days:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # --- Rolling statistics ---
        windows = [7, 14, 30]
        for col in weather_cols:
            for window in windows:
                df[f'{col}_roll_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                df[f'{col}_roll_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
        
        # --- Cumulative rainfall ---
        for window in [7, 30]:
            df[f'rainfall_cumsum_{window}'] = df['rainfall'].rolling(window=window, min_periods=1).sum()
        
        # --- Temporal features ---
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
        
        # Season encoding (Kharif=0, Rabi=1, Zaid=2)
        df['season'] = df['month'].apply(self._encode_season)
        
        # --- Temperature-derived features ---
        df['temp_range'] = df['temp_max'] - df['temp_min']
        df['temp_avg'] = (df['temp_max'] + df['temp_min']) / 2
        
        # Growing Degree Days
        df['gdd'] = np.maximum(df['temp_avg'] - 10, 0)
        
        # Drop rows with NaN from lag features
        df = df.dropna().reset_index(drop=True)
        
        logger.info(f"Created XGBoost features: {len(df)} samples, {len(df.columns)} features")
        return df
    
    # ---- Sequence Generation for LSTM ----
    
    def create_lstm_sequences(
        self,
        df: pd.DataFrame,
        lookback: int = 30,
        forecast_horizon: int = 7,
        target_cols: List[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding window sequences for LSTM training.
        
        Args:
            df: Weather DataFrame (should be sorted by date)
            lookback: Number of past days to use as input (default 30)
            forecast_horizon: Number of future days to predict (default 7)
            target_cols: Columns to predict (default: temp_max, temp_min, rainfall)
            
        Returns:
            X: shape (num_samples, lookback, num_features)
            y: shape (num_samples, forecast_horizon, num_targets)
        """
        if target_cols is None:
            target_cols = ['temp_max', 'temp_min', 'rainfall']
        
        # Select feature columns
        feature_cols = ['temp_max', 'temp_min', 'rainfall']
        if 'humidity' in df.columns:
            feature_cols.append('humidity')
        if 'wind_speed' in df.columns:
            feature_cols.append('wind_speed')
        
        # Add temporal features
        df = df.copy()
        df['month_sin'] = np.sin(2 * np.pi * df['date'].dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['date'].dt.month / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['date'].dt.dayofyear / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['date'].dt.dayofyear / 365)
        feature_cols.extend(['month_sin', 'month_cos', 'day_sin', 'day_cos'])
        
        data = df[feature_cols].values
        targets = df[target_cols].values
        
        X, y = [], []
        for i in range(lookback, len(data) - forecast_horizon + 1):
            X.append(data[i - lookback:i])
            y.append(targets[i:i + forecast_horizon])
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        logger.info(
            f"Created LSTM sequences: X={X.shape}, y={y.shape} "
            f"(lookback={lookback}, horizon={forecast_horizon})"
        )
        return X, y
    
    # ---- Train/Val/Test Splits ----
    
    def split_by_year(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically (no data leakage).
        
        Args:
            df: DataFrame with 'date' column
            train_ratio: Fraction for training (default 0.7)
            val_ratio: Fraction for validation (default 0.15)
            
        Returns:
            train_df, val_df, test_df
        """
        df = df.sort_values('date').reset_index(drop=True)
        n = len(df)
        
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        logger.info(
            f"Split data: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )
        return train_df, val_df, test_df
    
    # ---- Normalization ----
    
    def normalize_data(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """
        Normalize features using train set statistics (prevents data leakage).
        
        Returns:
            Normalized train, val, test DataFrames + normalization params dict
        """
        norm_params = {}
        
        for col in feature_cols:
            if col in train_df.columns:
                mean = train_df[col].mean()
                std = train_df[col].std()
                if std == 0:
                    std = 1.0  # Prevent division by zero
                
                norm_params[col] = {'mean': float(mean), 'std': float(std)}
                
                train_df[col] = (train_df[col] - mean) / std
                val_df[col] = (val_df[col] - mean) / std
                test_df[col] = (test_df[col] - mean) / std
        
        logger.info(f"Normalized {len(feature_cols)} features using train set statistics")
        return train_df, val_df, test_df, norm_params
    
    # ---- Helper Methods ----
    
    @staticmethod
    def _encode_season(month: int) -> int:
        """Encode month to Indian agricultural season."""
        if month in [6, 7, 8, 9]:
            return 0  # Kharif
        elif month in [10, 11, 12, 1, 2]:
            return 1  # Rabi
        else:
            return 2  # Zaid
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns (everything except date and target columns)."""
        exclude = ['date']
        return [col for col in df.columns if col not in exclude]


class CropTrainingDataGenerator:
    """
    Generates labeled training data for the Random Forest crop suitability model.
    
    Creates a dataset by combining:
    - All crops × regions × seasons × soil types × weather scenarios
    - Uses existing rule-based scoring as labels
    """
    
    def __init__(self):
        from src.crops.database import crop_db, CROPS_DATA
        from src.utils.regions import RegionManager
        from src.services.recommender import calculate_suitability_score
        
        self.crop_db = crop_db
        self.crops_data = CROPS_DATA
        self.region_manager = RegionManager()
        self.calculate_score = calculate_suitability_score
    
    def generate_training_data(
        self,
        num_weather_scenarios: int = 50,
        random_seed: int = 42,
        output_path: str = None,
        chunk_size: int = 50_000,
    ) -> pd.DataFrame:
        """
        Generate labeled training data for crop suitability model.

        Records are written to *output_path* in chunks to avoid RAM exhaustion
        when the combination space is large (many regions × crops × scenarios).
        If *output_path* is None the full DataFrame is returned in memory
        (only safe for small datasets / testing).

        Args:
            num_weather_scenarios: Number of random weather scenarios per combination
            random_seed: Random seed for reproducibility
            output_path: CSV path to stream records into (recommended for production)
            chunk_size: Flush to disk every this many records (default 50 000)

        Returns:
            DataFrame with features and suitability_score label
        """
        import csv, os

        np.random.seed(random_seed)

        regions = self.region_manager.get_all_regions()
        seasons = ['Kharif', 'Rabi', 'Zaid']
        soil_textures = ['Clay', 'Loam', 'Sandy', 'Clay-Loam', 'Sandy-Loam']
        irrigation_options = [True, False]

        chunk: list = []
        total_written = 0
        header_written = False
        file_handle = None

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            file_handle = open(output_path, "w", newline="", encoding="utf-8")
            csv_writer = None  # initialised on first record so we know field names

        try:
            for crop in self.crops_data:
                for region in regions:
                    for season in seasons:
                        for texture in soil_textures:
                            for irrigation in irrigation_options:
                                for _ in range(num_weather_scenarios):
                                    record = self._generate_single_record(
                                        crop, region, season, texture, irrigation
                                    )
                                    if record is None:
                                        continue

                                    if output_path:
                                        # Lazy-initialise CSV writer on first valid record
                                        if csv_writer is None:
                                            csv_writer = csv.DictWriter(
                                                file_handle, fieldnames=list(record.keys())
                                            )
                                            csv_writer.writeheader()
                                        chunk.append(record)
                                        if len(chunk) >= chunk_size:
                                            csv_writer.writerows(chunk)
                                            total_written += len(chunk)
                                            logger.info(
                                                f"  Flushed {total_written:,} records → {output_path}"
                                            )
                                            chunk = []
                                    else:
                                        chunk.append(record)

            # Flush remaining records
            if output_path and csv_writer and chunk:
                csv_writer.writerows(chunk)
                total_written += len(chunk)
                chunk = []

        finally:
            if file_handle:
                file_handle.close()

        if output_path:
            logger.info(
                f"Generated {total_written:,} training records → {output_path}"
            )
            return pd.read_csv(output_path)
        else:
            df = pd.DataFrame(chunk)
            logger.info(f"Generated {len(df)} training records for crop suitability model")
            return df
    
    def _generate_single_record(
        self, crop, region, season, soil_texture, irrigation
    ) -> Optional[Dict]:
        """Generate a single training record with random weather."""
        from src.crops.soil import SoilInfo
        
        # Random weather within realistic ranges for the season
        if season == 'Kharif':  # Monsoon - hot, wet
            avg_temp = np.random.uniform(25, 38)
            rainfall = np.random.uniform(200, 1200)
            max_dry_spell = np.random.randint(0, 10)
        elif season == 'Rabi':  # Winter - cool, dry
            avg_temp = np.random.uniform(12, 28)
            rainfall = np.random.uniform(10, 300)
            max_dry_spell = np.random.randint(3, 20)
        else:  # Zaid - hot, dry
            avg_temp = np.random.uniform(28, 42)
            rainfall = np.random.uniform(5, 150)
            max_dry_spell = np.random.randint(5, 25)
        
        # Random soil pH
        soil_ph = np.random.uniform(5.5, 8.5)
        organic_matter = np.random.choice(['Low', 'Medium', 'High'])
        drainage = np.random.choice(['Poor', 'Medium', 'Good'])
        
        soil = SoilInfo(
            texture=soil_texture,
            ph=soil_ph,
            organic_matter=organic_matter,
            drainage=drainage
        )
        
        try:
            # Calculate suitability score using existing rule-based engine
            score = self.calculate_score(
                crop=crop,
                avg_temp=avg_temp,
                expected_rainfall=rainfall,
                max_dry_spell=max_dry_spell,
                season=season,
                region_id=region.region_id,
                soil=soil,
                irrigation_available=irrigation
            )
            
            # Create feature record
            record = {
                'crop_id': crop.crop_id,
                'region_id': region.region_id,
                'season': season,
                'avg_temp': round(avg_temp, 2),
                'total_rainfall': round(rainfall, 2),
                'max_dry_spell': max_dry_spell,
                'soil_texture': soil_texture,
                'soil_ph': round(soil_ph, 2),
                'organic_matter': organic_matter,
                'drainage': drainage,
                'irrigation': int(irrigation),  # 0 or 1 — consistent with prediction encoding
                'crop_temp_min': crop.temp_min,
                'crop_temp_max': crop.temp_max,
                'crop_water_req': crop.water_requirement_mm,
                'crop_duration': crop.duration_days,
                'drought_tolerance': crop.drought_tolerance,
                'regional_suitability': crop.regional_suitability.get(region.region_id, 0.5),
                # Add small Gaussian noise to labels so RF sees a real distribution
                # and learns non-trivial boundaries (not a pure copy of rule-based engine)
                'suitability_score': round(min(100.0, max(0.0, score + np.random.normal(0, 3))), 2)
            }
            return record
        except Exception as e:
            logger.debug(f"Skipping record: {e}")
            return None
