"""
Random Forest Crop Suitability Prediction Model

Predicts crop suitability scores using a Random Forest regressor
trained on features derived from weather, soil, region, and crop data.

How it works (basics):
    1. Takes weather conditions, soil properties, region, season, and crop info
    2. Random Forest (ensemble of decision trees) learns non-linear relationships
    3. Outputs a suitability score (0-100) for each crop
    4. Feature importance analysis reveals which factors matter most

Integration:
    - ML score is blended with rule-based score: 
      final = 0.6 * ml_score + 0.4 * rule_based_score
    - Falls back to pure rule-based if ML model unavailable
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import logging
import json

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models/crop_suitability")


# ── Inline metric helpers (previously in deleted ml/utils.py) ───────────────

def _calculate_metrics(y_true, y_pred):
    """MAE, RMSE, R², MAPE."""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2   = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else 0.0
    return {'mae': round(float(mae), 4), 'rmse': round(float(rmse), 4),
            'r2':  round(float(r2),  4), 'mape': round(float(mape), 2)}


def _print_metrics(metrics, model_name="Model"):
    """Pretty-print evaluation metrics."""
    print(f"\n{'='*50}\n  {model_name}\n{'='*50}")
    print(f"  MAE : {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R²  : {metrics['r2']:.4f}")
    print(f"  MAPE: {metrics['mape']:.2f}%\n{'='*50}\n")



class CropSuitabilityRF:
    """
    Random Forest model for predicting crop suitability scores.
    
    Example:
        >>> model = CropSuitabilityRF()
        >>> model.train(training_data_df)
        >>> score = model.predict_score(crop_features)
    """
    
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_names = None
        self.feature_importances = None
        self.categorical_cols = [
            'crop_id', 'region_id', 'season', 
            'soil_texture', 'organic_matter', 'drainage', 'drought_tolerance'
        ]
        self.numerical_cols = [
            'avg_temp', 'total_rainfall', 'max_dry_spell',
            'soil_ph', 'crop_temp_min', 'crop_temp_max',
            'crop_water_req', 'crop_duration', 'regional_suitability'
        ]
    
    def train(
        self,
        data: pd.DataFrame,
        target_col: str = 'suitability_score',
        n_estimators: int = 200,
        max_depth: int = 15,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict:
        """
        Train the Random Forest model on labeled crop suitability data.
        
        Args:
            data: Training DataFrame with features and target
            target_col: Name of the target column
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of each tree
            test_size: Fraction of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Training results with metrics
        """
        from sklearn.model_selection import train_test_split
        
        # Prepare features
        X, feature_names = self._prepare_features(data)
        y = data[target_col].values
        self.feature_names = feature_names
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Training Random Forest: {len(X_train)} train, {len(X_test)} test samples")
        
        # Build and train the model
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        )
        
        self.model.fit(X_train, y_train)
        
        # Store feature importances
        self.feature_importances = self.model.feature_importances_
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_metrics = _calculate_metrics(y_train, train_pred)
        test_metrics  = _calculate_metrics(y_test,  test_pred)
        _print_metrics(train_metrics, "Random Forest — Training")
        _print_metrics(test_metrics,  "Random Forest — Test")
        
        result = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'num_features': len(feature_names),
            'num_train_samples': len(X_train),
            'num_test_samples': len(X_test)
        }
        
        logger.info(f"Training complete. Test R²: {test_metrics['r2']:.4f}")
        return result
    
    def predict_score(self, features: Dict) -> float:
        """
        Predict suitability score for a single crop-region-weather combination.
        
        Args:
            features: Dictionary with feature values matching training columns
            
        Returns:
            Predicted suitability score (0-100)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() or load() first.")
        
        # Convert to DataFrame for consistent processing
        df = pd.DataFrame([features])
        X, _ = self._prepare_features(df, fit_encoders=False)
        
        score = float(self.model.predict(X)[0])
        
        # Clamp to valid range
        score = max(0, min(100, score))
        
        return round(score, 2)
    
    def predict_batch(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Predict suitability scores for multiple combinations.
        
        Args:
            features_df: DataFrame with feature columns
            
        Returns:
            Array of predicted suitability scores
        """
        if self.model is None:
            raise ValueError("Model not trained.")
        
        X, _ = self._prepare_features(features_df, fit_encoders=False)
        scores = self.model.predict(X)
        
        # Clamp to valid range
        scores = np.clip(scores, 0, 100)
        
        return np.round(scores, 2)
    
    def get_feature_importance(self, top_n: int = 15) -> List[Dict]:
        """
        Get top-N most important features.
        
        Feature importance in Random Forest = how much each feature
        reduces impurity (Gini/MSE) across all trees, averaged.
        
        Returns:
            List of dicts with feature name and importance
        """
        if self.feature_importances is None or self.feature_names is None:
            return []
        
        indices = np.argsort(self.feature_importances)[-top_n:][::-1]
        
        return [
            {
                'feature': self.feature_names[i],
                'importance': round(float(self.feature_importances[i]), 4)
            }
            for i in indices
        ]
    
    def _prepare_features(
        self, df: pd.DataFrame, fit_encoders: bool = True
    ) -> tuple:
        """
        Prepare feature matrix from raw data.
        
        Encodes categorical variables and selects numerical features.
        
        Args:
            df: Raw data DataFrame
            fit_encoders: If True, fit label encoders (training). 
                         If False, use existing encoders (prediction).
                         
        Returns:
            X (feature matrix), feature_names (list of column names)
        """
        result_df = pd.DataFrame()
        feature_names = []
        
        # Encode categorical columns
        for col in self.categorical_cols:
            if col in df.columns:
                if fit_encoders:
                    le = LabelEncoder()
                    result_df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    le = self.label_encoders.get(col)
                    if le is not None:
                        # Handle unseen labels
                        values = df[col].astype(str)
                        encoded = []
                        for v in values:
                            if v in le.classes_:
                                encoded.append(le.transform([v])[0])
                            else:
                                encoded.append(-1)  # Unknown
                        result_df[col] = encoded
                    else:
                        result_df[col] = 0
                feature_names.append(col)
        
        # Add numerical columns
        for col in self.numerical_cols:
            if col in df.columns:
                result_df[col] = df[col].values
                feature_names.append(col)
        
        # Add boolean columns
        if 'irrigation' in df.columns:
            result_df['irrigation'] = df['irrigation'].astype(int)
            feature_names.append('irrigation')
        
        X = result_df.values
        
        return X, feature_names
    
    def save(self, model_dir: str = None):
        """Save the trained model and encoders."""
        import joblib
        base_dir = Path(model_dir) if model_dir else MODELS_DIR
        base_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, str(base_dir / "rf_model.joblib"))
        logger.info(f"Saved RF model to {base_dir / 'rf_model.joblib'}")
        
        # Save label encoders
        import joblib
        joblib.dump(self.label_encoders, str(base_dir / "label_encoders.joblib"))
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'categorical_cols': self.categorical_cols,
            'numerical_cols': self.numerical_cols,
            'feature_importances': (
                self.feature_importances.tolist() 
                if self.feature_importances is not None else None
            )
        }
        with open(base_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved Random Forest model to {base_dir}")
    
    @classmethod
    def load(cls, model_dir: str = None) -> Optional['CropSuitabilityRF']:
        """Load a trained model."""
        import joblib
        
        base_dir = Path(model_dir) if model_dir else MODELS_DIR
        if not base_dir.exists():
            logger.warning(f"No crop suitability model found at {base_dir}")
            return None
        
        instance = cls()
        
        model_path = base_dir / "rf_model.joblib"
        if not model_path.exists():
            logger.warning(f"Model file not found: {model_path}")
            return None
        instance.model = joblib.load(str(model_path))
        logger.info(f"Loaded RF model from {model_path}")
        
        # Load encoders
        encoders_path = base_dir / "label_encoders.joblib"
        if encoders_path.exists():
            instance.label_encoders = joblib.load(str(encoders_path))
        
        # Load metadata
        metadata_path = base_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            instance.feature_names = metadata.get('feature_names')
            importances = metadata.get('feature_importances')
            if importances:
                instance.feature_importances = np.array(importances)
        
        logger.info(f"Loaded Random Forest crop suitability model from {base_dir}")
        return instance


# ---------------------------------------------------------------------------
# Visualisation helper
# (previously in src/ml/utils.py – merged here as this is its only caller)
# ---------------------------------------------------------------------------

def plot_feature_importance(
    feature_names: list,
    importances,
    title: str = "Feature Importance",
    save_path=None,
    top_n: int = 20,
):
    """Plot and optionally save a horizontal bar chart of RF feature importances."""
    try:
        import matplotlib.pyplot as plt

        indices = np.argsort(importances)[-top_n:]
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(indices)), importances[indices], color="forestgreen", alpha=0.8)
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel("Importance")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved feature importance plot to {save_path}")
        plt.close(fig)
    except ImportError:
        logger.warning("matplotlib not available — skipping feature importance plot")

