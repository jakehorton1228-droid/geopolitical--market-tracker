"""
Model Loader — loads the champion model from MLflow and serves predictions.

The training pipeline registers the winning model in MLflow's Model Registry
with the alias "champion". This module loads that model (plus its scaler and
feature names) once at startup and caches it in memory. Predictions reuse
the same preprocessing the model was trained with.

If no champion is registered (e.g., before the first training run), the
calling code should fall back to the legacy LogisticRegressionAnalyzer.

USAGE:
------
    from src.ml.model_loader import get_champion_model

    champion = get_champion_model()
    if champion:
        prediction = champion.predict(feature_dict)
        print(prediction)
        #   {
        #     "model_name": "xgboost",
        #     "version": 3,
        #     "prediction": "UP",
        #     "probability_up": 0.67,
        #     "test_auc": 0.72,
        #     "feature_contributions": [...],
        #   }
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
import mlflow
from mlflow.tracking import MlflowClient

from src.config.settings import MLFLOW_TRACKING_URI
from src.training.trainer import CHAMPION_MODEL_NAME, CHAMPION_ALIAS

logger = logging.getLogger(__name__)


class ChampionModel:
    """Wraps a loaded MLflow champion model with its preprocessing pipeline."""

    def __init__(
        self,
        model,
        scaler,
        feature_names: list[str],
        model_name: str,
        version: str,
        test_auc: float,
    ):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.model_name = model_name
        self.version = version
        self.test_auc = test_auc

    def predict(self, features: dict[str, float]) -> dict:
        """Run a prediction with a feature dict.

        Args:
            features: Dict mapping feature name to value. Unknown features
                are set to 0; missing features also default to 0.

        Returns:
            Dict with prediction (UP/DOWN), probability_up, and per-feature
            contributions.
        """
        # Build feature vector in the order the model expects
        feature_vector = np.array([
            features.get(name, 0.0) for name in self.feature_names
        ]).reshape(1, -1)

        # Apply the same scaling used during training
        feature_vector_scaled = self.scaler.transform(feature_vector)

        # Predict probability
        probs = self.model.predict_proba(feature_vector_scaled)[0]
        prob_up = float(probs[1]) if len(probs) > 1 else float(probs[0])
        prediction = "UP" if prob_up >= 0.5 else "DOWN"

        # Per-feature contributions: only meaningful for linear models, but
        # we still return the raw feature values for all model types so the
        # UI can show what went into the prediction
        contributions = []
        for i, name in enumerate(self.feature_names):
            raw_value = float(feature_vector[0, i])
            scaled_value = float(feature_vector_scaled[0, i])
            contributions.append({
                "feature": name,
                "value": raw_value,
                "scaled_value": scaled_value,
            })

        return {
            "model_name": self.model_name,
            "model_source": "champion",
            "version": self.version,
            "auc": self.test_auc,
            "prediction": prediction,
            "probability_up": round(prob_up, 4),
            "feature_contributions": contributions,
        }

    def metadata(self) -> dict:
        """Return model metadata for display (without running a prediction)."""
        return {
            "model_name": self.model_name,
            "model_source": "champion",
            "version": self.version,
            "auc": self.test_auc,
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
        }


# Module-level cache — loaded once, reused across requests
_CACHED_CHAMPION: Optional[ChampionModel] = None
_CACHE_ATTEMPTED = False


def get_champion_model(force_reload: bool = False) -> Optional[ChampionModel]:
    """Load the champion model from MLflow, or return the cached copy.

    Returns None if no champion is registered yet. Callers should fall
    back to the legacy predictor when this returns None.

    Args:
        force_reload: If True, ignore the cache and reload from MLflow.
            Use this after a training run to pick up the new champion.
    """
    global _CACHED_CHAMPION, _CACHE_ATTEMPTED

    if _CACHED_CHAMPION is not None and not force_reload:
        return _CACHED_CHAMPION

    if _CACHE_ATTEMPTED and not force_reload:
        # Already tried and failed — don't keep retrying on every request.
        # A subsequent training run will call reload_champion() to reset this.
        return None

    _CACHE_ATTEMPTED = True

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

        # Get the model version aliased as "champion"
        champion_version = client.get_model_version_by_alias(
            CHAMPION_MODEL_NAME, CHAMPION_ALIAS,
        )
        run_id = champion_version.run_id
        model_uri = f"models:/{CHAMPION_MODEL_NAME}@{CHAMPION_ALIAS}"

        logger.info(f"Loading champion model from {model_uri}")

        # Load the model via MLflow's universal loader (handles sklearn/xgb/lgbm)
        model = mlflow.sklearn.load_model(model_uri)

        # Load the preprocessing artifacts from the run
        scaler, feature_names = _load_preprocessing(client, run_id)

        # Extract metadata from tags
        model_name = champion_version.tags.get("model_type", "unknown")
        test_auc = float(champion_version.tags.get("test_auc", 0.0))

        _CACHED_CHAMPION = ChampionModel(
            model=model,
            scaler=scaler,
            feature_names=feature_names,
            model_name=model_name,
            version=champion_version.version,
            test_auc=test_auc,
        )

        logger.info(
            f"Loaded champion: {model_name} v{champion_version.version} "
            f"(AUC={test_auc:.4f}, {len(feature_names)} features)"
        )

        return _CACHED_CHAMPION

    except Exception as e:
        logger.info(f"No champion model available yet: {e}")
        _CACHED_CHAMPION = None
        return None


def reload_champion() -> Optional[ChampionModel]:
    """Force-reload the champion model. Call after a new training run."""
    global _CACHED_CHAMPION, _CACHE_ATTEMPTED
    _CACHED_CHAMPION = None
    _CACHE_ATTEMPTED = False
    return get_champion_model()


def _load_preprocessing(
    client: MlflowClient, run_id: str,
) -> tuple[object, list[str]]:
    """Download the scaler and feature names from an MLflow run."""
    with tempfile.TemporaryDirectory() as tmp:
        local_dir = client.download_artifacts(
            run_id=run_id,
            path="preprocessing",
            dst_path=tmp,
        )
        scaler_path = Path(local_dir) / "scaler.joblib"
        features_path = Path(local_dir) / "feature_names.json"

        scaler = joblib.load(scaler_path)
        feature_names = json.loads(features_path.read_text())

    return scaler, feature_names
