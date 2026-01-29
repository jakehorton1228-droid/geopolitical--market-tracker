"""Tests for the explainability module."""

import sys
from pathlib import Path
from datetime import date, timedelta
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestModelExplainer:
    """Tests for the ModelExplainer class."""

    def _make_explainer(self):
        from src.analysis.explainability import ModelExplainer
        return ModelExplainer()

    def _make_mock_classifier(self, n_samples=100, n_features=9):
        """Create a mock GradientBoostClassifier with training data."""
        np.random.seed(42)

        feature_names = [
            "goldstein_mean", "goldstein_min", "goldstein_max", "goldstein_std",
            "mentions_total", "mentions_max", "avg_tone",
            "conflict_count", "cooperation_count",
        ][:n_features]

        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] > 0).astype(int)

        # Train a real xgboost model
        try:
            import xgboost as xgb
            model = xgb.XGBClassifier(
                n_estimators=10,
                max_depth=3,
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss",
            )
            model.fit(X, y)
        except ImportError:
            pytest.skip("xgboost not installed")
            return None, None, None, None

        classifier = MagicMock()
        classifier.models = {"SPY": {"xgboost": model}}
        classifier.training_data = {
            "SPY": {
                "X": X,
                "y": y,
                "feature_names": feature_names,
            }
        }

        return classifier, X, y, feature_names

    def test_explain_model_returns_global_explanation(self):
        """explain_model should return a GlobalExplanation."""
        try:
            import shap
        except ImportError:
            pytest.skip("shap not installed")

        classifier, X, y, names = self._make_mock_classifier()
        if classifier is None:
            return

        explainer = self._make_explainer()
        result = explainer.explain_model(classifier, "SPY")

        assert result is not None
        assert len(result.feature_importance) > 0
        assert result.model_type in ("xgboost", "lightgbm")

        # Feature importance should be sorted descending
        values = list(result.feature_importance.values())
        assert values == sorted(values, reverse=True)

    def test_explain_prediction_returns_explanation(self):
        """explain_prediction should return a PredictionExplanation."""
        try:
            import shap
        except ImportError:
            pytest.skip("shap not installed")

        classifier, X, y, names = self._make_mock_classifier()
        if classifier is None:
            return

        explainer = self._make_explainer()
        result = explainer.explain_prediction(classifier, "SPY", X[0])

        assert result is not None
        assert len(result.feature_contributions) > 0
        assert result.predicted_class in (0, 1)
        assert len(result.narrative) > 0

    def test_explain_model_missing_symbol(self):
        """Should return None for a symbol without training data."""
        try:
            import shap
        except ImportError:
            pytest.skip("shap not installed")

        classifier, _, _, _ = self._make_mock_classifier()
        if classifier is None:
            return

        explainer = self._make_explainer()
        result = explainer.explain_model(classifier, "NONEXISTENT")

        assert result is None
