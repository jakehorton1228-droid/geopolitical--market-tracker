"""
Model Explainability Module using SHAP.

Provides interpretable explanations for ML model predictions using
SHAP (SHapley Additive exPlanations) values.

WHY SHAP?
---------
In defense and mission-critical applications, it's not enough to know
WHAT a model predicts - you need to know WHY. SHAP provides:
1. Feature importance that's theoretically grounded (Shapley values)
2. Per-prediction explanations (not just global importance)
3. Feature interaction detection
4. Consistent and locally accurate attributions

USAGE:
------
    from src.analysis.explainability import ModelExplainer

    explainer = ModelExplainer()
    result = explainer.explain_model("CL=F", start_date, end_date)

    # Global feature importance
    print(result.feature_importance)

    # Explain a single prediction
    explanation = explainer.explain_prediction("CL=F", features, start_date, end_date)
    print(explanation.narrative)
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional
import logging
import io

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class GlobalExplanation:
    """Global model explanation with SHAP values."""
    symbol: str
    model_type: str  # "xgboost" or "lightgbm"

    # Feature importance (mean absolute SHAP values)
    feature_importance: dict[str, float]

    # Raw SHAP values matrix (n_samples x n_features)
    shap_values: Optional[np.ndarray] = None

    # Feature names
    feature_names: list[str] = field(default_factory=list)

    # Feature interaction strengths (top pairs)
    interactions: dict[str, float] = field(default_factory=dict)

    # Model performance
    cv_accuracy: float = 0.0

    # Summary text
    summary: str = ""


@dataclass
class PredictionExplanation:
    """Explanation for a single prediction."""
    symbol: str
    prediction: str  # "UP" or "DOWN"
    probability: float

    # Per-feature contributions
    feature_contributions: dict[str, float]
    base_value: float

    # Human-readable explanation
    narrative: str = ""

    # Top positive and negative drivers
    top_positive: list[tuple[str, float]] = field(default_factory=list)
    top_negative: list[tuple[str, float]] = field(default_factory=list)


class ModelExplainer:
    """
    SHAP-based model explainability for gradient boosting models.

    Supports both global explanations (which features matter overall)
    and local explanations (why this specific prediction was made).
    """

    def __init__(self):
        self._trained_models: dict[str, dict] = {}

    def _train_model(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> Optional[dict]:
        """Train a model and cache it for explanation."""
        if symbol in self._trained_models:
            return self._trained_models[symbol]

        try:
            from src.analysis.gradient_boost_classifier import GradientBoostClassifier

            classifier = GradientBoostClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5
            )
            comparison = classifier.train_and_compare(symbol, start_date, end_date)

            if comparison is None:
                logger.warning(f"Insufficient data to train model for {symbol}")
                return None

            self._trained_models[symbol] = {
                "classifier": classifier,
                "comparison": comparison,
            }
            return self._trained_models[symbol]

        except Exception as e:
            logger.error(f"Failed to train model for {symbol}: {e}")
            return None

    def explain_model(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        model_type: str = "xgboost",
    ) -> Optional[GlobalExplanation]:
        """
        Generate global SHAP explanation for a trained model.

        Returns feature importance based on mean absolute SHAP values,
        which is more reliable than the built-in feature importance
        (gain-based) because SHAP accounts for feature interactions.
        """
        try:
            import shap
        except ImportError:
            logger.error("shap package not installed. Run: pip install shap")
            return None

        trained = self._train_model(symbol, start_date, end_date)
        if trained is None:
            return None

        classifier = trained["classifier"]
        comparison = trained["comparison"]

        # Get the trained model and training data
        if model_type == "xgboost":
            model = classifier.xgb_models.get(symbol)
            cv_accuracy = comparison.xgboost_metrics.cv_accuracy
        else:
            model = classifier.lgb_models.get(symbol)
            cv_accuracy = comparison.lightgbm_metrics.cv_accuracy

        if model is None:
            logger.warning(f"No {model_type} model found for {symbol}")
            return None

        # Get training data for SHAP background
        training_data = classifier.training_data.get(symbol)
        if training_data is None:
            logger.warning(f"No training data cached for {symbol}")
            return None

        X_train = training_data["X"]
        feature_names = training_data["feature_names"]

        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)

        # Handle binary classification output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Class 1 (UP) explanations

        # Calculate mean absolute SHAP values per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = dict(zip(feature_names, mean_abs_shap.tolist()))

        # Sort by importance
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )

        # Calculate feature interactions (top pairs)
        interactions = {}
        if len(feature_names) > 1:
            try:
                shap_interaction = explainer.shap_interaction_values(
                    X_train[:min(100, len(X_train))]
                )
                if isinstance(shap_interaction, list):
                    shap_interaction = shap_interaction[1]

                n_features = len(feature_names)
                for i in range(n_features):
                    for j in range(i + 1, n_features):
                        interaction_strength = np.abs(
                            shap_interaction[:, i, j]
                        ).mean()
                        pair_name = f"{feature_names[i]} x {feature_names[j]}"
                        interactions[pair_name] = float(interaction_strength)

                interactions = dict(
                    sorted(interactions.items(), key=lambda x: x[1], reverse=True)
                )
            except Exception:
                # Interaction values not always available
                pass

        # Build summary
        top_feature = list(feature_importance.keys())[0]
        summary = (
            f"SHAP Analysis for {symbol} ({model_type}):\n"
            f"Most important feature: {top_feature} "
            f"(mean |SHAP| = {feature_importance[top_feature]:.4f})\n"
            f"Model CV accuracy: {cv_accuracy:.1%}\n"
            f"Training samples: {len(X_train)}"
        )

        return GlobalExplanation(
            symbol=symbol,
            model_type=model_type,
            feature_importance=feature_importance,
            shap_values=shap_values,
            feature_names=feature_names,
            interactions=interactions,
            cv_accuracy=cv_accuracy,
            summary=summary,
        )

    def explain_prediction(
        self,
        symbol: str,
        features: dict[str, float],
        start_date: date,
        end_date: date,
        model_type: str = "xgboost",
    ) -> Optional[PredictionExplanation]:
        """
        Explain a single prediction using SHAP values.

        Shows which features pushed the prediction toward UP vs DOWN,
        and generates a human-readable narrative.
        """
        try:
            import shap
        except ImportError:
            logger.error("shap package not installed")
            return None

        trained = self._train_model(symbol, start_date, end_date)
        if trained is None:
            return None

        classifier = trained["classifier"]

        # Get model
        if model_type == "xgboost":
            model = classifier.xgb_models.get(symbol)
        else:
            model = classifier.lgb_models.get(symbol)

        if model is None:
            return None

        training_data = classifier.training_data.get(symbol)
        if training_data is None:
            return None

        feature_names = training_data["feature_names"]

        # Build feature vector
        X_single = np.array([[features.get(f, 0) for f in feature_names]])

        # Get prediction
        pred = classifier.predict(symbol, features, model_type)
        if pred is None:
            return None

        # Get SHAP values for this prediction
        explainer = shap.TreeExplainer(model)
        shap_values_single = explainer.shap_values(X_single)

        if isinstance(shap_values_single, list):
            shap_values_single = shap_values_single[1]

        shap_vals = shap_values_single[0]
        base_value = float(explainer.expected_value)
        if isinstance(base_value, np.ndarray):
            base_value = float(base_value[1]) if len(base_value) > 1 else float(base_value[0])

        # Feature contributions
        contributions = dict(zip(feature_names, shap_vals.tolist()))

        # Sort into positive and negative drivers
        sorted_contribs = sorted(contributions.items(), key=lambda x: x[1])
        top_negative = [(k, v) for k, v in sorted_contribs if v < 0][:3]
        top_positive = [(k, v) for k, v in sorted_contribs if v > 0][-3:][::-1]

        # Build narrative
        narrative_parts = [
            f"Prediction for {symbol}: {pred.prediction} "
            f"(probability {pred.probability:.1%})",
            "",
            "Key drivers pushing UP:",
        ]
        for feat, val in top_positive:
            feat_val = features.get(feat, 0)
            narrative_parts.append(
                f"  + {feat} = {feat_val:.2f} (SHAP: +{val:.4f})"
            )

        narrative_parts.append("")
        narrative_parts.append("Key drivers pushing DOWN:")
        for feat, val in top_negative:
            feat_val = features.get(feat, 0)
            narrative_parts.append(
                f"  - {feat} = {feat_val:.2f} (SHAP: {val:.4f})"
            )

        return PredictionExplanation(
            symbol=symbol,
            prediction=pred.prediction,
            probability=pred.probability,
            feature_contributions=contributions,
            base_value=base_value,
            narrative="\n".join(narrative_parts),
            top_positive=top_positive,
            top_negative=top_negative,
        )

    def get_shap_summary_plot_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        model_type: str = "xgboost",
    ) -> Optional[pd.DataFrame]:
        """
        Get SHAP values as a DataFrame for custom visualization.

        Returns a DataFrame with columns for each feature containing
        their SHAP values across all training samples.
        """
        explanation = self.explain_model(symbol, start_date, end_date, model_type)
        if explanation is None or explanation.shap_values is None:
            return None

        return pd.DataFrame(
            explanation.shap_values,
            columns=explanation.feature_names,
        )
