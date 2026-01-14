"""
Gradient Boosting Classifier for Market Direction Prediction.

================================================================================
WHAT IS GRADIENT BOOSTING? (The Intuition)
================================================================================

Imagine you're trying to predict if the market will go UP or DOWN.

1. SINGLE DECISION TREE (weak):
   - Makes simple rules like "if goldstein_scale < -2, predict DOWN"
   - Easy to understand but not very accurate
   - Like asking ONE person for their opinion

2. RANDOM FOREST (bagging):
   - Trains MANY trees on random subsets of data
   - Each tree votes, majority wins
   - Like asking 100 random people and going with majority
   - Trees are trained IN PARALLEL (independently)

3. GRADIENT BOOSTING (boosting):
   - Trains trees SEQUENTIALLY, each one fixing previous mistakes
   - Tree 1 makes predictions → Find errors
   - Tree 2 focuses on those errors → Find remaining errors
   - Tree 3 focuses on remaining errors → And so on...
   - Like having experts who each specialize in what others got wrong

WHY "GRADIENT"?
---------------
The "gradient" refers to gradient descent optimization. Each new tree is trained
to predict the GRADIENT (direction of steepest improvement) of the loss function.
In simpler terms: each tree learns "which way to adjust predictions to reduce errors."

================================================================================
XGBOOST vs LIGHTGBM
================================================================================

Both are gradient boosting implementations, but with different optimizations:

XGBOOST (eXtreme Gradient Boosting):
- The original "fast" gradient boosting library (2014)
- Grows trees LEVEL-BY-LEVEL (all nodes at depth 1, then depth 2, etc.)
- More regularization options built-in
- Generally more robust to overfitting
- Slightly slower on large datasets

LIGHTGBM (Light Gradient Boosting Machine):
- Microsoft's implementation (2017)
- Grows trees LEAF-BY-LEAF (picks the leaf that reduces loss most)
- Uses histogram-based splitting (buckets continuous values)
- Much faster on large datasets
- Can overfit more easily on small data
- Native support for categorical features

For this project, both will work well. We'll implement both so you can compare.

================================================================================
KEY HYPERPARAMETERS
================================================================================

1. n_estimators (num_trees):
   - How many trees to build
   - More trees = more capacity to learn, but slower and risk of overfitting
   - Typical: 100-1000

2. learning_rate (eta):
   - How much each tree contributes to final prediction
   - Lower = each tree has smaller impact, need more trees
   - Higher = faster learning but can overshoot optimal solution
   - Typical: 0.01-0.3
   - TRADE-OFF: learning_rate ↓ means n_estimators ↑ for same performance

3. max_depth:
   - Maximum depth of each tree
   - Deeper = more complex patterns, but risk of overfitting
   - XGBoost default: 6, LightGBM default: -1 (unlimited)
   - Typical: 3-10

4. subsample:
   - Fraction of training data used for each tree
   - Like random forest's bootstrap sampling
   - Reduces overfitting through randomization
   - Typical: 0.7-1.0

5. colsample_bytree:
   - Fraction of features used for each tree
   - Also reduces overfitting through randomization
   - Typical: 0.7-1.0

6. reg_alpha (L1) and reg_lambda (L2):
   - Regularization to prevent overfitting
   - L1 (lasso): Can zero out unimportant features
   - L2 (ridge): Shrinks all coefficients
   - Typical: 0-10

================================================================================
USAGE
================================================================================

    from src.analysis.gradient_boost_classifier import GradientBoostClassifier

    # Train with both XGBoost and LightGBM
    classifier = GradientBoostClassifier()
    comparison = classifier.train_and_compare("SPY", start_date, end_date)

    # See which model performed better
    print(comparison)

    # Make predictions
    prediction = classifier.predict("SPY", features, model_type="xgboost")

    # Get feature importance (which features matter most)
    importance = classifier.get_feature_importance("SPY", model_type="xgboost")
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Optional, Literal
import logging

import numpy as np
import pandas as pd

# XGBoost: The original fast gradient boosting library
# Under the hood, it's implemented in C++ for speed
import xgboost as xgb

# LightGBM: Microsoft's faster alternative
# Uses histogram-based algorithms for even faster training
import lightgbm as lgb

# Sklearn utilities - these work with any model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from src.config.constants import get_all_symbols
from src.db.connection import get_session
from src.db.queries import get_events_by_date_range, get_market_data

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES FOR RESULTS
# =============================================================================

@dataclass
class GBClassificationResult:
    """
    Container for a single prediction.

    WHY DATACLASS?
    - Automatically generates __init__, __repr__, __eq__
    - Cleaner than dictionaries, with type hints
    - Immutable-ish (can set frozen=True for true immutability)
    """
    symbol: str
    prediction: str  # "UP" or "DOWN"
    probability: float  # Probability of UP (0 to 1)
    confidence: str  # "high", "medium", "low"
    model_type: str  # "xgboost" or "lightgbm"


@dataclass
class GBModelMetrics:
    """
    Container for model evaluation metrics.

    KEY METRICS EXPLAINED:
    - accuracy: Overall % correct (can be misleading if classes imbalanced)
    - precision: Of all UP predictions, what % were actually UP?
    - recall: Of all actual UPs, what % did we catch?
    - f1_score: Harmonic mean of precision and recall
    - roc_auc: Area Under ROC Curve (0.5 = random, 1.0 = perfect)
    """
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    cv_accuracy: float  # Cross-validation accuracy (more reliable)
    cv_std: float  # Standard deviation across CV folds
    n_samples: int
    model_type: str


@dataclass
class ModelComparison:
    """Compare XGBoost vs LightGBM performance."""
    symbol: str
    xgboost_metrics: Optional[GBModelMetrics]
    lightgbm_metrics: Optional[GBModelMetrics]
    winner: str  # "xgboost", "lightgbm", or "tie"
    feature_importance_xgb: dict = field(default_factory=dict)
    feature_importance_lgb: dict = field(default_factory=dict)


# =============================================================================
# MAIN CLASSIFIER CLASS
# =============================================================================

class GradientBoostClassifier:
    """
    Gradient Boosting classifier using both XGBoost and LightGBM.

    This class allows you to:
    1. Train both models and compare them
    2. Use whichever performs better for predictions
    3. Understand which features drive predictions (interpretability)

    WHY TWO MODELS?
    Different models have different strengths. By training both, you can:
    - See if they agree (higher confidence when they do)
    - Use the better one for each symbol
    - Ensemble them for even better predictions
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        random_state: int = 42,
    ):
        """
        Initialize classifier with hyperparameters.

        These defaults are good starting points. We use the same hyperparameters
        for both XGBoost and LightGBM so we can fairly compare them.

        Args:
            n_estimators: Number of trees to build (more = more capacity)
            learning_rate: How much each tree contributes (lower = slower learning)
            max_depth: Maximum tree depth (deeper = more complex patterns)
            random_state: Seed for reproducibility (same seed = same results)
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

        # Storage for trained models (one per symbol, per model type)
        self.xgb_models: dict[str, xgb.XGBClassifier] = {}
        self.lgb_models: dict[str, lgb.LGBMClassifier] = {}
        self.scalers: dict[str, StandardScaler] = {}
        self.feature_names: list[str] = []

    # =========================================================================
    # FEATURE ENGINEERING
    # =========================================================================

    def _prepare_features(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Prepare feature matrix (X) and target vector (y).

        FEATURE ENGINEERING is crucial for ML success. The raw data isn't
        directly usable - we need to transform it into meaningful features.

        Our features capture:
        1. Event severity (goldstein_scale statistics)
        2. Media attention (mention counts)
        3. Sentiment (avg_tone)
        4. Event types (conflict vs cooperation counts)

        Returns:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target vector of shape (n_samples,) with 1=UP, 0=DOWN
            feature_names: List of feature names for interpretability
        """
        # ----- Step 1: Get market data -----
        with get_session() as session:
            market_data = get_market_data(session, symbol, start_date, end_date)
            if not market_data:
                logger.warning(f"No market data for {symbol}")
                return np.array([]), np.array([]), []

            # Convert to DataFrame for easier manipulation
            market_df = pd.DataFrame([
                {"date": m.date, "log_return": m.log_return}
                for m in market_data
            ])

        # ----- Step 2: Get event data -----
        with get_session() as session:
            events = get_events_by_date_range(session, start_date, end_date)
            if not events:
                logger.warning(f"No events found for date range")
                return np.array([]), np.array([]), []

            # Extract features from each event
            # WHY THESE FEATURES?
            # - goldstein_scale: Measures event severity (-10 to +10)
            # - num_mentions: Proxy for event importance
            # - avg_tone: Sentiment of news coverage
            # - is_conflict: Military/violent events (codes 18-20)
            # - is_cooperation: Positive diplomatic events (codes 03-06)
            events_df = pd.DataFrame([
                {
                    "date": e.event_date,
                    "goldstein_scale": e.goldstein_scale or 0,
                    "num_mentions": e.num_mentions or 0,
                    "avg_tone": e.avg_tone or 0,
                    "is_conflict": 1 if e.event_root_code in ["18", "19", "20"] else 0,
                    "is_cooperation": 1 if e.event_root_code in ["03", "04", "05", "06"] else 0,
                }
                for e in events
            ])

        # ----- Step 3: Aggregate events by date -----
        # Multiple events happen per day, so we aggregate them.
        # WHY THESE AGGREGATIONS?
        # - mean: Overall sentiment that day
        # - min/max: Captures extreme events (one bad event can move markets)
        # - sum: Total volume of mentions/conflicts
        event_agg = events_df.groupby("date").agg({
            "goldstein_scale": ["mean", "min", "max", "std"],  # Added std for volatility
            "num_mentions": ["sum", "max"],  # Total and peak attention
            "avg_tone": "mean",
            "is_conflict": "sum",
            "is_cooperation": "sum",
        }).reset_index()

        # Flatten hierarchical column names
        event_agg.columns = [
            "date",
            "goldstein_mean", "goldstein_min", "goldstein_max", "goldstein_std",
            "mentions_total", "mentions_max",
            "avg_tone",
            "conflict_count", "cooperation_count"
        ]

        # Fill NaN standard deviation (happens when only 1 event)
        event_agg["goldstein_std"] = event_agg["goldstein_std"].fillna(0)

        # ----- Step 4: Merge market and event data -----
        merged = pd.merge(market_df, event_agg, on="date", how="left")

        # Fill missing event data with neutral values
        # (days with no events get zeros)
        merged = merged.fillna(0)

        # Remove rows where we can't calculate return
        merged = merged.dropna(subset=["log_return"])

        # Need minimum samples for meaningful training
        if len(merged) < 30:
            logger.warning(f"Insufficient data for {symbol}: {len(merged)} samples")
            return np.array([]), np.array([]), []

        # ----- Step 5: Create target variable -----
        # Binary classification: UP (1) if positive return, DOWN (0) otherwise
        y = (merged["log_return"] > 0).astype(int).values

        # ----- Step 6: Create feature matrix -----
        feature_cols = [
            "goldstein_mean", "goldstein_min", "goldstein_max", "goldstein_std",
            "mentions_total", "mentions_max",
            "avg_tone",
            "conflict_count", "cooperation_count"
        ]
        X = merged[feature_cols].values

        logger.info(f"Prepared {len(X)} samples with {len(feature_cols)} features for {symbol}")

        return X, y, feature_cols

    # =========================================================================
    # MODEL TRAINING
    # =========================================================================

    def _create_xgb_model(self) -> xgb.XGBClassifier:
        """
        Create an XGBoost classifier with our hyperparameters.

        XGBoost-specific settings explained:
        - use_label_encoder=False: Avoids deprecated warning
        - eval_metric='logloss': Log loss for binary classification
        - tree_method='hist': Histogram-based (faster)
        """
        return xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,

            # XGBoost-specific
            use_label_encoder=False,
            eval_metric='logloss',
            tree_method='hist',  # Faster histogram-based algorithm

            # Regularization to prevent overfitting
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=1.0,  # L2 regularization

            # Subsampling for robustness
            subsample=0.8,
            colsample_bytree=0.8,
        )

    def _create_lgb_model(self) -> lgb.LGBMClassifier:
        """
        Create a LightGBM classifier with our hyperparameters.

        LightGBM-specific settings explained:
        - boosting_type='gbdt': Gradient Boosting Decision Tree (default)
        - num_leaves: Max leaves per tree (controls complexity)
        - verbose=-1: Suppress training output
        """
        return lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,

            # LightGBM-specific
            boosting_type='gbdt',
            num_leaves=31,  # Default, should be < 2^max_depth
            verbose=-1,  # Suppress output

            # Regularization
            reg_alpha=0.1,
            reg_lambda=1.0,

            # Subsampling
            subsample=0.8,
            colsample_bytree=0.8,
        )

    def _train_single_model(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str,
    ) -> GBModelMetrics:
        """
        Train a single model and compute metrics.

        CROSS-VALIDATION EXPLAINED:
        Instead of one train/test split, we do K splits:

        Fold 1: [TEST] [TRAIN] [TRAIN] [TRAIN] [TRAIN]
        Fold 2: [TRAIN] [TEST] [TRAIN] [TRAIN] [TRAIN]
        Fold 3: [TRAIN] [TRAIN] [TEST] [TRAIN] [TRAIN]
        Fold 4: [TRAIN] [TRAIN] [TRAIN] [TEST] [TRAIN]
        Fold 5: [TRAIN] [TRAIN] [TRAIN] [TRAIN] [TEST]

        Each fold gives us an accuracy. We report the mean and std.
        This is more reliable than a single split because:
        - Every sample gets to be in the test set once
        - We see how much performance varies across splits
        - High std = model is unstable, might not generalize well
        """
        # StratifiedKFold ensures each fold has similar class distribution
        # (important when classes might be imbalanced)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        # Cross-validation scores
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

        # Train final model on ALL data
        # (CV is for evaluation, final model uses everything)
        model.fit(X, y)

        # Predictions on training data (for metrics comparison)
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]  # Probability of class 1 (UP)

        return GBModelMetrics(
            accuracy=accuracy_score(y, y_pred),
            precision=precision_score(y, y_pred, zero_division=0),
            recall=recall_score(y, y_pred, zero_division=0),
            f1_score=f1_score(y, y_pred, zero_division=0),
            roc_auc=roc_auc_score(y, y_prob),
            cv_accuracy=cv_scores.mean(),
            cv_std=cv_scores.std(),
            n_samples=len(y),
            model_type=model_type,
        )

    def train_and_compare(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> Optional[ModelComparison]:
        """
        Train both XGBoost and LightGBM, then compare them.

        This is the main training method. It:
        1. Prepares features
        2. Trains XGBoost
        3. Trains LightGBM
        4. Compares performance
        5. Extracts feature importance from both

        Returns:
            ModelComparison with metrics from both models
        """
        logger.info(f"Training gradient boosting models for {symbol}")

        # Prepare data
        X, y, feature_names = self._prepare_features(symbol, start_date, end_date)

        if len(X) == 0:
            logger.warning(f"No data available for {symbol}")
            return None

        self.feature_names = feature_names

        # Scale features
        # WHY SCALE?
        # Tree-based models don't strictly need scaling (they split on values),
        # but it can help with regularization and makes feature importance
        # more comparable across features with different magnitudes.
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[symbol] = scaler

        # ----- Train XGBoost -----
        logger.info(f"Training XGBoost for {symbol}...")
        xgb_model = self._create_xgb_model()
        xgb_metrics = self._train_single_model(xgb_model, X_scaled, y, "xgboost")
        self.xgb_models[symbol] = xgb_model

        # ----- Train LightGBM -----
        logger.info(f"Training LightGBM for {symbol}...")
        lgb_model = self._create_lgb_model()
        lgb_metrics = self._train_single_model(lgb_model, X_scaled, y, "lightgbm")
        self.lgb_models[symbol] = lgb_model

        # ----- Compare models -----
        # We use CV accuracy as the comparison metric because it's more reliable
        xgb_score = xgb_metrics.cv_accuracy
        lgb_score = lgb_metrics.cv_accuracy

        if abs(xgb_score - lgb_score) < 0.01:  # Within 1% is a tie
            winner = "tie"
        elif xgb_score > lgb_score:
            winner = "xgboost"
        else:
            winner = "lightgbm"

        # ----- Extract feature importance -----
        xgb_importance = self._get_feature_importance_internal(xgb_model, feature_names)
        lgb_importance = self._get_feature_importance_internal(lgb_model, feature_names)

        comparison = ModelComparison(
            symbol=symbol,
            xgboost_metrics=xgb_metrics,
            lightgbm_metrics=lgb_metrics,
            winner=winner,
            feature_importance_xgb=xgb_importance,
            feature_importance_lgb=lgb_importance,
        )

        # Print results
        self._print_comparison(comparison)

        return comparison

    def train_all_symbols(
        self,
        start_date: date,
        end_date: date,
        symbols: list[str] = None,
    ) -> dict[str, Optional[ModelComparison]]:
        """Train models for all symbols and compare."""
        if symbols is None:
            symbols = get_all_symbols()

        results = {}
        for symbol in symbols:
            try:
                comparison = self.train_and_compare(symbol, start_date, end_date)
                results[symbol] = comparison
            except Exception as e:
                logger.error(f"Error training {symbol}: {e}")
                results[symbol] = None

        # Print summary
        self._print_summary(results)

        return results

    # =========================================================================
    # PREDICTION
    # =========================================================================

    def predict(
        self,
        symbol: str,
        features: dict[str, float],
        model_type: Literal["xgboost", "lightgbm"] = "xgboost",
    ) -> Optional[GBClassificationResult]:
        """
        Make a prediction for a symbol using trained model.

        Args:
            symbol: The market symbol to predict
            features: Dictionary of feature values
            model_type: Which model to use ("xgboost" or "lightgbm")

        Returns:
            Prediction result with probability and confidence
        """
        # Get the appropriate model
        if model_type == "xgboost":
            models = self.xgb_models
        else:
            models = self.lgb_models

        if symbol not in models:
            logger.warning(f"No {model_type} model trained for {symbol}")
            return None

        model = models[symbol]
        scaler = self.scalers[symbol]

        # Build feature vector in correct order
        X = np.array([[features.get(name, 0) for name in self.feature_names]])
        X_scaled = scaler.transform(X)

        # Get probability
        prob = model.predict_proba(X_scaled)[0, 1]  # P(UP)
        prediction = "UP" if prob >= 0.5 else "DOWN"

        # Confidence based on how far from 0.5
        # 0.5 = maximum uncertainty, 0 or 1 = maximum certainty
        distance = abs(prob - 0.5)
        if distance > 0.3:
            confidence = "high"
        elif distance > 0.15:
            confidence = "medium"
        else:
            confidence = "low"

        return GBClassificationResult(
            symbol=symbol,
            prediction=prediction,
            probability=prob,
            confidence=confidence,
            model_type=model_type,
        )

    def predict_with_both(
        self,
        symbol: str,
        features: dict[str, float],
    ) -> dict[str, Optional[GBClassificationResult]]:
        """
        Get predictions from both models for comparison.

        When both models agree, we can be more confident.
        When they disagree, it signals uncertainty.
        """
        return {
            "xgboost": self.predict(symbol, features, "xgboost"),
            "lightgbm": self.predict(symbol, features, "lightgbm"),
        }

    # =========================================================================
    # FEATURE IMPORTANCE (Interpretability)
    # =========================================================================

    def _get_feature_importance_internal(
        self,
        model,
        feature_names: list[str],
    ) -> dict[str, float]:
        """
        Extract feature importance from a trained model.

        WHAT IS FEATURE IMPORTANCE?
        It measures how much each feature contributes to predictions.

        For tree-based models, this is typically based on:
        - "gain": Total reduction in loss from splits using this feature
        - "weight": Number of times the feature is used in splits
        - "cover": Number of samples affected by splits on this feature

        Both XGBoost and LightGBM use "gain" by default, which is usually
        the most informative.
        """
        # Both XGBoost and LightGBM have feature_importances_ attribute
        importance = model.feature_importances_

        # Create dict and sort by importance
        importance_dict = dict(zip(feature_names, importance))
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

    def get_feature_importance(
        self,
        symbol: str,
        model_type: Literal["xgboost", "lightgbm"] = "xgboost",
    ) -> dict[str, float]:
        """
        Get feature importance for a trained model.

        USE THIS TO UNDERSTAND:
        - Which features drive predictions most?
        - Are conflict events more important than sentiment?
        - Should we add more features like this one?
        """
        if model_type == "xgboost":
            models = self.xgb_models
        else:
            models = self.lgb_models

        if symbol not in models:
            return {}

        return self._get_feature_importance_internal(models[symbol], self.feature_names)

    # =========================================================================
    # PRINTING / REPORTING
    # =========================================================================

    def _print_comparison(self, comparison: ModelComparison) -> None:
        """Print comparison results for one symbol."""
        print(f"\n{'='*60}")
        print(f"MODEL COMPARISON: {comparison.symbol}")
        print(f"{'='*60}")

        xgb = comparison.xgboost_metrics
        lgb = comparison.lightgbm_metrics

        if xgb and lgb:
            print(f"\n{'Metric':<20} {'XGBoost':>15} {'LightGBM':>15}")
            print("-" * 50)
            print(f"{'Accuracy':<20} {xgb.accuracy:>15.2%} {lgb.accuracy:>15.2%}")
            print(f"{'CV Accuracy':<20} {xgb.cv_accuracy:>15.2%} {lgb.cv_accuracy:>15.2%}")
            print(f"{'CV Std':<20} {xgb.cv_std:>15.2%} {lgb.cv_std:>15.2%}")
            print(f"{'Precision':<20} {xgb.precision:>15.2%} {lgb.precision:>15.2%}")
            print(f"{'Recall':<20} {xgb.recall:>15.2%} {lgb.recall:>15.2%}")
            print(f"{'F1 Score':<20} {xgb.f1_score:>15.2%} {lgb.f1_score:>15.2%}")
            print(f"{'ROC AUC':<20} {xgb.roc_auc:>15.3f} {lgb.roc_auc:>15.3f}")
            print(f"{'Samples':<20} {xgb.n_samples:>15} {lgb.n_samples:>15}")

            print(f"\nWinner: {comparison.winner.upper()}")

            # Print feature importance
            print(f"\nTop Features (XGBoost):")
            for i, (feat, imp) in enumerate(list(comparison.feature_importance_xgb.items())[:5]):
                print(f"  {i+1}. {feat}: {imp:.4f}")

    def _print_summary(self, results: dict[str, Optional[ModelComparison]]) -> None:
        """Print summary of all training results."""
        successful = [(s, c) for s, c in results.items() if c is not None]
        failed = [s for s, c in results.items() if c is None]

        print(f"\n{'='*60}")
        print("GRADIENT BOOSTING TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"Successfully trained: {len(successful)}/{len(results)}")

        if successful:
            # Count winners
            xgb_wins = sum(1 for _, c in successful if c.winner == "xgboost")
            lgb_wins = sum(1 for _, c in successful if c.winner == "lightgbm")
            ties = sum(1 for _, c in successful if c.winner == "tie")

            print(f"\nWinner breakdown:")
            print(f"  XGBoost: {xgb_wins}")
            print(f"  LightGBM: {lgb_wins}")
            print(f"  Tie: {ties}")

            # Average metrics
            avg_xgb_cv = np.mean([c.xgboost_metrics.cv_accuracy for _, c in successful])
            avg_lgb_cv = np.mean([c.lightgbm_metrics.cv_accuracy for _, c in successful])

            print(f"\nAverage CV Accuracy:")
            print(f"  XGBoost: {avg_xgb_cv:.2%}")
            print(f"  LightGBM: {avg_lgb_cv:.2%}")

            # Detailed results table
            print(f"\n{'Symbol':<12} {'XGB CV':>10} {'LGB CV':>10} {'Winner':>12}")
            print("-" * 46)
            for symbol, c in sorted(successful, key=lambda x: x[1].xgboost_metrics.cv_accuracy, reverse=True):
                print(
                    f"{symbol:<12} "
                    f"{c.xgboost_metrics.cv_accuracy:>10.2%} "
                    f"{c.lightgbm_metrics.cv_accuracy:>10.2%} "
                    f"{c.winner:>12}"
                )

        if failed:
            print(f"\nFailed symbols: {', '.join(failed)}")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """
    Example: Train gradient boosting classifiers and compare performance.

    Run this file directly to see how it works:
        python -m src.analysis.gradient_boost_classifier
    """
    from datetime import timedelta
    import logging

    logging.basicConfig(level=logging.INFO)

    # Training parameters
    end_date = date.today()
    start_date = end_date - timedelta(days=365)  # 1 year of data

    # Create classifier
    classifier = GradientBoostClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
    )

    # Train for a single symbol first (for quick testing)
    print("Training for SPY...")
    comparison = classifier.train_and_compare("SPY", start_date, end_date)

    if comparison:
        # Make a sample prediction
        sample_features = {
            "goldstein_mean": -2.5,
            "goldstein_min": -8.0,
            "goldstein_max": 1.0,
            "goldstein_std": 2.5,
            "mentions_total": 500,
            "mentions_max": 100,
            "avg_tone": -1.5,
            "conflict_count": 3,
            "cooperation_count": 1,
        }

        print("\nSample prediction (high conflict day):")
        pred_xgb = classifier.predict("SPY", sample_features, "xgboost")
        pred_lgb = classifier.predict("SPY", sample_features, "lightgbm")

        if pred_xgb:
            print(f"  XGBoost: {pred_xgb.prediction} ({pred_xgb.probability:.2%}, {pred_xgb.confidence})")
        if pred_lgb:
            print(f"  LightGBM: {pred_lgb.prediction} ({pred_lgb.probability:.2%}, {pred_lgb.confidence})")
