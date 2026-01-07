"""
Production Classification Module using scikit-learn.

This is the INDUSTRY-STANDARD way to do classification. It uses sklearn,
the most popular machine learning library in Python.

COMPARISON TO OUR LEARNING VERSION:
-----------------------------------
Learning Version (classification.py):
- 500+ lines of code
- Manual gradient descent
- Manual sigmoid function
- Manual normalization
- Educational, shows all the math

Production Version (this file):
- ~150 lines of code
- sklearn handles everything
- Battle-tested, optimized code
- What you'd actually use at work

KEY SKLEARN CLASSES USED:
-------------------------
- LogisticRegression: The classifier itself
- StandardScaler: Normalizes features (like our _normalize_features)
- train_test_split: Splits data for proper evaluation
- cross_val_score: K-fold cross-validation
- Various metrics: accuracy_score, f1_score, etc.

USAGE:
------
    from src.analysis.production_classifier import ProductionClassifier

    classifier = ProductionClassifier()
    results = classifier.train_all_markets(start_date, end_date)
    prediction = classifier.predict("CL=F", features)
"""

from dataclasses import dataclass
from datetime import date
from typing import Optional
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.config.constants import get_all_symbols
from src.db.connection import get_session
from src.db.queries import get_events_by_date_range, get_market_data

logger = logging.getLogger(__name__)


@dataclass
class ProdClassificationResult:
    """Container for a prediction result."""
    symbol: str
    prediction: str  # "UP" or "DOWN"
    probability: float
    confidence: str


@dataclass
class ProdModelMetrics:
    """Container for model evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    cv_accuracy: float  # Cross-validation accuracy
    cv_std: float  # Standard deviation of CV scores
    n_samples: int


class ProductionClassifier:
    """
    Production-ready classifier using scikit-learn.

    This does the same thing as MarketClassifier but uses
    industry-standard libraries instead of manual implementations.
    """

    def __init__(self):
        """Initialize with sklearn components."""
        # The scaler normalizes features (like our manual normalization)
        self.scaler = StandardScaler()

        # Store trained models for each symbol
        self.models: dict[str, LogisticRegression] = {}
        self.scalers: dict[str, StandardScaler] = {}
        self.feature_names: list[str] = []

    def _prepare_features(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Prepare feature matrix and target vector.
        (Same logic as learning version, just cleaner)
        """
        # Get market data
        with get_session() as session:
            market_data = get_market_data(session, symbol, start_date, end_date)
            if not market_data:
                return np.array([]), np.array([]), []

            market_df = pd.DataFrame([
                {"date": m.date, "log_return": m.log_return}
                for m in market_data
            ])

        # Get events
        with get_session() as session:
            events = get_events_by_date_range(session, start_date, end_date)
            if not events:
                return np.array([]), np.array([]), []

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

        # Aggregate events by date
        event_agg = events_df.groupby("date").agg({
            "goldstein_scale": ["mean", "min", "max"],
            "num_mentions": "sum",
            "avg_tone": "mean",
            "is_conflict": "sum",
            "is_cooperation": "sum",
        }).reset_index()

        # Flatten column names
        event_agg.columns = [
            "date", "goldstein_mean", "goldstein_min", "goldstein_max",
            "mentions_total", "avg_tone", "conflict_count", "cooperation_count"
        ]

        # Merge
        merged = pd.merge(market_df, event_agg, on="date", how="left").fillna(0)
        merged = merged.dropna(subset=["log_return"])

        if len(merged) < 20:
            return np.array([]), np.array([]), []

        # Target: 1 = UP, 0 = DOWN
        y = (merged["log_return"] > 0).astype(int).values

        # Features
        feature_cols = [
            "goldstein_mean", "goldstein_min", "goldstein_max",
            "mentions_total", "avg_tone", "conflict_count", "cooperation_count"
        ]
        X = merged[feature_cols].values

        return X, y, feature_cols

    def train(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> Optional[ProdModelMetrics]:
        """
        Train a classifier for one symbol.

        This is where sklearn shines - training is just model.fit(X, y)!
        """
        logger.info(f"Training production classifier for {symbol}")

        # Prepare data
        X, y, feature_names = self._prepare_features(symbol, start_date, end_date)

        if len(X) == 0:
            logger.warning(f"No data for {symbol}")
            return None

        self.feature_names = feature_names

        # Create fresh scaler and model for this symbol
        scaler = StandardScaler()
        model = LogisticRegression(
            max_iter=1000,      # Max iterations (like our max_iterations)
            C=1.0,              # Inverse of regularization strength
            random_state=42,    # For reproducibility
        )

        # Scale features - sklearn's StandardScaler does what our _normalize_features did
        X_scaled = scaler.fit_transform(X)

        # CROSS-VALIDATION: More robust than single train/test split
        # This trains 5 models on different data splits and averages performance
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')

        # Train final model on all data
        model.fit(X_scaled, y)

        # Store for predictions
        self.models[symbol] = model
        self.scalers[symbol] = scaler

        # Evaluate on training data (for comparison with learning version)
        y_pred = model.predict(X_scaled)

        metrics = ProdModelMetrics(
            accuracy=accuracy_score(y, y_pred),
            precision=precision_score(y, y_pred, zero_division=0),
            recall=recall_score(y, y_pred, zero_division=0),
            f1_score=f1_score(y, y_pred, zero_division=0),
            cv_accuracy=cv_scores.mean(),
            cv_std=cv_scores.std(),
            n_samples=len(y),
        )

        logger.info(
            f"{symbol}: Accuracy={metrics.accuracy:.2%}, "
            f"CV={metrics.cv_accuracy:.2%} (+/- {metrics.cv_std*2:.2%})"
        )

        return metrics

    def train_all_markets(
        self,
        start_date: date,
        end_date: date,
        symbols: list[str] = None,
    ) -> dict[str, Optional[ProdModelMetrics]]:
        """Train classifiers for all markets."""
        if symbols is None:
            symbols = get_all_symbols()

        results = {}
        for symbol in symbols:
            try:
                metrics = self.train(symbol, start_date, end_date)
                results[symbol] = metrics
            except Exception as e:
                logger.error(f"Error training {symbol}: {e}")
                results[symbol] = None

        # Print summary
        self._print_summary(results)
        return results

    def predict(
        self,
        symbol: str,
        features: dict[str, float],
    ) -> Optional[ProdClassificationResult]:
        """Make a prediction for a symbol."""
        if symbol not in self.models:
            logger.warning(f"No model for {symbol}")
            return None

        model = self.models[symbol]
        scaler = self.scalers[symbol]

        # Build feature vector
        X = np.array([[features.get(name, 0) for name in self.feature_names]])

        # Scale using stored scaler
        X_scaled = scaler.transform(X)

        # Predict
        prob = model.predict_proba(X_scaled)[0, 1]  # Probability of class 1 (UP)
        prediction = "UP" if prob >= 0.5 else "DOWN"

        # Confidence
        distance = abs(prob - 0.5)
        confidence = "high" if distance > 0.3 else "medium" if distance > 0.15 else "low"

        return ProdClassificationResult(
            symbol=symbol,
            prediction=prediction,
            probability=prob,
            confidence=confidence,
        )

    def get_feature_importance(self, symbol: str) -> dict[str, float]:
        """Get feature importance (coefficient magnitudes)."""
        if symbol not in self.models:
            return {}

        model = self.models[symbol]
        importance = dict(zip(self.feature_names, np.abs(model.coef_[0])))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def get_detailed_report(self, symbol: str, X: np.ndarray, y: np.ndarray) -> str:
        """Get sklearn's classification report."""
        if symbol not in self.models:
            return "No model trained"

        model = self.models[symbol]
        scaler = self.scalers[symbol]

        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)

        return classification_report(y, y_pred, target_names=['DOWN', 'UP'])

    def _print_summary(self, results: dict[str, Optional[ProdModelMetrics]]) -> None:
        """Print training summary."""
        successful = [(s, m) for s, m in results.items() if m is not None]
        failed = [s for s, m in results.items() if m is None]

        print(f"\n{'='*60}")
        print("PRODUCTION CLASSIFIER TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"Successfully trained: {len(successful)}/{len(results)}")

        if successful:
            avg_acc = np.mean([m.accuracy for _, m in successful])
            avg_cv = np.mean([m.cv_accuracy for _, m in successful])
            print(f"Average Accuracy: {avg_acc:.2%}")
            print(f"Average CV Accuracy: {avg_cv:.2%}")

            print(f"\n{'Symbol':<12} {'Accuracy':>10} {'CV Acc':>10} {'F1':>10}")
            print("-" * 45)
            for symbol, m in sorted(successful, key=lambda x: x[1].cv_accuracy, reverse=True):
                print(f"{symbol:<12} {m.accuracy:>10.2%} {m.cv_accuracy:>10.2%} {m.f1_score:>10.2%}")
