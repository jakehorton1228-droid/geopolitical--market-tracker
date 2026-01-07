"""
Classification Module for Market Direction Prediction.

WHAT IS CLASSIFICATION?
-----------------------
Classification predicts CATEGORIES, not numbers:
- Will the market go UP or DOWN?
- Is this event significant or not?
- Will there be high volatility or low?

This is different from regression, which predicts continuous values.

WHY LOGISTIC REGRESSION?
------------------------
1. **Interpretable**: Each coefficient tells you exactly how much a feature
   influences the prediction. "A 1-point increase in Goldstein scale
   increases the odds of an UP move by X%."

2. **Probabilistic**: It doesn't just say "UP" or "DOWN" - it gives you
   a probability (e.g., 73% chance of UP). This lets you set thresholds.

3. **Works with limited data**: Unlike neural networks, logistic regression
   doesn't need millions of samples to train.

4. **Industry standard**: It's used extensively in finance because
   regulators often require explainable models.

THE MATH (Simplified):
---------------------
1. We have features X = [goldstein, mentions, conflicts, ...]
2. We learn weights W = [w1, w2, w3, ...]
3. We compute: z = w0 + w1*x1 + w2*x2 + ...
4. We apply sigmoid: probability = 1 / (1 + e^(-z))
5. If probability > 0.5, predict UP; else predict DOWN

TRAINING:
---------
The model learns weights by finding values that maximize the likelihood
of the training data. This is done through gradient descent:
1. Start with random weights
2. Make predictions
3. Calculate error (how wrong were we?)
4. Adjust weights in the direction that reduces error
5. Repeat until weights stop changing

USAGE:
------
    from src.analysis.classification import MarketClassifier

    classifier = MarketClassifier()

    # Train on historical data
    classifier.train(start_date, end_date)

    # Predict for a specific date and symbol
    prediction = classifier.predict("CL=F", features)

    # Get predictions for all markets
    predictions = classifier.predict_all_markets(features)
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional
import logging

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit  # This is the sigmoid function

from src.config.constants import get_all_symbols, get_event_group
from src.db.connection import get_session
from src.db.models import Event, MarketData
from src.db.queries import get_events_by_date_range, get_market_data

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Container for a single prediction result."""

    symbol: str
    prediction: str  # "UP" or "DOWN"
    probability: float  # Probability of UP (0 to 1)
    confidence: str  # "high", "medium", "low"

    # Feature contributions (which features drove this prediction?)
    feature_contributions: dict[str, float] = field(default_factory=dict)


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""

    accuracy: float  # Correct predictions / Total predictions
    precision: float  # True positives / (True positives + False positives)
    recall: float  # True positives / (True positives + False negatives)
    f1_score: float  # Harmonic mean of precision and recall

    # Confusion matrix
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int

    # Per-class metrics
    up_accuracy: float
    down_accuracy: float

    # Sample info
    n_samples: int
    n_up: int
    n_down: int


class MarketClassifier:
    """
    Logistic Regression classifier for market direction prediction.

    This classifier predicts whether a market will go UP or DOWN based
    on geopolitical event features.

    KEY CONCEPTS:
    -------------
    - Features (X): The inputs we use to make predictions
      Examples: goldstein_scale, num_mentions, is_conflict

    - Target (y): What we're trying to predict
      In our case: 1 = UP (positive return), 0 = DOWN (negative return)

    - Coefficients (weights): What the model learns
      Each feature gets a weight that determines its importance

    - Sigmoid function: Converts raw scores to probabilities
      sigmoid(z) = 1 / (1 + e^(-z))

    - Training: Finding the best weights using gradient descent
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        regularization: float = 0.01,
        convergence_threshold: float = 1e-6,
    ):
        """
        Initialize the classifier.

        Args:
            learning_rate: How big of steps to take during gradient descent.
                          Too high = overshoot, too low = slow training.
            max_iterations: Maximum training iterations.
            regularization: L2 regularization strength to prevent overfitting.
                           Higher = simpler model, lower = more complex.
            convergence_threshold: Stop when weight changes are below this.
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.regularization = regularization
        self.convergence_threshold = convergence_threshold

        # These get set during training
        self.coefficients: Optional[np.ndarray] = None
        self.feature_names: list[str] = []
        self.is_trained: bool = False

        # Training history (for debugging/visualization)
        self.training_history: list[dict] = []

        # Separate models for each symbol (markets behave differently)
        self.symbol_models: dict[str, np.ndarray] = {}

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Apply the sigmoid function.

        THE SIGMOID FUNCTION:
        --------------------
        sigmoid(z) = 1 / (1 + e^(-z))

        This transforms any real number into a value between 0 and 1,
        which we interpret as a probability.

        Examples:
        - z = 0 → sigmoid = 0.5 (50% chance)
        - z = 2 → sigmoid ≈ 0.88 (88% chance)
        - z = -2 → sigmoid ≈ 0.12 (12% chance)
        - z = 10 → sigmoid ≈ 1.0 (nearly certain)
        - z = -10 → sigmoid ≈ 0.0 (nearly impossible)

        Args:
            z: Raw scores (can be any real number)

        Returns:
            Probabilities between 0 and 1
        """
        # Using scipy's expit for numerical stability
        # (handles very large/small numbers without overflow)
        return expit(z)

    def _prepare_features(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Prepare feature matrix X and target vector y from database.

        This is where we transform raw data into the format needed for
        machine learning:

        1. Fetch market data (we need returns to know UP/DOWN)
        2. Fetch events (our features)
        3. Merge them by date
        4. Create feature columns
        5. Create target column (1 = UP, 0 = DOWN)

        Args:
            symbol: Ticker symbol
            start_date: Start of training period
            end_date: End of training period

        Returns:
            Tuple of (X features, y target, feature_names)
        """
        # Step 1: Get market data
        with get_session() as session:
            market_data = get_market_data(session, symbol, start_date, end_date)

            if not market_data:
                return np.array([]), np.array([]), []

            market_df = pd.DataFrame([
                {
                    "date": m.date,
                    "log_return": m.log_return,
                }
                for m in market_data
            ])

        # Step 2: Get events
        with get_session() as session:
            events = get_events_by_date_range(session, start_date, end_date)

            if not events:
                return np.array([]), np.array([]), []

            events_df = pd.DataFrame([
                {
                    "date": e.event_date,
                    "goldstein_scale": e.goldstein_scale or 0,
                    "num_mentions": e.num_mentions or 0,
                    "num_articles": e.num_articles or 0,
                    "avg_tone": e.avg_tone or 0,
                    "event_root_code": e.event_root_code,
                }
                for e in events
            ])

        # Step 3: Create feature columns from events
        # We aggregate multiple events per day into summary statistics

        # Event type indicators (one-hot encoding for event groups)
        events_df["is_conflict"] = events_df["event_root_code"].isin(["18", "19", "20"]).astype(int)
        events_df["is_cooperation"] = events_df["event_root_code"].isin(["03", "04", "05", "06"]).astype(int)
        events_df["is_diplomatic"] = events_df["event_root_code"].isin(["01", "02"]).astype(int)
        events_df["is_material"] = events_df["event_root_code"].isin(["07", "08", "09"]).astype(int)

        # Aggregate events by date
        event_agg = events_df.groupby("date").agg({
            # Goldstein scale statistics
            "goldstein_scale": ["mean", "min", "max", "std"],
            # Media coverage
            "num_mentions": ["sum", "max"],
            "num_articles": "sum",
            # Tone
            "avg_tone": "mean",
            # Event type counts
            "is_conflict": "sum",
            "is_cooperation": "sum",
            "is_diplomatic": "sum",
            "is_material": "sum",
        }).reset_index()

        # Flatten column names
        event_agg.columns = [
            "date",
            "goldstein_mean", "goldstein_min", "goldstein_max", "goldstein_std",
            "mentions_total", "mentions_max",
            "articles_total",
            "avg_tone",
            "conflict_count", "cooperation_count", "diplomatic_count", "material_count",
        ]

        # Fill NaN in std (happens when only one event per day)
        event_agg["goldstein_std"] = event_agg["goldstein_std"].fillna(0)

        # Step 4: Merge market and event data
        merged = pd.merge(market_df, event_agg, on="date", how="left")

        # Fill NaN for days with no events (use neutral values)
        fill_values = {
            "goldstein_mean": 0,
            "goldstein_min": 0,
            "goldstein_max": 0,
            "goldstein_std": 0,
            "mentions_total": 0,
            "mentions_max": 0,
            "articles_total": 0,
            "avg_tone": 0,
            "conflict_count": 0,
            "cooperation_count": 0,
            "diplomatic_count": 0,
            "material_count": 0,
        }
        merged = merged.fillna(fill_values)

        # Remove rows with missing returns
        merged = merged.dropna(subset=["log_return"])

        if len(merged) < 10:
            logger.warning(f"Insufficient data for {symbol}: {len(merged)} rows")
            return np.array([]), np.array([]), []

        # Step 5: Create target variable
        # 1 = UP (positive return), 0 = DOWN (negative/zero return)
        y = (merged["log_return"] > 0).astype(int).values

        # Step 6: Create feature matrix
        feature_columns = [
            "goldstein_mean",
            "goldstein_min",  # Worst event of the day
            "goldstein_max",  # Best event of the day
            "mentions_total",
            "avg_tone",
            "conflict_count",
            "cooperation_count",
        ]

        X = merged[feature_columns].values

        # Step 7: Normalize features (important for gradient descent!)
        # This makes all features have similar scales
        X = self._normalize_features(X)

        # Step 8: Add intercept column (column of 1s)
        intercept = np.ones((X.shape[0], 1))
        X = np.hstack([intercept, X])

        feature_names = ["intercept"] + feature_columns

        return X, y, feature_names

    def _normalize_features(self, X: np.ndarray) -> np.ndarray:
        """
        Normalize features to have zero mean and unit variance.

        WHY NORMALIZE?
        -------------
        Different features have different scales:
        - goldstein_scale: -10 to +10
        - num_mentions: 0 to 10,000+
        - conflict_count: 0 to 50

        Without normalization, features with larger scales would dominate
        the learning process. Normalization puts all features on equal footing.

        STANDARDIZATION FORMULA:
        -----------------------
        z = (x - mean) / std

        After this:
        - Mean of each feature = 0
        - Standard deviation = 1

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Normalized feature matrix
        """
        # Calculate mean and std for each column
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)

        # Avoid division by zero
        stds = np.where(stds == 0, 1, stds)

        # Store for later use (when making predictions on new data)
        self._feature_means = means
        self._feature_stds = stds

        return (X - means) / stds

    def _compute_loss(
        self,
        X: np.ndarray,
        y: np.ndarray,
        coefficients: np.ndarray,
    ) -> float:
        """
        Compute the logistic loss (cross-entropy loss).

        THE LOSS FUNCTION:
        -----------------
        This measures how wrong our predictions are. Lower = better.

        For each sample:
        - If y=1 (UP): loss = -log(p)      # Penalize low probability for UP
        - If y=0 (DOWN): loss = -log(1-p)  # Penalize high probability for UP

        Combined: loss = -[y*log(p) + (1-y)*log(1-p)]

        Total loss = average over all samples + regularization term

        INTUITION:
        ----------
        - If we predict 90% UP and it actually went UP: low loss
        - If we predict 90% UP and it actually went DOWN: high loss
        - The log makes mistakes more costly as confidence increases

        Args:
            X: Feature matrix
            y: True labels (0 or 1)
            coefficients: Current model weights

        Returns:
            Loss value (lower is better)
        """
        n = len(y)

        # Compute predictions
        z = X @ coefficients
        probabilities = self._sigmoid(z)

        # Clip probabilities to avoid log(0)
        eps = 1e-15
        probabilities = np.clip(probabilities, eps, 1 - eps)

        # Cross-entropy loss
        loss = -np.mean(y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities))

        # Add L2 regularization (don't regularize intercept)
        reg_term = (self.regularization / (2 * n)) * np.sum(coefficients[1:] ** 2)

        return loss + reg_term

    def _compute_gradient(
        self,
        X: np.ndarray,
        y: np.ndarray,
        coefficients: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the gradient of the loss function.

        THE GRADIENT:
        ------------
        The gradient tells us which direction to move the coefficients
        to reduce the loss. It's like a compass pointing "downhill."

        For logistic regression, the gradient is beautifully simple:
        gradient = X^T @ (predictions - actual) / n

        This is the same form as linear regression! The sigmoid derivative
        cancels out in a very elegant way.

        INTUITION:
        ----------
        For each feature:
        - If prediction > actual: gradient is positive → decrease coefficient
        - If prediction < actual: gradient is negative → increase coefficient
        - The magnitude tells us how big of an adjustment to make

        Args:
            X: Feature matrix
            y: True labels
            coefficients: Current weights

        Returns:
            Gradient vector (same shape as coefficients)
        """
        n = len(y)

        # Compute predictions
        z = X @ coefficients
        probabilities = self._sigmoid(z)

        # Gradient: X^T @ (p - y) / n
        error = probabilities - y
        gradient = X.T @ error / n

        # Add regularization gradient (don't regularize intercept)
        reg_gradient = np.zeros_like(coefficients)
        reg_gradient[1:] = (self.regularization / n) * coefficients[1:]

        return gradient + reg_gradient

    def train(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        verbose: bool = True,
    ) -> Optional[ModelMetrics]:
        """
        Train the classifier on historical data for a specific symbol.

        TRAINING PROCESS:
        ----------------
        1. Prepare features and targets from database
        2. Initialize coefficients to zeros
        3. Iteratively update coefficients using gradient descent:
           - Compute gradient (direction of steepest increase in loss)
           - Move coefficients in opposite direction (to decrease loss)
           - Check for convergence
        4. Evaluate model on training data

        Args:
            symbol: Ticker symbol to train on
            start_date: Start of training period
            end_date: End of training period
            verbose: Whether to print training progress

        Returns:
            ModelMetrics with training performance, or None if training failed
        """
        logger.info(f"Training classifier for {symbol}")

        # Prepare data
        X, y, feature_names = self._prepare_features(symbol, start_date, end_date)

        if len(X) == 0:
            logger.warning(f"No training data for {symbol}")
            return None

        self.feature_names = feature_names
        n_features = X.shape[1]

        # Initialize coefficients to zeros
        coefficients = np.zeros(n_features)

        # Training loop (gradient descent)
        self.training_history = []

        for iteration in range(self.max_iterations):
            # Compute loss (for tracking progress)
            loss = self._compute_loss(X, y, coefficients)

            # Compute gradient
            gradient = self._compute_gradient(X, y, coefficients)

            # Update coefficients (move in opposite direction of gradient)
            new_coefficients = coefficients - self.learning_rate * gradient

            # Check for convergence
            change = np.max(np.abs(new_coefficients - coefficients))

            # Store history
            self.training_history.append({
                "iteration": iteration,
                "loss": loss,
                "max_gradient": np.max(np.abs(gradient)),
                "coefficient_change": change,
            })

            # Update coefficients
            coefficients = new_coefficients

            # Check convergence
            if change < self.convergence_threshold:
                if verbose:
                    logger.info(f"Converged at iteration {iteration}")
                break

            # Print progress every 100 iterations
            if verbose and iteration % 100 == 0:
                logger.info(f"Iteration {iteration}: loss = {loss:.6f}")

        # Store trained coefficients
        self.coefficients = coefficients
        self.symbol_models[symbol] = coefficients.copy()
        self.is_trained = True

        # Evaluate on training data
        metrics = self._evaluate(X, y)

        if verbose:
            self._print_training_summary(symbol, metrics)

        return metrics

    def train_all_markets(
        self,
        start_date: date,
        end_date: date,
        symbols: list[str] = None,
    ) -> dict[str, Optional[ModelMetrics]]:
        """
        Train separate classifiers for all markets.

        WHY SEPARATE MODELS?
        -------------------
        Different markets respond differently to events:
        - Oil (CL=F) is very sensitive to conflict events
        - Gold (GC=F) responds to uncertainty/fear
        - VIX goes up when markets are stressed
        - Currencies have their own dynamics

        Training separate models captures these differences.

        Args:
            start_date: Start of training period
            end_date: End of training period
            symbols: Symbols to train on (default: all symbols)

        Returns:
            Dictionary mapping symbol to ModelMetrics
        """
        if symbols is None:
            symbols = get_all_symbols()

        results = {}

        for symbol in symbols:
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"Training model for {symbol}")
                logger.info(f"{'='*50}")

                metrics = self.train(symbol, start_date, end_date, verbose=False)
                results[symbol] = metrics

                if metrics:
                    logger.info(f"{symbol}: Accuracy = {metrics.accuracy:.2%}")
                else:
                    logger.warning(f"{symbol}: Training failed")

            except Exception as e:
                logger.error(f"Error training {symbol}: {e}")
                results[symbol] = None

        # Print summary
        self._print_overall_summary(results)

        return results

    def _evaluate(self, X: np.ndarray, y: np.ndarray) -> ModelMetrics:
        """
        Evaluate model performance on a dataset.

        METRICS EXPLAINED:
        -----------------
        - Accuracy: What fraction of predictions were correct?
          accuracy = (TP + TN) / (TP + TN + FP + FN)

        - Precision: When we predicted UP, how often were we right?
          precision = TP / (TP + FP)
          "Don't cry wolf"

        - Recall: Of all actual UPs, how many did we catch?
          recall = TP / (TP + FN)
          "Don't miss the wolves"

        - F1 Score: Harmonic mean of precision and recall
          f1 = 2 * (precision * recall) / (precision + recall)
          Balances both concerns

        CONFUSION MATRIX:
        ----------------
                          Predicted
                        UP      DOWN
        Actual  UP     TP       FN
               DOWN    FP       TN

        Args:
            X: Feature matrix
            y: True labels

        Returns:
            ModelMetrics with all evaluation metrics
        """
        # Make predictions
        z = X @ self.coefficients
        probabilities = self._sigmoid(z)
        predictions = (probabilities >= 0.5).astype(int)

        # Confusion matrix components
        true_positives = np.sum((predictions == 1) & (y == 1))
        true_negatives = np.sum((predictions == 0) & (y == 0))
        false_positives = np.sum((predictions == 1) & (y == 0))
        false_negatives = np.sum((predictions == 0) & (y == 1))

        # Basic metrics
        n = len(y)
        accuracy = (true_positives + true_negatives) / n

        # Precision (handle division by zero)
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

        # Recall
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

        # F1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Per-class accuracy
        n_up = np.sum(y == 1)
        n_down = np.sum(y == 0)
        up_accuracy = true_positives / n_up if n_up > 0 else 0
        down_accuracy = true_negatives / n_down if n_down > 0 else 0

        return ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            true_positives=int(true_positives),
            true_negatives=int(true_negatives),
            false_positives=int(false_positives),
            false_negatives=int(false_negatives),
            up_accuracy=up_accuracy,
            down_accuracy=down_accuracy,
            n_samples=n,
            n_up=int(n_up),
            n_down=int(n_down),
        )

    def predict(
        self,
        symbol: str,
        features: dict[str, float],
    ) -> Optional[ClassificationResult]:
        """
        Make a prediction for a specific symbol given features.

        Args:
            symbol: Ticker symbol
            features: Dictionary of feature values

        Returns:
            ClassificationResult with prediction and probability
        """
        if symbol not in self.symbol_models:
            logger.warning(f"No trained model for {symbol}")
            return None

        coefficients = self.symbol_models[symbol]

        # Build feature vector (same order as training)
        feature_values = [1.0]  # Intercept
        for name in self.feature_names[1:]:  # Skip intercept
            feature_values.append(features.get(name, 0.0))

        X = np.array(feature_values)

        # Normalize using stored parameters
        # (Skip intercept which is at index 0)
        if hasattr(self, '_feature_means'):
            X[1:] = (X[1:] - self._feature_means) / self._feature_stds

        # Compute probability
        z = X @ coefficients
        probability = self._sigmoid(z)

        # Make prediction
        prediction = "UP" if probability >= 0.5 else "DOWN"

        # Determine confidence
        distance_from_05 = abs(probability - 0.5)
        if distance_from_05 > 0.3:
            confidence = "high"
        elif distance_from_05 > 0.15:
            confidence = "medium"
        else:
            confidence = "low"

        # Calculate feature contributions
        contributions = {}
        for i, name in enumerate(self.feature_names):
            contributions[name] = float(X[i] * coefficients[i])

        return ClassificationResult(
            symbol=symbol,
            prediction=prediction,
            probability=float(probability),
            confidence=confidence,
            feature_contributions=contributions,
        )

    def get_feature_importance(self, symbol: str = None) -> dict[str, float]:
        """
        Get the importance of each feature (absolute coefficient values).

        INTERPRETING COEFFICIENTS:
        -------------------------
        - Positive coefficient: Feature increases probability of UP
        - Negative coefficient: Feature decreases probability of UP
        - Larger absolute value: Feature has more influence

        Note: These are normalized coefficients. To interpret in original
        units, you'd need to adjust for the normalization.

        Args:
            symbol: Specific symbol to get importance for (default: last trained)

        Returns:
            Dictionary mapping feature name to importance (absolute coefficient)
        """
        if symbol and symbol in self.symbol_models:
            coefficients = self.symbol_models[symbol]
        elif self.coefficients is not None:
            coefficients = self.coefficients
        else:
            return {}

        importance = {}
        for name, coef in zip(self.feature_names, coefficients):
            importance[name] = abs(float(coef))

        # Sort by importance
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def _print_training_summary(self, symbol: str, metrics: ModelMetrics) -> None:
        """Print a summary of training results."""
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE: {symbol}")
        print(f"{'='*60}")
        print(f"\nDATA SUMMARY:")
        print(f"  Total samples: {metrics.n_samples}")
        print(f"  UP days: {metrics.n_up} ({metrics.n_up/metrics.n_samples:.1%})")
        print(f"  DOWN days: {metrics.n_down} ({metrics.n_down/metrics.n_samples:.1%})")

        print(f"\nMODEL PERFORMANCE:")
        print(f"  Accuracy: {metrics.accuracy:.2%}")
        print(f"  Precision: {metrics.precision:.2%}")
        print(f"  Recall: {metrics.recall:.2%}")
        print(f"  F1 Score: {metrics.f1_score:.2%}")

        print(f"\nPER-CLASS ACCURACY:")
        print(f"  UP days correctly predicted: {metrics.up_accuracy:.2%}")
        print(f"  DOWN days correctly predicted: {metrics.down_accuracy:.2%}")

        print(f"\nCONFUSION MATRIX:")
        print(f"                  Predicted")
        print(f"                 UP    DOWN")
        print(f"  Actual UP    {metrics.true_positives:4d}    {metrics.false_negatives:4d}")
        print(f"        DOWN   {metrics.false_positives:4d}    {metrics.true_negatives:4d}")

        print(f"\nFEATURE IMPORTANCE (Top 5):")
        importance = self.get_feature_importance(symbol)
        for i, (name, imp) in enumerate(list(importance.items())[:5]):
            coef = self.symbol_models[symbol][self.feature_names.index(name)]
            direction = "↑" if coef > 0 else "↓"
            print(f"  {i+1}. {name}: {imp:.4f} ({direction} probability of UP)")

    def _print_overall_summary(self, results: dict[str, Optional[ModelMetrics]]) -> None:
        """Print summary of all trained models."""
        print(f"\n{'='*60}")
        print("OVERALL TRAINING SUMMARY")
        print(f"{'='*60}")

        successful = [(s, m) for s, m in results.items() if m is not None]
        failed = [s for s, m in results.items() if m is None]

        print(f"\nSuccessfully trained: {len(successful)}/{len(results)} models")

        if failed:
            print(f"Failed to train: {', '.join(failed)}")

        if successful:
            # Average metrics
            avg_accuracy = np.mean([m.accuracy for _, m in successful])
            avg_f1 = np.mean([m.f1_score for _, m in successful])

            print(f"\nAVERAGE METRICS:")
            print(f"  Accuracy: {avg_accuracy:.2%}")
            print(f"  F1 Score: {avg_f1:.2%}")

            # Best and worst
            best = max(successful, key=lambda x: x[1].accuracy)
            worst = min(successful, key=lambda x: x[1].accuracy)

            print(f"\nBEST MODEL: {best[0]} ({best[1].accuracy:.2%} accuracy)")
            print(f"WORST MODEL: {worst[0]} ({worst[1].accuracy:.2%} accuracy)")

            # Detailed table
            print(f"\nDETAILED RESULTS:")
            print(f"{'Symbol':<12} {'Accuracy':>10} {'F1':>10} {'Samples':>10}")
            print("-" * 45)
            for symbol, metrics in sorted(successful, key=lambda x: x[1].accuracy, reverse=True):
                print(f"{symbol:<12} {metrics.accuracy:>10.2%} {metrics.f1_score:>10.2%} {metrics.n_samples:>10}")


def explain_classification(result: ClassificationResult) -> str:
    """
    Generate a human-readable explanation of a classification result.

    Args:
        result: ClassificationResult to explain

    Returns:
        Explanation string
    """
    direction_emoji = "📈" if result.prediction == "UP" else "📉"

    lines = [
        f"MARKET PREDICTION: {result.symbol}",
        "=" * 50,
        "",
        f"Prediction: {direction_emoji} {result.prediction}",
        f"Probability of UP: {result.probability:.1%}",
        f"Confidence: {result.confidence.upper()}",
        "",
        "WHAT INFLUENCED THIS PREDICTION:",
        "-" * 50,
    ]

    # Sort contributions by absolute value
    sorted_contributions = sorted(
        result.feature_contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    for name, contribution in sorted_contributions[:5]:
        if name == "intercept":
            continue
        direction = "↑" if contribution > 0 else "↓"
        effect = "pushes toward UP" if contribution > 0 else "pushes toward DOWN"
        lines.append(f"  {name}: {direction} {abs(contribution):.4f} ({effect})")

    lines.extend([
        "",
        "INTERPRETATION:",
    ])

    if result.confidence == "high":
        lines.append(f"  The model is highly confident in this {result.prediction} prediction.")
    elif result.confidence == "medium":
        lines.append(f"  The model has moderate confidence in this {result.prediction} prediction.")
    else:
        lines.append(f"  The model has low confidence. The market direction is uncertain.")

    return "\n".join(lines)
