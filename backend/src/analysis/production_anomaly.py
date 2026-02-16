"""
Production Anomaly Detection Module using scikit-learn.

This is the INDUSTRY-STANDARD way to do anomaly detection. It uses sklearn's
IsolationForest and statistical methods for robust anomaly detection.

COMPARISON TO LEARNING VERSION:
-------------------------------
Learning Version (anomaly_detection.py):
- Manual Z-score calculation
- Simple threshold-based detection
- Rule-based unexplained move detection
- Educational, shows the concepts

Production Version (this file):
- sklearn's IsolationForest (unsupervised ML)
- Multiple detection methods combined
- More robust to outliers and edge cases
- What you'd use in production

KEY SKLEARN CLASSES USED:
-------------------------
- IsolationForest: Detects anomalies by isolating observations
- StandardScaler: Normalizes features for consistent detection
- LocalOutlierFactor: Alternative density-based detection

HOW ISOLATION FOREST WORKS:
---------------------------
1. Randomly select a feature and split point
2. Recursively partition the data
3. Anomalies are isolated in fewer splits (shorter path)
4. Normal points require more splits (longer path)

Intuition: Anomalies are "few and different" - they're easier to isolate.

USAGE:
------
    from src.analysis.production_anomaly import ProductionAnomalyDetector

    detector = ProductionAnomalyDetector()
    anomalies = detector.detect_all(symbol, start_date, end_date)

    # Get detailed report
    report = detector.get_anomaly_report(anomalies)
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats

from src.config.settings import ANOMALY_ZSCORE_THRESHOLD
from src.config.constants import COUNTRY_ASSET_MAP, EVENT_ASSET_MAP, get_event_group
from src.db.connection import get_session
from src.db.queries import get_market_data, get_events_by_date_range

logger = logging.getLogger(__name__)


@dataclass
class ProdAnomaly:
    """Container for a detected anomaly."""
    date: date
    symbol: str
    anomaly_type: str  # "unexplained_move", "muted_response", "statistical_outlier"

    # Market data
    actual_return: float
    expected_return: float
    z_score: float

    # ML detection scores
    isolation_score: float  # -1 = anomaly, 1 = normal (sklearn convention)
    anomaly_probability: float  # 0-1 probability of being anomalous

    # Event context (if applicable)
    event_count: int = 0
    avg_goldstein: float = 0.0

    # Detection method
    detected_by: list[str] = None  # Which methods flagged this

    def __post_init__(self):
        if self.detected_by is None:
            self.detected_by = []


@dataclass
class AnomalyReport:
    """Summary report of anomaly detection results."""
    symbol: str
    start_date: date
    end_date: date

    total_days: int
    anomaly_count: int
    anomaly_rate: float

    # Breakdown by type
    unexplained_moves: int
    muted_responses: int
    statistical_outliers: int

    # Top anomalies
    top_anomalies: list[ProdAnomaly]

    # Summary text
    summary: str


class ProductionAnomalyDetector:
    """
    Production-ready anomaly detector using scikit-learn.

    Combines multiple detection methods:
    1. Isolation Forest (unsupervised ML)
    2. Z-score analysis (statistical)
    3. Event-return mismatch (domain knowledge)

    An observation is flagged as anomalous if ANY method detects it,
    with the anomaly_probability reflecting confidence.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        zscore_threshold: float = None,
        lookback_days: int = 30,
    ):
        """
        Initialize the anomaly detector.

        Args:
            contamination: Expected proportion of anomalies (0.05 = 5%)
                          This is a key parameter for IsolationForest.
            zscore_threshold: Threshold for Z-score based detection
            lookback_days: Days to use for rolling statistics
        """
        self.contamination = contamination
        self.zscore_threshold = zscore_threshold or ANOMALY_ZSCORE_THRESHOLD
        self.lookback_days = lookback_days

        # Initialize sklearn models
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
        )
        self.scaler = StandardScaler()

    def _prepare_features(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Prepare feature matrix for anomaly detection.

        Delegates to shared FeatureEngineering module for consistent
        feature preparation. Returns a full DataFrame with rolling
        statistics, market features, and event features.
        """
        from src.analysis.feature_engineering import FeatureEngineering

        fe = FeatureEngineering()
        return fe.prepare_anomaly_features(
            symbol, start_date, end_date,
            lookback_days=self.lookback_days,
        )

    def detect_with_isolation_forest(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Detect anomalies using Isolation Forest.

        IsolationForest is an unsupervised algorithm that:
        1. Builds an ensemble of random trees
        2. Measures how quickly each point is isolated
        3. Anomalies are isolated faster (shorter paths)

        Returns DataFrame with isolation_score column added.
        """
        if df.empty:
            return df

        # Features for Isolation Forest
        feature_cols = [
            "log_return", "abs_return", "z_score",
            "goldstein_mean", "event_count"
        ]

        # Filter to available columns
        available_cols = [c for c in feature_cols if c in df.columns]
        X = df[available_cols].values

        # Handle any remaining NaN
        X = np.nan_to_num(X, nan=0.0)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Fit and predict
        # -1 = anomaly, 1 = normal
        predictions = self.isolation_forest.fit_predict(X_scaled)

        # Get anomaly scores (lower = more anomalous)
        scores = self.isolation_forest.decision_function(X_scaled)

        # Convert to probability (0-1, higher = more anomalous)
        # Scores are typically in [-0.5, 0.5], we normalize
        probabilities = 1 - (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)

        df = df.copy()
        df["isolation_prediction"] = predictions
        df["isolation_score"] = scores
        df["anomaly_probability"] = probabilities

        return df

    def detect_with_zscore(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Detect anomalies using Z-score threshold.

        This is the traditional statistical approach:
        - Z > threshold = unusual positive move
        - Z < -threshold = unusual negative move
        """
        if df.empty:
            return df

        df = df.copy()
        df["zscore_anomaly"] = df["z_score"].abs() > self.zscore_threshold

        return df

    def detect_event_mismatches(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Detect mismatches between events and market reactions.

        Two types:
        1. Unexplained move: Big return but no significant events
        2. Muted response: Big event but small return
        """
        if df.empty:
            return df

        df = df.copy()

        # Unexplained move: |z_score| > 1.5 but no events
        df["unexplained_move"] = (
            (df["z_score"].abs() > 1.5) &
            (df["event_count"] == 0)
        )

        # Muted response: |goldstein| > 5 but |return| < 0.5%
        df["muted_response"] = (
            (df["goldstein_min"].abs() > 5) &
            (df["abs_return"] < 0.005)
        )

        return df

    def detect_all(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> list[ProdAnomaly]:
        """
        Run all detection methods and combine results.

        This is the main method that:
        1. Prepares features
        2. Runs Isolation Forest
        3. Runs Z-score detection
        4. Runs event mismatch detection
        5. Combines results into ProdAnomaly objects
        """
        logger.info(f"Running production anomaly detection for {symbol}")

        # Prepare data
        df = self._prepare_features(symbol, start_date, end_date)

        if df.empty:
            logger.warning(f"No data for {symbol}")
            return []

        # Run detection methods
        df = self.detect_with_isolation_forest(df)
        df = self.detect_with_zscore(df)
        df = self.detect_event_mismatches(df)

        # Combine results
        anomalies = []

        for _, row in df.iterrows():
            detected_by = []

            # Check each detection method
            if row.get("isolation_prediction", 1) == -1:
                detected_by.append("isolation_forest")
            if row.get("zscore_anomaly", False):
                detected_by.append("zscore")
            if row.get("unexplained_move", False):
                detected_by.append("unexplained_move")
            if row.get("muted_response", False):
                detected_by.append("muted_response")

            # Only create anomaly if at least one method detected it
            if detected_by:
                # Determine primary anomaly type
                if "unexplained_move" in detected_by:
                    anomaly_type = "unexplained_move"
                elif "muted_response" in detected_by:
                    anomaly_type = "muted_response"
                else:
                    anomaly_type = "statistical_outlier"

                anomaly = ProdAnomaly(
                    date=row["date"],
                    symbol=symbol,
                    anomaly_type=anomaly_type,
                    actual_return=row["log_return"],
                    expected_return=row.get("rolling_mean", 0) or 0,
                    z_score=row.get("z_score", 0) or 0,
                    isolation_score=row.get("isolation_score", 0) or 0,
                    anomaly_probability=row.get("anomaly_probability", 0) or 0,
                    event_count=int(row.get("event_count", 0) or 0),
                    avg_goldstein=row.get("goldstein_mean", 0) or 0,
                    detected_by=detected_by,
                )
                anomalies.append(anomaly)

        logger.info(f"Found {len(anomalies)} anomalies for {symbol}")
        return anomalies

    def get_anomaly_report(
        self,
        anomalies: list[ProdAnomaly],
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> AnomalyReport:
        """
        Generate a summary report of detected anomalies.
        """
        total_days = (end_date - start_date).days

        # Count by type
        unexplained = sum(1 for a in anomalies if a.anomaly_type == "unexplained_move")
        muted = sum(1 for a in anomalies if a.anomaly_type == "muted_response")
        statistical = sum(1 for a in anomalies if a.anomaly_type == "statistical_outlier")

        # Sort by probability for top anomalies
        sorted_anomalies = sorted(
            anomalies,
            key=lambda a: a.anomaly_probability,
            reverse=True
        )

        # Build summary text
        summary_lines = [
            "",
            "=" * 60,
            f"ANOMALY DETECTION REPORT: {symbol}",
            "=" * 60,
            "",
            f"Period: {start_date} to {end_date}",
            f"Total trading days analyzed: {total_days}",
            "",
            "SUMMARY:",
            f"  Total anomalies detected: {len(anomalies)}",
            f"  Anomaly rate: {len(anomalies)/max(total_days,1)*100:.1f}%",
            "",
            "BREAKDOWN BY TYPE:",
            f"  Unexplained moves: {unexplained}",
            f"  Muted responses: {muted}",
            f"  Statistical outliers: {statistical}",
            "",
        ]

        if sorted_anomalies:
            summary_lines.extend([
                "TOP 5 ANOMALIES:",
                "-" * 60,
            ])
            for i, a in enumerate(sorted_anomalies[:5], 1):
                summary_lines.append(
                    f"  {i}. {a.date}: {a.anomaly_type} "
                    f"(return={a.actual_return*100:.2f}%, "
                    f"prob={a.anomaly_probability:.2f})"
                )
                summary_lines.append(f"     Detected by: {', '.join(a.detected_by)}")

        summary_lines.append("=" * 60)

        return AnomalyReport(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            total_days=total_days,
            anomaly_count=len(anomalies),
            anomaly_rate=len(anomalies) / max(total_days, 1),
            unexplained_moves=unexplained,
            muted_responses=muted,
            statistical_outliers=statistical,
            top_anomalies=sorted_anomalies[:10],
            summary="\n".join(summary_lines),
        )

    def compare_symbols(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Compare anomaly rates across multiple symbols.

        Useful for identifying which markets have the most unusual behavior.
        """
        rows = []

        for symbol in symbols:
            try:
                anomalies = self.detect_all(symbol, start_date, end_date)
                report = self.get_anomaly_report(anomalies, symbol, start_date, end_date)

                rows.append({
                    "symbol": symbol,
                    "anomaly_count": report.anomaly_count,
                    "anomaly_rate": report.anomaly_rate,
                    "unexplained_moves": report.unexplained_moves,
                    "muted_responses": report.muted_responses,
                    "statistical_outliers": report.statistical_outliers,
                })
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("anomaly_rate", ascending=False)

        return df


def run_quick_anomaly_detection(
    symbol: str,
    days: int = 90,
) -> Optional[str]:
    """
    Quick helper to run anomaly detection and get a report.

    Usage:
        from src.analysis.production_anomaly import run_quick_anomaly_detection
        print(run_quick_anomaly_detection("CL=F", days=90))
    """
    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    detector = ProductionAnomalyDetector()
    anomalies = detector.detect_all(symbol, start_date, end_date)
    report = detector.get_anomaly_report(anomalies, symbol, start_date, end_date)

    return report.summary
