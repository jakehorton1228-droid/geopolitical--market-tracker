"""
Shared Feature Engineering Module.

Centralizes the feature preparation logic that was previously
duplicated across production_classifier.py, production_regression.py,
production_anomaly.py, and gradient_boost_classifier.py.

This module provides a single FeatureEngineering class that handles:
1. Fetching market and event data from the database
2. Parsing CAMEO event codes into conflict/cooperation categories
3. Aggregating events by date with configurable metrics
4. Merging market and event data
5. Computing rolling statistics for time-series features

Each analysis module can call the appropriate method to get exactly
the features it needs without duplicating the ETL logic.

USAGE:
------
    from src.analysis.feature_engineering import FeatureEngineering

    fe = FeatureEngineering()

    # For classification (binary target)
    X, y, names = fe.prepare_classification_features("SPY", start, end)

    # For regression (continuous target)
    X, y, names = fe.prepare_regression_features("SPY", start, end)

    # For anomaly detection (full DataFrame with rolling stats)
    df = fe.prepare_anomaly_features("SPY", start, end, lookback_days=30)

    # For gradient boosting (extended feature set)
    X, y, names = fe.prepare_extended_features("SPY", start, end)
"""

from datetime import date, timedelta
from typing import Optional
import logging

import numpy as np
import pandas as pd

from src.db.connection import get_session
from src.db.queries import get_market_data, get_events_by_date_range

logger = logging.getLogger(__name__)

# CAMEO root codes for conflict events (military/violent)
CONFLICT_CODES = {"18", "19", "20"}

# CAMEO root codes for cooperation events (diplomatic/positive)
COOPERATION_CODES = {"03", "04", "05", "06"}


class FeatureEngineering:
    """
    Shared feature engineering for all analysis modules.

    Eliminates code duplication by providing a single source for
    data fetching, event aggregation, and feature computation.
    """

    def fetch_market_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        include_volume: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch market data and return as a DataFrame.

        Args:
            symbol: Market symbol to fetch
            start_date: Start of date range
            end_date: End of date range
            include_volume: Whether to include volume data

        Returns:
            DataFrame with date, close, log_return, and optionally volume
        """
        with get_session() as session:
            market_data = get_market_data(session, symbol, start_date, end_date)

            if not market_data:
                return pd.DataFrame()

            rows = []
            for m in market_data:
                row = {
                    "date": m.date,
                    "close": float(m.close),
                    "log_return": m.log_return,
                    "daily_return": m.daily_return,
                }
                if include_volume:
                    row["volume"] = m.volume
                rows.append(row)

            return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    def fetch_events(
        self,
        start_date: date,
        end_date: date,
        include_cooperation: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch events and parse CAMEO codes into categories.

        Returns:
            DataFrame with date, goldstein_scale, num_mentions,
            avg_tone, is_conflict, and optionally is_cooperation
        """
        with get_session() as session:
            events = get_events_by_date_range(session, start_date, end_date)

            if not events:
                return pd.DataFrame()

            rows = []
            for e in events:
                row = {
                    "date": e.event_date,
                    "goldstein_scale": e.goldstein_scale or 0,
                    "num_mentions": e.num_mentions or 0,
                    "avg_tone": e.avg_tone or 0,
                    "is_conflict": 1 if e.event_root_code in CONFLICT_CODES else 0,
                }
                if include_cooperation:
                    row["is_cooperation"] = (
                        1 if e.event_root_code in COOPERATION_CODES else 0
                    )
                rows.append(row)

            return pd.DataFrame(rows)

    def aggregate_events(
        self,
        events_df: pd.DataFrame,
        goldstein_metrics: list[str] = None,
        mention_metrics: list[str] = None,
        include_cooperation: bool = True,
    ) -> pd.DataFrame:
        """
        Aggregate events by date with configurable metrics.

        Args:
            events_df: Raw events DataFrame
            goldstein_metrics: Aggregation functions for goldstein_scale
                             (default: ["mean", "min", "max"])
            mention_metrics: Aggregation functions for num_mentions
                           (default: ["sum"])
            include_cooperation: Whether to include cooperation_count

        Returns:
            Aggregated DataFrame with one row per date
        """
        if events_df.empty:
            return pd.DataFrame()

        if goldstein_metrics is None:
            goldstein_metrics = ["mean", "min", "max"]
        if mention_metrics is None:
            mention_metrics = ["sum"]

        agg_config = {
            "goldstein_scale": goldstein_metrics,
            "num_mentions": mention_metrics,
            "avg_tone": "mean",
            "is_conflict": "sum",
        }
        if include_cooperation and "is_cooperation" in events_df.columns:
            agg_config["is_cooperation"] = "sum"

        event_agg = events_df.groupby("date").agg(agg_config).reset_index()

        # Flatten hierarchical column names
        columns = ["date"]
        for metric in goldstein_metrics:
            columns.append(f"goldstein_{metric}")
        for metric in mention_metrics:
            if metric == "sum":
                columns.append("mentions_total")
            elif metric == "max":
                columns.append("mentions_max")
            else:
                columns.append(f"mentions_{metric}")
        columns.append("avg_tone")
        columns.append("conflict_count")
        if include_cooperation and "is_cooperation" in events_df.columns:
            columns.append("cooperation_count")

        event_agg.columns = columns

        # Fill NaN std (happens when only 1 event per day)
        if "goldstein_std" in event_agg.columns:
            event_agg["goldstein_std"] = event_agg["goldstein_std"].fillna(0)

        return event_agg

    def merge_market_events(
        self,
        market_df: pd.DataFrame,
        event_agg: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge market data with aggregated event data.

        Left join ensures we keep all market days, filling
        event-less days with zeros.
        """
        merged = pd.merge(market_df, event_agg, on="date", how="left")
        merged = merged.fillna(0)
        merged = merged.dropna(subset=["log_return"])
        return merged

    def add_rolling_stats(
        self,
        df: pd.DataFrame,
        lookback_days: int = 30,
    ) -> pd.DataFrame:
        """
        Add rolling statistics for anomaly detection and drift analysis.

        Adds: abs_return, rolling_mean, rolling_std, z_score,
              volume_change, volume_zscore (if volume present)
        """
        df = df.copy()

        df["abs_return"] = df["log_return"].abs()
        df["rolling_mean"] = df["log_return"].rolling(
            window=lookback_days, min_periods=10
        ).mean()
        df["rolling_std"] = df["log_return"].rolling(
            window=lookback_days, min_periods=10
        ).std()
        df["z_score"] = (
            (df["log_return"] - df["rolling_mean"]) / df["rolling_std"]
        )

        if "volume" in df.columns and df["volume"].notna().any():
            df["volume_change"] = df["volume"].pct_change()
            df["volume_zscore"] = (
                (df["volume"] - df["volume"].rolling(lookback_days).mean())
                / df["volume"].rolling(lookback_days).std()
            )
        else:
            df["volume_change"] = 0
            df["volume_zscore"] = 0

        return df

    def prepare_classification_features(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        min_samples: int = 20,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Prepare features for binary classification (UP/DOWN).

        Used by: ProductionClassifier

        Features: goldstein_mean, goldstein_min, goldstein_max,
                 mentions_total, avg_tone, conflict_count, cooperation_count
        Target: 1 if log_return > 0, else 0
        """
        market_df = self.fetch_market_data(symbol, start_date, end_date)
        if market_df.empty:
            return np.array([]), np.array([]), []

        events_df = self.fetch_events(start_date, end_date, include_cooperation=True)
        if events_df.empty:
            return np.array([]), np.array([]), []

        event_agg = self.aggregate_events(
            events_df,
            goldstein_metrics=["mean", "min", "max"],
            mention_metrics=["sum"],
            include_cooperation=True,
        )

        merged = self.merge_market_events(market_df, event_agg)

        if len(merged) < min_samples:
            return np.array([]), np.array([]), []

        y = (merged["log_return"] > 0).astype(int).values

        feature_cols = [
            "goldstein_mean", "goldstein_min", "goldstein_max",
            "mentions_total", "avg_tone", "conflict_count", "cooperation_count",
        ]
        X = merged[feature_cols].values

        return X, y, feature_cols

    def prepare_regression_features(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        min_samples: int = 10,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Prepare features for regression (continuous target).

        Used by: ProductionRegression

        Features: goldstein_mean, mentions_total, avg_tone, conflict_count
        Target: log_return (continuous)
        """
        market_df = self.fetch_market_data(symbol, start_date, end_date)
        if market_df.empty:
            return np.array([]), np.array([]), []

        events_df = self.fetch_events(start_date, end_date, include_cooperation=False)
        if events_df.empty:
            return np.array([]), np.array([]), []

        event_agg = self.aggregate_events(
            events_df,
            goldstein_metrics=["mean"],
            mention_metrics=["sum"],
            include_cooperation=False,
        )

        merged = self.merge_market_events(market_df, event_agg)

        if len(merged) < min_samples:
            return np.array([]), np.array([]), []

        y = merged["log_return"].values

        feature_cols = ["goldstein_mean", "mentions_total", "avg_tone", "conflict_count"]
        X = merged[feature_cols].values

        return X, y, feature_cols

    def prepare_extended_features(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        min_samples: int = 30,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Prepare extended features for gradient boosting.

        Used by: GradientBoostClassifier

        Features: goldstein_mean, goldstein_min, goldstein_max, goldstein_std,
                 mentions_total, mentions_max, avg_tone,
                 conflict_count, cooperation_count
        Target: 1 if log_return > 0, else 0
        """
        market_df = self.fetch_market_data(symbol, start_date, end_date)
        if market_df.empty:
            logger.warning(f"No market data for {symbol}")
            return np.array([]), np.array([]), []

        events_df = self.fetch_events(start_date, end_date, include_cooperation=True)
        if events_df.empty:
            logger.warning(f"No events found for date range")
            return np.array([]), np.array([]), []

        event_agg = self.aggregate_events(
            events_df,
            goldstein_metrics=["mean", "min", "max", "std"],
            mention_metrics=["sum", "max"],
            include_cooperation=True,
        )

        merged = self.merge_market_events(market_df, event_agg)

        if len(merged) < min_samples:
            logger.warning(f"Insufficient data for {symbol}: {len(merged)} samples")
            return np.array([]), np.array([]), []

        y = (merged["log_return"] > 0).astype(int).values

        feature_cols = [
            "goldstein_mean", "goldstein_min", "goldstein_max", "goldstein_std",
            "mentions_total", "mentions_max",
            "avg_tone",
            "conflict_count", "cooperation_count",
        ]
        X = merged[feature_cols].values

        logger.info(
            f"Prepared {len(X)} samples with {len(feature_cols)} features for {symbol}"
        )
        return X, y, feature_cols

    def prepare_anomaly_features(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        lookback_days: int = 30,
    ) -> pd.DataFrame:
        """
        Prepare full feature DataFrame for anomaly detection.

        Used by: ProductionAnomalyDetector

        Returns a DataFrame (not numpy arrays) with rolling statistics,
        market features, and event features. Fetches extra historical
        data for the lookback window.
        """
        fetch_start = start_date - timedelta(days=lookback_days + 10)

        market_df = self.fetch_market_data(
            symbol, fetch_start, end_date, include_volume=True
        )
        if market_df.empty or len(market_df) < lookback_days:
            return pd.DataFrame()

        # Add rolling stats
        market_df = self.add_rolling_stats(market_df, lookback_days)

        # Get events
        events_df = self.fetch_events(fetch_start, end_date, include_cooperation=False)

        if not events_df.empty:
            event_agg = events_df.groupby("date").agg({
                "goldstein_scale": ["mean", "min", "count"],
                "num_mentions": "sum",
            }).reset_index()
            event_agg.columns = [
                "date", "goldstein_mean", "goldstein_min",
                "event_count", "mentions_total",
            ]
            market_df = pd.merge(market_df, event_agg, on="date", how="left")

        # Fill missing event data
        for col in ["goldstein_mean", "goldstein_min", "event_count", "mentions_total"]:
            if col not in market_df.columns:
                market_df[col] = 0
            else:
                market_df[col] = market_df[col].fillna(0)

        # Filter to requested date range
        market_df = market_df[
            (market_df["date"] >= start_date) & (market_df["date"] <= end_date)
        ].copy()

        market_df = market_df.dropna(subset=["log_return", "z_score"])

        return market_df
