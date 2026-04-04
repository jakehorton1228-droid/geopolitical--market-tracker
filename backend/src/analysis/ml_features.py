"""ML Feature Pipeline — produces training datasets for event impact classification.

Generates two dataset formats from the same underlying data:

1. FLAT FEATURES (for tree models + MLP):
   One row per trading day. Features are aggregated event metrics, market
   context, sentiment, and temporal indicators. Target is binary: did a
   significant market move occur?

   Shape: (n_samples, n_features)
   Models: XGBoost, LightGBM, Random Forest, Logistic Regression, MLP

2. WINDOWED SEQUENCES (for CNN + LSTM):
   A 30-day sliding window of daily feature vectors. Each sample is a
   matrix where rows are days and columns are features. Target is the
   next-day significant move.

   Shape: (n_samples, window_size, n_features)
   Models: 1D CNN, LSTM

Both formats share the same time-series aware train/val/test split to
prevent data leakage (no future data in training set).

USAGE:
------
    from src.analysis.ml_features import MLFeaturePipeline

    pipeline = MLFeaturePipeline()

    # Flat features for tree models
    flat = pipeline.build_flat_dataset(symbols=["SPY", "CL=F", "GC=F"])
    X_train, X_val, X_test = flat["X_train"], flat["X_val"], flat["X_test"]
    y_train, y_val, y_test = flat["y_train"], flat["y_val"], flat["y_test"]

    # Windowed sequences for CNN/LSTM
    seq = pipeline.build_sequence_dataset(symbols=["SPY", "CL=F", "GC=F"])
    X_train_seq = seq["X_train"]  # (n, 30, n_features)
"""

import logging
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import select, func, case

from src.db.connection import get_session
from src.db.models import Event, MarketData, NewsHeadline
from src.analysis.feature_engineering import FeatureEngineering, CONFLICT_CODES, COOPERATION_CODES

logger = logging.getLogger(__name__)

# Default date range — use all available data
DEFAULT_START = date(2016, 1, 1)
DEFAULT_END = date.today()

# Train/val/test split ratios (by time, not random)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
# TEST_RATIO = 0.15 (remainder)

# Sequence window size for CNN/LSTM
WINDOW_SIZE = 30

# Threshold for "significant" market move (absolute daily return)
SIGNIFICANT_MOVE_THRESHOLD = 0.015  # 1.5%


class MLFeaturePipeline:
    """Produces ML-ready datasets for event impact classification."""

    def __init__(self):
        self.fe = FeatureEngineering()

    # ------------------------------------------------------------------
    # Core: build merged daily DataFrame with all features
    # ------------------------------------------------------------------

    def _build_daily_features(
        self,
        symbol: str,
        start_date: date = DEFAULT_START,
        end_date: date = DEFAULT_END,
    ) -> pd.DataFrame:
        """Build a single DataFrame with all daily features for one symbol.

        Merges: market data + event aggregates + sentiment + temporal features.
        Returns one row per trading day.
        """
        # 1. Market data
        market_df = self.fe.fetch_market_data(
            symbol, start_date, end_date, include_volume=True
        )
        if market_df.empty or len(market_df) < 60:
            logger.warning(f"Insufficient market data for {symbol}")
            return pd.DataFrame()

        # 2. Event aggregates (SQL-level for performance)
        event_agg = self.fe.fetch_events_aggregated_sql(
            start_date, end_date,
            goldstein_metrics=["mean", "min", "max", "std"],
            include_cooperation=True,
        )

        # 3. Merge market + events
        if not event_agg.empty:
            merged = self.fe.merge_market_events(market_df, event_agg)
        else:
            merged = market_df.copy()
            for col in [
                "goldstein_mean", "goldstein_min", "goldstein_max",
                "goldstein_std", "mentions_total", "avg_tone",
                "conflict_count", "cooperation_count",
            ]:
                merged[col] = 0.0

        # 4. Add sentiment features
        merged = self._add_sentiment_features(merged, start_date, end_date)

        # 5. Add market context features (rolling stats)
        merged = self._add_market_context(merged)

        # 6. Add temporal features
        merged = self._add_temporal_features(merged)

        # 7. Add event velocity (rate of change in event intensity)
        merged = self._add_event_velocity(merged)

        # 8. Add target variable
        merged["target"] = (
            merged["log_return"].abs() >= SIGNIFICANT_MOVE_THRESHOLD
        ).astype(int)

        # 9. Add symbol column for multi-symbol datasets
        merged["symbol"] = symbol

        # Drop rows with NaN from rolling calculations
        merged = merged.dropna().reset_index(drop=True)

        return merged

    def _add_sentiment_features(
        self, df: pd.DataFrame, start_date: date, end_date: date,
    ) -> pd.DataFrame:
        """Add daily aggregated headline sentiment features."""
        df = df.copy()

        with get_session() as session:
            rows = session.execute(
                select(
                    func.date(NewsHeadline.published_at).label("date"),
                    func.avg(NewsHeadline.sentiment_score).label("sentiment_mean"),
                    func.min(NewsHeadline.sentiment_score).label("sentiment_min"),
                    func.max(NewsHeadline.sentiment_score).label("sentiment_max"),
                    func.count(NewsHeadline.id).label("headline_count"),
                ).where(
                    func.date(NewsHeadline.published_at) >= start_date,
                    func.date(NewsHeadline.published_at) <= end_date,
                    NewsHeadline.sentiment_score.isnot(None),
                ).group_by(
                    func.date(NewsHeadline.published_at),
                )
            ).all()

        if rows:
            sent_df = pd.DataFrame(rows, columns=[
                "date", "sentiment_mean", "sentiment_min",
                "sentiment_max", "headline_count",
            ])
            df = pd.merge(df, sent_df, on="date", how="left")

        for col in ["sentiment_mean", "sentiment_min", "sentiment_max", "headline_count"]:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = df[col].fillna(0.0)

        return df

    def _add_market_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling market statistics as features."""
        df = df.copy()

        # Rolling volatility (5, 10, 20 day)
        for window in [5, 10, 20]:
            df[f"volatility_{window}d"] = (
                df["log_return"].rolling(window=window, min_periods=3).std()
            )

        # Rolling mean return
        df["return_5d_mean"] = df["log_return"].rolling(window=5, min_periods=3).mean()
        df["return_20d_mean"] = df["log_return"].rolling(window=20, min_periods=10).mean()

        # Price momentum (% change from N days ago)
        df["momentum_5d"] = df["close"].pct_change(periods=5)
        df["momentum_20d"] = df["close"].pct_change(periods=20)

        # Volume features
        if "volume" in df.columns and df["volume"].notna().any():
            df["volume_ratio"] = df["volume"] / df["volume"].rolling(
                window=20, min_periods=10
            ).mean()
            df["volume_ratio"] = df["volume_ratio"].fillna(1.0)
        else:
            df["volume_ratio"] = 1.0

        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        df = df.copy()

        df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek  # 0=Mon, 4=Fri
        df["month"] = pd.to_datetime(df["date"]).dt.month
        df["is_monday"] = (df["day_of_week"] == 0).astype(int)
        df["is_friday"] = (df["day_of_week"] == 4).astype(int)

        return df

    def _add_event_velocity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rate-of-change features for event intensity."""
        df = df.copy()

        # Change in conflict count vs 5-day average
        df["conflict_velocity"] = (
            df["conflict_count"]
            - df["conflict_count"].rolling(window=5, min_periods=1).mean()
        )

        # Change in mentions vs 5-day average
        df["mentions_velocity"] = (
            df["mentions_total"]
            - df["mentions_total"].rolling(window=5, min_periods=1).mean()
        )

        # Goldstein trend (is the situation getting worse?)
        df["goldstein_trend"] = (
            df["goldstein_mean"]
            - df["goldstein_mean"].rolling(window=5, min_periods=1).mean()
        )

        return df

    # ------------------------------------------------------------------
    # Feature column definitions
    # ------------------------------------------------------------------

    FLAT_FEATURE_COLS = [
        # Event features
        "goldstein_mean", "goldstein_min", "goldstein_max", "goldstein_std",
        "mentions_total", "avg_tone",
        "conflict_count", "cooperation_count",
        # Sentiment features
        "sentiment_mean", "sentiment_min", "sentiment_max", "headline_count",
        # Market context
        "volatility_5d", "volatility_10d", "volatility_20d",
        "return_5d_mean", "return_20d_mean",
        "momentum_5d", "momentum_20d",
        "volume_ratio",
        # Temporal
        "day_of_week", "month", "is_monday", "is_friday",
        # Event velocity
        "conflict_velocity", "mentions_velocity", "goldstein_trend",
    ]

    SEQUENCE_FEATURE_COLS = [
        # Subset for sequences — exclude temporal categoricals
        "goldstein_mean", "goldstein_min", "goldstein_max",
        "mentions_total", "avg_tone",
        "conflict_count", "cooperation_count",
        "sentiment_mean", "headline_count",
        "volatility_5d", "return_5d_mean",
        "volume_ratio",
        "conflict_velocity", "mentions_velocity", "goldstein_trend",
        "log_return",
    ]

    # ------------------------------------------------------------------
    # Public API: build datasets
    # ------------------------------------------------------------------

    def build_flat_dataset(
        self,
        symbols: list[str],
        start_date: date = DEFAULT_START,
        end_date: date = DEFAULT_END,
    ) -> dict:
        """Build flat feature dataset for tree models + MLP.

        Returns dict with X_train, X_val, X_test, y_train, y_val, y_test,
        feature_names, dates_train, dates_val, dates_test.
        """
        all_dfs = []
        for symbol in symbols:
            df = self._build_daily_features(symbol, start_date, end_date)
            if not df.empty:
                all_dfs.append(df)
                logger.info(f"{symbol}: {len(df)} samples")

        if not all_dfs:
            raise ValueError("No data available for any symbol")

        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.sort_values("date").reset_index(drop=True)

        logger.info(
            f"Combined dataset: {len(combined)} samples, "
            f"target distribution: {combined['target'].mean():.1%} positive"
        )

        # Time-series split
        return self._time_split(
            combined, self.FLAT_FEATURE_COLS, "target"
        )

    def build_sequence_dataset(
        self,
        symbols: list[str],
        window_size: int = WINDOW_SIZE,
        start_date: date = DEFAULT_START,
        end_date: date = DEFAULT_END,
    ) -> dict:
        """Build windowed sequence dataset for CNN/LSTM.

        Returns dict with X_train, X_val, X_test (3D arrays),
        y_train, y_val, y_test, feature_names.
        """
        all_dfs = []
        for symbol in symbols:
            df = self._build_daily_features(symbol, start_date, end_date)
            if not df.empty:
                all_dfs.append(df)

        if not all_dfs:
            raise ValueError("No data available for any symbol")

        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.sort_values("date").reset_index(drop=True)

        # Build sliding windows
        feature_cols = self.SEQUENCE_FEATURE_COLS
        X_windows = []
        y_windows = []
        dates_windows = []

        for i in range(window_size, len(combined)):
            window = combined.iloc[i - window_size:i]

            # Only create window if it's contiguous (same symbol, no gaps > 5 days)
            if window["symbol"].nunique() > 1:
                continue
            date_range = (window["date"].max() - window["date"].min()).days
            if date_range > window_size * 2:  # Too many gaps
                continue

            X_windows.append(window[feature_cols].values)
            y_windows.append(combined.iloc[i]["target"])
            dates_windows.append(combined.iloc[i]["date"])

        if not X_windows:
            raise ValueError("No valid windows could be created")

        X = np.array(X_windows, dtype=np.float32)
        y = np.array(y_windows, dtype=np.int64)
        dates = np.array(dates_windows)

        logger.info(
            f"Sequence dataset: {X.shape[0]} windows, "
            f"shape: {X.shape}, target: {y.mean():.1%} positive"
        )

        # Time-series split on windows
        n = len(X)
        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

        return {
            "X_train": X[:train_end],
            "X_val": X[train_end:val_end],
            "X_test": X[val_end:],
            "y_train": y[:train_end],
            "y_val": y[train_end:val_end],
            "y_test": y[val_end:],
            "feature_names": feature_cols,
            "window_size": window_size,
            "dates_train": dates[:train_end],
            "dates_val": dates[train_end:val_end],
            "dates_test": dates[val_end:],
        }

    # ------------------------------------------------------------------
    # Train/val/test split (time-series aware)
    # ------------------------------------------------------------------

    def _time_split(
        self, df: pd.DataFrame, feature_cols: list[str], target_col: str,
    ) -> dict:
        """Split data chronologically — no shuffling, no leakage.

        Train: first 70%
        Validation: next 15%
        Test: final 15%
        """
        n = len(df)
        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

        train = df.iloc[:train_end]
        val = df.iloc[train_end:val_end]
        test = df.iloc[val_end:]

        logger.info(
            f"Split: train={len(train)} ({train['date'].min()} to {train['date'].max()}), "
            f"val={len(val)} ({val['date'].min()} to {val['date'].max()}), "
            f"test={len(test)} ({test['date'].min()} to {test['date'].max()})"
        )

        return {
            "X_train": train[feature_cols].values.astype(np.float32),
            "X_val": val[feature_cols].values.astype(np.float32),
            "X_test": test[feature_cols].values.astype(np.float32),
            "y_train": train[target_col].values.astype(np.int64),
            "y_val": val[target_col].values.astype(np.int64),
            "y_test": test[target_col].values.astype(np.int64),
            "feature_names": feature_cols,
            "dates_train": train["date"].values,
            "dates_val": val["date"].values,
            "dates_test": test["date"].values,
            "symbols_train": train["symbol"].values,
            "symbols_val": val["symbol"].values,
            "symbols_test": test["symbol"].values,
        }
