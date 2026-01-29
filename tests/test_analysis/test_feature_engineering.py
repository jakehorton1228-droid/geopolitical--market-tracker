"""Tests for the shared feature engineering module."""

import sys
from pathlib import Path
from datetime import date, timedelta
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.conftest import make_event_series, make_market_series


class TestFeatureEngineering:
    """Tests for the FeatureEngineering class."""

    def _make_fe(self):
        from src.analysis.feature_engineering import FeatureEngineering
        return FeatureEngineering()

    @patch("src.analysis.feature_engineering.get_session")
    @patch("src.analysis.feature_engineering.get_market_data")
    def test_fetch_market_data_returns_dataframe(self, mock_get_md, mock_session):
        """fetch_market_data should return a DataFrame with expected columns."""
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        market_data = make_market_series(20, "SPY")
        mock_get_md.return_value = market_data

        fe = self._make_fe()
        df = fe.fetch_market_data("SPY", date.today() - timedelta(days=30), date.today())

        assert not df.empty
        assert "date" in df.columns
        assert "close" in df.columns
        assert "log_return" in df.columns
        assert len(df) == 20

    @patch("src.analysis.feature_engineering.get_session")
    @patch("src.analysis.feature_engineering.get_market_data")
    def test_fetch_market_data_empty(self, mock_get_md, mock_session):
        """fetch_market_data should return empty DataFrame when no data."""
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_md.return_value = []

        fe = self._make_fe()
        df = fe.fetch_market_data("SPY", date.today() - timedelta(days=30), date.today())

        assert df.empty

    @patch("src.analysis.feature_engineering.get_session")
    @patch("src.analysis.feature_engineering.get_events_by_date_range")
    def test_fetch_events_parses_conflict_codes(self, mock_get_events, mock_session):
        """fetch_events should correctly classify conflict and cooperation events."""
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        from tests.conftest import make_event

        events = [
            make_event(id=1, event_root_code="18"),  # conflict
            make_event(id=2, event_root_code="03"),  # cooperation
            make_event(id=3, event_root_code="14"),  # neither
        ]
        mock_get_events.return_value = events

        fe = self._make_fe()
        df = fe.fetch_events(date.today() - timedelta(days=30), date.today())

        assert len(df) == 3
        assert df["is_conflict"].sum() == 1
        assert df["is_cooperation"].sum() == 1

    def test_aggregate_events_default_metrics(self):
        """aggregate_events should produce correct column names."""
        fe = self._make_fe()

        events_df = pd.DataFrame({
            "date": [date.today()] * 3,
            "goldstein_scale": [-5.0, -2.0, 3.0],
            "num_mentions": [10, 20, 30],
            "avg_tone": [-3.0, -1.0, 1.0],
            "is_conflict": [1, 0, 0],
            "is_cooperation": [0, 0, 1],
        })

        result = fe.aggregate_events(events_df)

        assert "goldstein_mean" in result.columns
        assert "goldstein_min" in result.columns
        assert "goldstein_max" in result.columns
        assert "mentions_total" in result.columns
        assert "conflict_count" in result.columns
        assert "cooperation_count" in result.columns
        assert len(result) == 1  # One date

    def test_aggregate_events_custom_metrics(self):
        """aggregate_events should support custom aggregation configs."""
        fe = self._make_fe()

        events_df = pd.DataFrame({
            "date": [date.today()] * 5,
            "goldstein_scale": [-5.0, -2.0, 3.0, 1.0, -1.0],
            "num_mentions": [10, 20, 30, 5, 15],
            "avg_tone": [-3.0, -1.0, 1.0, 0.5, -0.5],
            "is_conflict": [1, 0, 0, 0, 1],
        })

        result = fe.aggregate_events(
            events_df,
            goldstein_metrics=["mean", "std"],
            mention_metrics=["sum", "max"],
            include_cooperation=False,
        )

        assert "goldstein_mean" in result.columns
        assert "goldstein_std" in result.columns
        assert "mentions_total" in result.columns
        assert "mentions_max" in result.columns
        assert "cooperation_count" not in result.columns

    def test_merge_market_events_fills_missing(self):
        """merge_market_events should fill NaN with 0."""
        fe = self._make_fe()

        market_df = pd.DataFrame({
            "date": [date.today() - timedelta(days=i) for i in range(5)],
            "close": [100, 101, 102, 103, 104],
            "log_return": [0.01, -0.01, 0.02, -0.005, 0.01],
        })

        # Only 2 days have events
        event_agg = pd.DataFrame({
            "date": [date.today() - timedelta(days=0), date.today() - timedelta(days=2)],
            "goldstein_mean": [-3.0, 2.0],
            "conflict_count": [5, 0],
        })

        merged = fe.merge_market_events(market_df, event_agg)

        assert len(merged) == 5
        # Days without events should have 0, not NaN
        assert merged["goldstein_mean"].isna().sum() == 0
        assert merged["conflict_count"].isna().sum() == 0

    def test_add_rolling_stats(self):
        """add_rolling_stats should add z_score and rolling columns."""
        fe = self._make_fe()

        np.random.seed(42)
        n = 50
        df = pd.DataFrame({
            "date": [date.today() - timedelta(days=n - i) for i in range(n)],
            "log_return": np.random.normal(0, 0.01, n),
            "volume": np.random.randint(500000, 2000000, n),
        })

        result = fe.add_rolling_stats(df, lookback_days=20)

        assert "abs_return" in result.columns
        assert "rolling_mean" in result.columns
        assert "rolling_std" in result.columns
        assert "z_score" in result.columns
        assert "volume_change" in result.columns
        assert "volume_zscore" in result.columns

    @patch("src.analysis.feature_engineering.get_session")
    @patch("src.analysis.feature_engineering.get_market_data")
    @patch("src.analysis.feature_engineering.get_events_by_date_range")
    def test_prepare_classification_features_shape(
        self, mock_events, mock_market, mock_session
    ):
        """prepare_classification_features should return correct shapes."""
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        mock_market.return_value = make_market_series(50, "SPY")
        mock_events.return_value = make_event_series(50)

        fe = self._make_fe()
        X, y, names = fe.prepare_classification_features(
            "SPY",
            date.today() - timedelta(days=60),
            date.today(),
        )

        assert len(X) > 0
        assert len(X) == len(y)
        assert len(names) == 7  # 7 features for classification
        assert set(y).issubset({0, 1})  # Binary target

    @patch("src.analysis.feature_engineering.get_session")
    @patch("src.analysis.feature_engineering.get_market_data")
    @patch("src.analysis.feature_engineering.get_events_by_date_range")
    def test_prepare_regression_features_continuous_target(
        self, mock_events, mock_market, mock_session
    ):
        """prepare_regression_features should return continuous target."""
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        mock_market.return_value = make_market_series(50, "SPY")
        mock_events.return_value = make_event_series(50)

        fe = self._make_fe()
        X, y, names = fe.prepare_regression_features(
            "SPY",
            date.today() - timedelta(days=60),
            date.today(),
        )

        assert len(X) > 0
        assert len(names) == 4  # 4 features for regression
        # Target should be continuous (not just 0/1)
        assert not set(y).issubset({0, 1})

    @patch("src.analysis.feature_engineering.get_session")
    @patch("src.analysis.feature_engineering.get_market_data")
    @patch("src.analysis.feature_engineering.get_events_by_date_range")
    def test_prepare_extended_features_more_columns(
        self, mock_events, mock_market, mock_session
    ):
        """prepare_extended_features should have more features than basic."""
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        mock_market.return_value = make_market_series(50, "SPY")
        mock_events.return_value = make_event_series(50)

        fe = self._make_fe()
        X, y, names = fe.prepare_extended_features(
            "SPY",
            date.today() - timedelta(days=60),
            date.today(),
        )

        assert len(X) > 0
        assert len(names) == 9  # 9 features for gradient boost
        assert "goldstein_std" in names
        assert "mentions_max" in names
