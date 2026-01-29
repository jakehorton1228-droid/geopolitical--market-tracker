"""Tests for the data quality module."""

import sys
from pathlib import Path
from datetime import date, timedelta
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.conftest import make_market_series, make_event_series, make_market_data


class TestDataQualityChecker:
    """Tests for the DataQualityChecker class."""

    def _make_checker(self):
        from src.analysis.data_quality import DataQualityChecker
        return DataQualityChecker()

    def test_count_business_days(self):
        """Should approximate business days correctly."""
        checker = self._make_checker()

        # 7 calendar days = ~5 business days
        result = checker._count_business_days(
            date(2024, 1, 1), date(2024, 1, 8)
        )
        assert result == 5

        # 14 calendar days = ~10 business days
        result = checker._count_business_days(
            date(2024, 1, 1), date(2024, 1, 15)
        )
        assert result == 10

    @patch("src.analysis.data_quality.get_session")
    @patch("src.analysis.data_quality.get_market_data")
    def test_check_completeness_full_data(self, mock_get_md, mock_session):
        """Should return high score when all data present."""
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        # 22 trading days for ~30 calendar days
        mock_get_md.return_value = make_market_series(22, "SPY")

        checker = self._make_checker()
        score, issues = checker.check_completeness(
            "SPY",
            date.today() - timedelta(days=30),
            date.today(),
        )

        assert score >= 0.8
        # Should have no high-severity issues
        high_issues = [i for i in issues if i.severity == "high"]
        assert len(high_issues) == 0

    @patch("src.analysis.data_quality.get_session")
    @patch("src.analysis.data_quality.get_market_data")
    def test_check_completeness_missing_data(self, mock_get_md, mock_session):
        """Should detect missing data and return low score."""
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        # Only 5 days of data for a 30-day window
        mock_get_md.return_value = make_market_series(5, "SPY")

        checker = self._make_checker()
        score, issues = checker.check_completeness(
            "SPY",
            date.today() - timedelta(days=30),
            date.today(),
        )

        assert score < 0.5
        assert len(issues) > 0

    @patch("src.analysis.data_quality.get_session")
    @patch("src.analysis.data_quality.get_market_data")
    def test_check_completeness_no_data(self, mock_get_md, mock_session):
        """Should return 0 score with high severity issue for no data."""
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_md.return_value = []

        checker = self._make_checker()
        score, issues = checker.check_completeness(
            "SPY",
            date.today() - timedelta(days=30),
            date.today(),
        )

        assert score == 0.0
        assert len(issues) == 1
        assert issues[0].severity == "high"

    @patch("src.analysis.data_quality.get_session")
    @patch("src.analysis.data_quality.get_latest_market_date")
    def test_check_freshness_fresh_data(self, mock_latest, mock_session):
        """Should return high score for recent data."""
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        # Data from yesterday
        mock_latest.return_value = date.today() - timedelta(days=1)

        checker = self._make_checker()
        score, days_stale, last_update, issues = checker.check_freshness("SPY")

        assert score >= 0.8
        assert days_stale == 1
        assert last_update == date.today() - timedelta(days=1)

    @patch("src.analysis.data_quality.get_session")
    @patch("src.analysis.data_quality.get_latest_market_date")
    def test_check_freshness_stale_data(self, mock_latest, mock_session):
        """Should flag stale data."""
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        # Data from 10 days ago
        mock_latest.return_value = date.today() - timedelta(days=10)

        checker = self._make_checker()
        score, days_stale, last_update, issues = checker.check_freshness("SPY")

        assert days_stale == 10
        assert len(issues) > 0

    @patch("src.analysis.data_quality.get_session")
    @patch("src.analysis.data_quality.get_latest_market_date")
    def test_check_freshness_no_data(self, mock_latest, mock_session):
        """Should handle missing data gracefully."""
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)
        mock_latest.return_value = None

        checker = self._make_checker()
        score, days_stale, last_update, issues = checker.check_freshness("SPY")

        assert score == 0.0
        assert days_stale == 999
        assert last_update is None
        assert issues[0].severity == "high"

    @patch("src.analysis.data_quality.get_session")
    @patch("src.analysis.data_quality.get_market_data")
    def test_check_validity_good_data(self, mock_get_md, mock_session):
        """Should return high validity for normal data."""
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        mock_get_md.return_value = make_market_series(30, "SPY")

        checker = self._make_checker()
        score, outliers, issues = checker.check_validity(
            "SPY",
            date.today() - timedelta(days=30),
            date.today(),
        )

        assert score >= 0.9
        assert outliers == 0

    @patch("src.analysis.data_quality.get_session")
    @patch("src.analysis.data_quality.get_market_data")
    def test_check_validity_extreme_returns(self, mock_get_md, mock_session):
        """Should flag extreme daily returns as outliers."""
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        data = make_market_series(10, "SPY")
        # Make one data point have an extreme return
        data[5].daily_return = 0.5  # 50% daily return

        mock_get_md.return_value = data

        checker = self._make_checker()
        score, outliers, issues = checker.check_validity(
            "SPY",
            date.today() - timedelta(days=30),
            date.today(),
        )

        assert outliers >= 1
        outlier_issues = [i for i in issues if i.issue_type == "outlier"]
        assert len(outlier_issues) >= 1

    @patch("src.analysis.data_quality.get_session")
    @patch("src.analysis.data_quality.get_events_by_date_range")
    def test_check_event_quality_valid_events(self, mock_events, mock_session):
        """Should return high score for valid events."""
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        mock_events.return_value = make_event_series(30)

        checker = self._make_checker()
        score, issues = checker.check_event_quality(
            date.today() - timedelta(days=30),
            date.today(),
        )

        assert score > 0.8

    @patch("src.analysis.data_quality.get_session")
    @patch("src.analysis.data_quality.get_events_by_date_range")
    def test_check_event_quality_no_events(self, mock_events, mock_session):
        """Should flag missing events."""
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)
        mock_events.return_value = []

        checker = self._make_checker()
        score, issues = checker.check_event_quality(
            date.today() - timedelta(days=30),
            date.today(),
        )

        assert score == 0.0
        assert len(issues) > 0
        assert issues[0].severity == "high"


class TestQualityIssue:
    """Tests for the QualityIssue dataclass."""

    def test_quality_issue_creation(self):
        from src.analysis.data_quality import QualityIssue

        issue = QualityIssue(
            issue_type="missing_data",
            severity="high",
            description="No data for SPY",
            metric_name="record_count",
            metric_value=0,
            threshold=1,
            affected_symbol="SPY",
        )

        assert issue.issue_type == "missing_data"
        assert issue.severity == "high"
        assert issue.affected_symbol == "SPY"
