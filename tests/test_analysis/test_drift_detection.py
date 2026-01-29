"""Tests for the drift detection module."""

import sys
from pathlib import Path
from datetime import date, timedelta
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestDriftDetector:
    """Tests for the DriftDetector class."""

    def _make_detector(self, **kwargs):
        from src.analysis.drift_detection import DriftDetector
        return DriftDetector(**kwargs)

    def test_calculate_psi_identical_distributions(self):
        """PSI should be near 0 for identical distributions."""
        from src.analysis.drift_detection import DriftDetector

        np.random.seed(42)
        baseline = np.random.normal(0, 1, 1000)
        current = np.random.normal(0, 1, 1000)

        psi = DriftDetector._calculate_psi(baseline, current)

        assert psi < 0.1  # Should be very low

    def test_calculate_psi_shifted_distribution(self):
        """PSI should be high for significantly shifted distributions."""
        from src.analysis.drift_detection import DriftDetector

        np.random.seed(42)
        baseline = np.random.normal(0, 1, 1000)
        current = np.random.normal(3, 1, 1000)  # Shifted mean

        psi = DriftDetector._calculate_psi(baseline, current)

        assert psi > 0.2  # Should detect significant shift

    def test_calculate_psi_small_samples(self):
        """PSI should return 0 for very small samples."""
        from src.analysis.drift_detection import DriftDetector

        psi = DriftDetector._calculate_psi(np.array([1, 2]), np.array([3, 4]))

        assert psi == 0.0

    def test_classify_severity_high(self):
        """High PSI or very low p-value should give 'high' severity."""
        detector = self._make_detector()

        assert detector._classify_severity(psi=0.3, ks_pvalue=0.001) == "high"
        assert detector._classify_severity(psi=0.05, ks_pvalue=0.001) == "high"

    def test_classify_severity_medium(self):
        """Moderate PSI or significant p-value should give 'medium'."""
        detector = self._make_detector()

        assert detector._classify_severity(psi=0.15, ks_pvalue=0.1) == "medium"
        assert detector._classify_severity(psi=0.05, ks_pvalue=0.03) == "medium"

    def test_classify_severity_none(self):
        """Low PSI and high p-value should give 'none'."""
        detector = self._make_detector()

        assert detector._classify_severity(psi=0.01, ks_pvalue=0.5) == "none"

    def test_detect_feature_drift_no_drift(self):
        """Should detect no drift when distributions are drawn from the same source."""
        # Use large windows so PSI has enough samples per bin to be stable
        detector = self._make_detector(baseline_window=500, test_window=250)

        np.random.seed(123)
        n = 1000
        end = date.today()
        returns = np.random.normal(0, 0.01, n)

        df = pd.DataFrame({
            "date": [end - timedelta(days=n - i) for i in range(n)],
            "log_return": returns,
        })

        results = detector.detect_feature_drift(df, end)

        # With data drawn from same distribution and large samples, no drift
        high_drifted = [f for f in results if f.severity == "high"]
        assert len(high_drifted) == 0

    def test_detect_feature_drift_with_shift(self):
        """Should detect drift when test window has shifted distributions."""
        detector = self._make_detector(baseline_window=30, test_window=10)

        np.random.seed(42)
        n = 50
        end = date.today()

        # Create data with a clear shift in the last 10 days
        log_returns = np.concatenate([
            np.random.normal(0, 0.01, 40),   # baseline
            np.random.normal(0.05, 0.03, 10), # shifted test period
        ])

        df = pd.DataFrame({
            "date": [end - timedelta(days=n - i) for i in range(n)],
            "log_return": log_returns,
        })

        results = detector.detect_feature_drift(df, end)

        # Should detect drift in log_return
        log_return_drift = [f for f in results if f.feature_name == "log_return"]
        assert len(log_return_drift) == 1
        assert log_return_drift[0].drift_detected

    def test_detect_feature_drift_empty_df(self):
        """Should return empty list for empty DataFrame."""
        detector = self._make_detector()
        results = detector.detect_feature_drift(pd.DataFrame(), date.today())
        assert results == []

    @patch("src.analysis.drift_detection.get_session")
    @patch("src.analysis.drift_detection.get_market_data")
    @patch("src.analysis.drift_detection.get_events_by_date_range")
    def test_analyze_returns_report(self, mock_events, mock_market, mock_session):
        """analyze should return a DriftReport."""
        from tests.conftest import make_market_series, make_event_series

        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        mock_market.return_value = make_market_series(100, "SPY")
        mock_events.return_value = make_event_series(100)

        detector = self._make_detector(baseline_window=30, test_window=14)
        report = detector.analyze(
            "SPY",
            date.today() - timedelta(days=100),
            date.today(),
        )

        assert report is not None
        assert report.symbol == "SPY"
        assert isinstance(report.drift_detected, bool)
        assert report.total_features > 0
        assert report.overall_severity in ("none", "low", "medium", "high")
        assert len(report.summary) > 0


class TestFeatureDrift:
    """Tests for the FeatureDrift dataclass."""

    def test_feature_drift_creation(self):
        from src.analysis.drift_detection import FeatureDrift

        fd = FeatureDrift(
            feature_name="log_return",
            drift_detected=True,
            baseline_mean=0.001,
            baseline_std=0.01,
            current_mean=0.05,
            current_std=0.03,
            ks_statistic=0.45,
            ks_pvalue=0.001,
            psi_value=0.35,
            severity="high",
        )

        assert fd.feature_name == "log_return"
        assert fd.drift_detected is True
        assert fd.severity == "high"
