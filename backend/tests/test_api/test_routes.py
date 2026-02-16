"""Tests for API routes."""

import sys
from pathlib import Path
from datetime import date, timedelta
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi not installed")

    # Mock the database before importing the app
    with patch("src.db.connection.engine"), \
         patch("src.db.connection.SessionLocal"), \
         patch("src.db.connection.get_session") as mock_session:

        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        from src.api.main import app
        with TestClient(app) as c:
            yield c


class TestAnalysisRoutes:
    """Tests for analysis API endpoints."""

    def test_health_endpoint(self, client):
        """Health endpoint should return 200."""
        response = client.get("/health")
        assert response.status_code == 200

    @patch("src.api.routes.analysis.get_session")
    def test_get_results_endpoint(self, mock_session, client):
        """GET /api/analysis/results should return list."""
        session = MagicMock()
        mock_session.return_value.__enter__ = MagicMock(return_value=session)
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        # Mock query returns empty list
        session.query.return_value.join.return_value.filter.return_value \
            .order_by.return_value.limit.return_value.all.return_value = []

        response = client.get("/api/analysis/results")
        assert response.status_code == 200

    @patch("src.api.routes.analysis.get_session")
    def test_get_anomalies_endpoint(self, mock_session, client):
        """GET /api/analysis/anomalies should return list."""
        session = MagicMock()
        mock_session.return_value.__enter__ = MagicMock(return_value=session)
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        session.query.return_value.filter.return_value \
            .order_by.return_value.limit.return_value.all.return_value = []

        response = client.get("/api/analysis/anomalies")
        assert response.status_code == 200


class TestSchemas:
    """Tests for Pydantic request/response schemas."""

    def test_logistic_prediction_request_validation(self):
        """LogisticPredictionRequest should validate fields."""
        from src.api.schemas import LogisticPredictionRequest

        req = LogisticPredictionRequest(
            symbol="SPY",
            goldstein_mean=-5.0,
            mentions_total=50,
            conflict_count=3,
        )

        assert req.symbol == "SPY"
        assert req.goldstein_mean == -5.0

    def test_logistic_prediction_response_creation(self):
        """LogisticPredictionResponse should accept all fields."""
        from src.api.schemas import LogisticPredictionResponse

        resp = LogisticPredictionResponse(
            symbol="SPY",
            prediction="DOWN",
            probability_up=0.28,
            probability_down=0.72,
            accuracy=0.54,
            n_training_samples=200,
            feature_contributions=[],
            coefficients={"goldstein_mean": -0.15},
        )

        assert resp.prediction == "DOWN"
        assert resp.accuracy == 0.54
        assert resp.probability_down == 0.72

    def test_regression_response_creation(self):
        """RegressionResponse should accept all fields."""
        from src.api.schemas import RegressionResponse

        resp = RegressionResponse(
            symbol="SPY",
            r_squared=0.05,
            adj_r_squared=0.03,
            f_statistic=2.5,
            f_pvalue=0.04,
            coefficients={"goldstein_mean": 0.001},
            std_errors={"goldstein_mean": 0.0005},
            t_values={"goldstein_mean": 2.0},
            p_values={"goldstein_mean": 0.03},
            conf_int_lower={"goldstein_mean": 0.0001},
            conf_int_upper={"goldstein_mean": 0.002},
            n_observations=100,
            n_features=1,
            summary="Regression analysis for SPY",
        )

        assert resp.r_squared == 0.05
        assert "goldstein_mean" in resp.coefficients
