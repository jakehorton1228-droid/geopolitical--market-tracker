"""
Predictions API Router.

Serves market direction predictions from the champion ML model (loaded
from MLflow) with a fallback to the legacy logistic regression analyzer
when no champion is registered yet.

Flow:
  1. Try to load the champion model from MLflow
  2. If champion exists -> use it for predictions and summaries
  3. If no champion -> fall back to LogisticRegressionAnalyzer (trains
     on demand from the request parameters)

This ensures the Signals page keeps working before the first training
run, then automatically picks up the champion once `make train` completes.
"""

from datetime import date, timedelta
import logging
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.ml.model_loader import get_champion_model

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predictions", tags=["Predictions"])


class LogisticPredictionRequest(BaseModel):
    """Request body for market direction prediction.

    The name is legacy — this endpoint now routes to the champion model
    (XGBoost, LightGBM, etc.) when one is registered. The request schema
    stays the same so the frontend doesn't need to change.
    """
    symbol: str = Field(..., description="Market symbol to predict")
    goldstein_mean: float = Field(0.0, description="Mean Goldstein score of today's events")
    goldstein_min: float = Field(0.0, description="Min Goldstein score")
    goldstein_max: float = Field(0.0, description="Max Goldstein score")
    mentions_total: int = Field(0, ge=0, description="Total media mentions")
    avg_tone: float = Field(0.0, description="Average media tone")
    conflict_count: int = Field(0, ge=0, description="Number of conflict events")
    cooperation_count: int = Field(0, ge=0, description="Number of cooperation events")
    training_days: int = Field(365, ge=90, le=730, description="Days of training data (legacy model only)")


@router.post("/logistic", response_model=dict)
def predict(request: LogisticPredictionRequest):
    """
    Predict market direction.

    Uses the champion model from MLflow if one is registered, otherwise
    falls back to training a logistic regression on demand.
    """
    # Try the champion model first
    champion = get_champion_model()
    if champion is not None:
        return _predict_with_champion(champion, request)

    # Fallback to legacy logistic regression
    return _predict_with_legacy(request)


@router.get("/logistic/{symbol}/summary", response_model=dict)
def get_model_summary(
    symbol: str,
    start_date: date | None = Query(None, description="Training start date (legacy model only)"),
    end_date: date | None = Query(None, description="Training end date (legacy model only)"),
):
    """
    Get metadata about the currently-serving model.

    Returns champion metadata if a champion is registered, otherwise
    returns the legacy logistic regression summary trained on demand.
    """
    champion = get_champion_model()
    if champion is not None:
        meta = champion.metadata()
        return {
            "symbol": symbol,
            "model_name": meta["model_name"],
            "model_source": "champion",
            "version": meta["version"],
            "auc": meta["auc"],
            "n_features": meta["n_features"],
            "feature_names": meta["feature_names"],
            "note": "Model trained across all tracked symbols; predictions are symbol-agnostic.",
        }

    # Fallback to legacy
    from src.analysis.production_regression import LogisticRegressionAnalyzer

    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = date(2016, 1, 1)

    analyzer = LogisticRegressionAnalyzer()
    summary = analyzer.get_model_summary(symbol, start_date, end_date)

    if summary is None:
        raise HTTPException(
            status_code=422,
            detail=f"Insufficient data to build legacy model for {symbol}",
        )

    # Tag the response so the frontend knows it's using the legacy model
    summary["model_name"] = "Logistic Regression (legacy)"
    summary["model_source"] = "legacy"
    return summary


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _predict_with_champion(champion, request: LogisticPredictionRequest) -> dict:
    """Route a prediction request to the MLflow champion model."""
    # The champion was trained on 27 features; the request only carries 7.
    # The remaining features default to 0 and the scaler handles them.
    features = {
        "goldstein_mean": request.goldstein_mean,
        "goldstein_min": request.goldstein_min,
        "goldstein_max": request.goldstein_max,
        "mentions_total": float(request.mentions_total),
        "avg_tone": request.avg_tone,
        "conflict_count": float(request.conflict_count),
        "cooperation_count": float(request.cooperation_count),
    }

    result = champion.predict(features)
    result["symbol"] = request.symbol
    result["probability_down"] = round(1 - result["probability_up"], 4)
    result["disclaimer"] = "ML model for educational purposes. Not financial advice."
    return result


def _predict_with_legacy(request: LogisticPredictionRequest) -> dict:
    """Fall back to training a logistic regression on demand."""
    from src.analysis.production_regression import LogisticRegressionAnalyzer

    end_date = date.today()
    start_date = end_date - timedelta(days=request.training_days)

    current_features = {
        "goldstein_mean": request.goldstein_mean,
        "goldstein_min": request.goldstein_min,
        "goldstein_max": request.goldstein_max,
        "mentions_total": request.mentions_total,
        "avg_tone": request.avg_tone,
        "conflict_count": request.conflict_count,
        "cooperation_count": request.cooperation_count,
    }

    analyzer = LogisticRegressionAnalyzer()
    result = analyzer.train_and_predict(
        request.symbol, start_date, end_date, current_features,
    )

    if result is None:
        raise HTTPException(
            status_code=422,
            detail=f"Insufficient training data for {request.symbol}",
        )

    return {
        "symbol": result.symbol,
        "model_name": "Logistic Regression (legacy)",
        "model_source": "legacy",
        "prediction": result.prediction,
        "probability_up": round(result.probability_up, 4),
        "probability_down": round(1 - result.probability_up, 4),
        "accuracy": round(result.accuracy, 4),
        "n_training_samples": result.n_training_samples,
        "feature_contributions": result.feature_contributions,
        "coefficients": {k: round(v, 4) for k, v in result.coefficients.items()},
        "disclaimer": "Statistical model for educational purposes. Not financial advice.",
    }
