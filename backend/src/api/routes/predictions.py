"""
Predictions API Router.

Endpoints for market direction predictions using:
- Level 1: Historical frequency patterns (conditional probability)
- Level 2: Logistic regression (interpretable ML)
"""

from datetime import date, timedelta
import logging
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predictions", tags=["Predictions"])


class LogisticPredictionRequest(BaseModel):
    """Request body for logistic regression prediction."""
    symbol: str = Field(..., description="Market symbol to predict")
    goldstein_mean: float = Field(0.0, description="Mean Goldstein score of today's events")
    goldstein_min: float = Field(0.0, description="Min Goldstein score")
    goldstein_max: float = Field(0.0, description="Max Goldstein score")
    mentions_total: int = Field(0, ge=0, description="Total media mentions")
    avg_tone: float = Field(0.0, description="Average media tone")
    conflict_count: int = Field(0, ge=0, description="Number of conflict events")
    cooperation_count: int = Field(0, ge=0, description="Number of cooperation events")
    training_days: int = Field(365, ge=90, le=730, description="Days of training data")


@router.post("/logistic", response_model=dict)
def predict_logistic(request: LogisticPredictionRequest):
    """
    Predict market direction using logistic regression.

    Trains on historical data up to today, then predicts using the provided
    event features. Returns prediction, probability, and feature contributions.
    """
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
        request.symbol, start_date, end_date, current_features
    )

    if result is None:
        raise HTTPException(
            status_code=422,
            detail=f"Insufficient training data for {request.symbol}",
        )

    return {
        "symbol": result.symbol,
        "prediction": result.prediction,
        "probability_up": round(result.probability_up, 4),
        "probability_down": round(1 - result.probability_up, 4),
        "accuracy": round(result.accuracy, 4),
        "n_training_samples": result.n_training_samples,
        "feature_contributions": result.feature_contributions,
        "coefficients": {k: round(v, 4) for k, v in result.coefficients.items()},
        "disclaimer": "Statistical model for educational purposes. Not financial advice.",
    }


@router.get("/logistic/{symbol}/summary", response_model=dict)
def get_model_summary(
    symbol: str,
    start_date: date | None = Query(None, description="Training start date"),
    end_date: date | None = Query(None, description="Training end date"),
):
    """
    Get logistic regression model statistics without making a prediction.

    Returns coefficients, feature importance, accuracy, and training info.
    """
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
            detail=f"Insufficient data to build model for {symbol}",
        )

    return summary
