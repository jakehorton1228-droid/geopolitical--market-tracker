"""
Analysis API Router.

Endpoints for querying analysis results and making predictions.

USAGE:
------
    GET /api/analysis/results - List event study results
    GET /api/analysis/anomalies - List detected anomalies
    GET /api/analysis/significant - List significant results
    POST /api/analysis/predict - Predict market direction
"""

from datetime import date, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
import numpy as np

from src.db.connection import get_session
from src.db.models import Event, AnalysisResult
from src.config.constants import CAMEO_CATEGORIES, get_event_group
from src.api.schemas import (
    AnalysisResultResponse,
    AnomalyResponse,
    PredictionRequest,
    PredictionResponse,
)

router = APIRouter(prefix="/analysis", tags=["Analysis"])


def get_db():
    """Dependency to get database session."""
    with get_session() as session:
        yield session


@router.get("/results", response_model=list[AnalysisResultResponse])
def list_analysis_results(
    event_id: int | None = Query(None, description="Filter by event ID"),
    symbol: str | None = Query(None, description="Filter by symbol"),
    analysis_type: str | None = Query(None, description="event_study or anomaly_detection"),
    is_significant: bool | None = Query(None, description="Only significant results"),
    min_car: float | None = Query(None, description="Minimum absolute CAR"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """
    List analysis results with optional filters.

    **Examples:**
    - Get significant results: `?is_significant=true`
    - Get results for oil: `?symbol=CL=F`
    - Get event study results: `?analysis_type=event_study`
    """
    query = db.query(AnalysisResult)

    if event_id:
        query = query.filter(AnalysisResult.event_id == event_id)

    if symbol:
        query = query.filter(AnalysisResult.symbol == symbol)

    if analysis_type:
        query = query.filter(AnalysisResult.analysis_type == analysis_type)

    if is_significant is not None:
        query = query.filter(AnalysisResult.is_significant == is_significant)

    if min_car is not None:
        query = query.filter(func.abs(AnalysisResult.car) >= min_car)

    results = query.order_by(
        func.abs(AnalysisResult.car).desc()
    ).offset(offset).limit(limit).all()

    return results


@router.get("/anomalies", response_model=list[dict])
def list_anomalies(
    anomaly_type: str | None = Query(None, description="unexplained_move, muted_response, etc."),
    symbol: str | None = Query(None, description="Filter by symbol"),
    min_score: float | None = Query(None, description="Minimum anomaly score"),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """
    List detected market anomalies.

    **Anomaly Types:**
    - `unexplained_move`: Large market move without major event
    - `muted_response`: Major event with surprisingly small reaction

    **Examples:**
    - Get all anomalies: `/api/analysis/anomalies`
    - Get unexplained moves: `?anomaly_type=unexplained_move`
    """
    query = db.query(AnalysisResult, Event).join(Event).filter(
        AnalysisResult.is_anomaly == True,
    )

    if anomaly_type:
        query = query.filter(AnalysisResult.anomaly_type == anomaly_type)

    if symbol:
        query = query.filter(AnalysisResult.symbol == symbol)

    if min_score is not None:
        query = query.filter(AnalysisResult.anomaly_score >= min_score)

    results = query.order_by(
        AnalysisResult.anomaly_score.desc()
    ).limit(limit).all()

    return [
        {
            "id": r.AnalysisResult.id,
            "event_id": r.Event.id,
            "event_date": r.Event.event_date,
            "event_type": CAMEO_CATEGORIES.get(str(r.Event.event_root_code).zfill(2), r.Event.event_root_code),
            "event_group": get_event_group(r.Event.event_root_code),
            "symbol": r.AnalysisResult.symbol,
            "anomaly_type": r.AnalysisResult.anomaly_type,
            "anomaly_score": r.AnalysisResult.anomaly_score,
            "expected_return": r.AnalysisResult.expected_return,
            "actual_return": r.AnalysisResult.actual_return,
            "goldstein_scale": r.Event.goldstein_scale,
            "actor1": r.Event.actor1_name or r.Event.actor1_code,
            "actor2": r.Event.actor2_name or r.Event.actor2_code,
        }
        for r in results
    ]


@router.get("/significant", response_model=list[dict])
def list_significant_results(
    symbol: str | None = Query(None),
    min_car: float | None = Query(0.01, description="Minimum absolute CAR (default 1%)"),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """
    List statistically significant event study results.

    Only returns results where p-value < 0.05.
    """
    query = db.query(AnalysisResult, Event).join(Event).filter(
        AnalysisResult.is_significant == True,
        AnalysisResult.analysis_type == "event_study",
    )

    if symbol:
        query = query.filter(AnalysisResult.symbol == symbol)

    if min_car:
        query = query.filter(func.abs(AnalysisResult.car) >= min_car)

    results = query.order_by(
        func.abs(AnalysisResult.car).desc()
    ).limit(limit).all()

    return [
        {
            "event_id": r.Event.id,
            "event_date": r.Event.event_date,
            "event_type": CAMEO_CATEGORIES.get(str(r.Event.event_root_code).zfill(2), r.Event.event_root_code),
            "symbol": r.AnalysisResult.symbol,
            "car_pct": r.AnalysisResult.car * 100 if r.AnalysisResult.car else 0,
            "t_stat": r.AnalysisResult.car_t_stat,
            "p_value": r.AnalysisResult.car_p_value,
            "goldstein_scale": r.Event.goldstein_scale,
            "num_mentions": r.Event.num_mentions,
        }
        for r in results
    ]


@router.get("/summary", response_model=dict)
def analysis_summary(db: Session = Depends(get_db)):
    """Get summary statistics of all analysis results."""
    total = db.query(func.count(AnalysisResult.id)).scalar() or 0
    significant = db.query(func.count(AnalysisResult.id)).filter(
        AnalysisResult.is_significant == True
    ).scalar() or 0
    anomalies = db.query(func.count(AnalysisResult.id)).filter(
        AnalysisResult.is_anomaly == True
    ).scalar() or 0

    # Anomaly breakdown
    anomaly_types = db.query(
        AnalysisResult.anomaly_type,
        func.count(AnalysisResult.id),
    ).filter(
        AnalysisResult.is_anomaly == True,
    ).group_by(
        AnalysisResult.anomaly_type
    ).all()

    return {
        "total_results": total,
        "significant_results": significant,
        "significance_rate": significant / total if total > 0 else 0,
        "total_anomalies": anomalies,
        "anomaly_breakdown": {t: c for t, c in anomaly_types if t},
    }


@router.post("/predict", response_model=PredictionResponse)
def predict_market_direction(request: PredictionRequest):
    """
    Predict market direction based on event characteristics.

    **Input:**
    - `symbol`: Market to predict (e.g., CL=F, SPY)
    - `goldstein_scale`: Event severity (-10 to +10)
    - `num_mentions`: Media coverage
    - `avg_tone`: Media sentiment
    - `is_violent_conflict`: Whether event involves violence

    **Output:**
    - `prediction`: UP or DOWN
    - `probability_up`: Probability of upward move
    - `confidence`: How confident the model is

    **Note:** This is a demonstration model using heuristics.
    In production, this would load a trained ML model.
    """
    # Simple heuristic-based prediction (demo)
    # In production, this would load a trained model
    base_prob = 0.5

    # Goldstein effect: negative = down, positive = up
    goldstein_effect = request.goldstein_scale * 0.03
    base_prob += goldstein_effect

    # Tone effect
    tone_effect = request.avg_tone * 0.02
    base_prob += tone_effect

    # Conflict tends to push markets down
    if request.is_violent_conflict:
        base_prob -= 0.1

    # More mentions = stronger signal
    if request.num_mentions > 50:
        base_prob += goldstein_effect * 0.3
    if request.num_mentions > 100:
        base_prob += goldstein_effect * 0.2

    # Symbol-specific adjustments
    safe_havens = ["GC=F", "TLT", "^VIX"]  # Gold, bonds, VIX
    if request.symbol in safe_havens and request.goldstein_scale < 0:
        # Safe havens go up during conflict
        base_prob = 1 - base_prob

    # Clamp probability
    probability = max(0.05, min(0.95, base_prob))

    # Add small noise for realism
    probability += np.random.normal(0, 0.02)
    probability = max(0.05, min(0.95, probability))

    prediction = "UP" if probability > 0.5 else "DOWN"
    confidence = abs(probability - 0.5) * 2  # 0 = uncertain, 1 = very confident

    return PredictionResponse(
        symbol=request.symbol,
        prediction=prediction,
        probability_up=round(probability, 4),
        probability_down=round(1 - probability, 4),
        confidence=round(confidence, 4),
        model_version="demo-v1",
        disclaimer="This is a demonstration model using heuristics. Not financial advice.",
    )
