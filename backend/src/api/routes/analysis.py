"""
Analysis API Router.

Endpoints for querying analysis results.

USAGE:
------
    GET /api/analysis/results - List event study results
    GET /api/analysis/anomalies - List detected anomalies
    GET /api/analysis/significant - List significant results
    POST /api/analysis/event-study - Run event study on demand
    GET /api/analysis/regression/{symbol} - Run regression analysis
    GET /api/analysis/anomalies/detect - Run production anomaly detection
"""

from datetime import date, timedelta
import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func

from src.db.connection import get_session
from src.db.models import Event, AnalysisResult
from src.config.constants import CAMEO_CATEGORIES, get_event_group
from src.api.schemas import (
    AnalysisResultResponse,
    AnomalyDetectionResponse,
    AnomalyReportResponse,
    EventStudyRequest,
    EventStudyResponse,
    RegressionResponse,
)

logger = logging.getLogger(__name__)

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


# =============================================================================
# Event Study, Regression, Anomaly Detection
# =============================================================================

@router.post("/event-study", response_model=EventStudyResponse)
def run_event_study(request: EventStudyRequest):
    """
    Run an event study to measure a single event's impact on a market symbol.

    Calculates Cumulative Abnormal Returns (CAR), t-statistics, p-values,
    and Wilcoxon signed-rank test results.
    """
    try:
        from src.analysis.production_event_study import ProductionEventStudy

        study = ProductionEventStudy()
        result = study.analyze_event(
            request.event_id, request.symbol, request.event_date
        )

        if result is None:
            raise HTTPException(
                status_code=422,
                detail=f"Insufficient data for event study on {request.symbol} at {request.event_date}",
            )

        return EventStudyResponse(
            event_id=result.event_id,
            symbol=result.symbol,
            event_date=result.event_date,
            car=result.car,
            car_percent=result.car_percent,
            t_statistic=result.t_statistic,
            p_value=result.p_value,
            is_significant=result.is_significant,
            ci_lower=result.ci_lower,
            ci_upper=result.ci_upper,
            expected_return=result.expected_return,
            actual_return=result.actual_return,
            std_dev=result.std_dev,
            wilcoxon_p=result.wilcoxon_p,
            estimation_days=result.estimation_days,
            event_days=result.event_days,
            summary=result.summary,
        )

    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Production event study module not available.",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Event study failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/regression/{symbol}", response_model=RegressionResponse)
def run_regression(
    symbol: str,
    start_date: date = Query(None, description="Start date (default: 365 days ago)"),
    end_date: date = Query(None, description="End date (default: today)"),
):
    """
    Run OLS regression analysis for a symbol using statsmodels.

    Analyzes the relationship between geopolitical event features
    (Goldstein scale, media mentions, tone, conflict count) and
    market returns.

    Returns coefficients, p-values, R-squared, confidence intervals,
    and the full statsmodels summary.
    """
    try:
        from src.analysis.production_regression import ProductionRegression

        if end_date is None:
            end_date = date.today()
        if start_date is None:
            start_date = end_date - timedelta(days=365)

        regression = ProductionRegression()
        result = regression.analyze(symbol, start_date, end_date)

        if result is None:
            raise HTTPException(
                status_code=422,
                detail=f"Insufficient data for regression on {symbol}",
            )

        return RegressionResponse(
            symbol=result.symbol,
            r_squared=result.r_squared,
            adj_r_squared=result.adj_r_squared,
            f_statistic=result.f_statistic,
            f_pvalue=result.f_pvalue,
            coefficients=result.coefficients,
            std_errors=result.std_errors,
            t_values=result.t_values,
            p_values=result.p_values,
            conf_int_lower=result.conf_int_lower,
            conf_int_upper=result.conf_int_upper,
            n_observations=result.n_observations,
            n_features=result.n_features,
            summary=result.summary,
        )

    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Production regression module not available. Install statsmodels.",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Regression failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/anomalies/detect", response_model=AnomalyReportResponse)
def detect_anomalies(
    symbol: str = Query(..., description="Market symbol to analyze"),
    days: int = Query(90, ge=7, le=365, description="Number of days to analyze"),
):
    """
    Run production anomaly detection for a symbol using Isolation Forest,
    Z-score analysis, and event-mismatch detection.

    Returns a full anomaly report with detected anomalies ranked by
    anomaly probability.
    """
    try:
        from src.analysis.production_anomaly import ProductionAnomalyDetector

        end_date = date.today()
        start_date = end_date - timedelta(days=days)

        detector = ProductionAnomalyDetector()
        anomalies = detector.detect_all(symbol, start_date, end_date)
        report = detector.get_anomaly_report(
            anomalies, symbol, start_date, end_date
        )

        return AnomalyReportResponse(
            symbol=report.symbol,
            start_date=report.start_date,
            end_date=report.end_date,
            total_days=report.total_days,
            anomaly_count=report.anomaly_count,
            anomaly_rate=report.anomaly_rate,
            unexplained_moves=report.unexplained_moves,
            muted_responses=report.muted_responses,
            statistical_outliers=report.statistical_outliers,
            top_anomalies=[
                AnomalyDetectionResponse(
                    date=a.date,
                    symbol=a.symbol,
                    anomaly_type=a.anomaly_type,
                    actual_return=a.actual_return,
                    expected_return=a.expected_return,
                    z_score=a.z_score,
                    isolation_score=a.isolation_score,
                    anomaly_probability=a.anomaly_probability,
                    event_count=a.event_count,
                    avg_goldstein=a.avg_goldstein,
                    detected_by=a.detected_by,
                )
                for a in report.top_anomalies
            ],
            summary=report.summary,
        )

    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Production anomaly detection not available. Install scikit-learn.",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


