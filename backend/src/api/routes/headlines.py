"""
News Headlines API Router.

Endpoints for querying RSS news headlines.

USAGE:
------
    GET  /api/headlines/recent    - Recent headlines, optionally filtered by source
    POST /api/headlines/score     - Trigger sentiment scoring on unprocessed headlines
"""

from datetime import date, timedelta
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from src.db.connection import get_session
from src.db.queries import get_headlines_by_date_range
from src.analysis.sentiment import score_unprocessed_headlines
from src.api.schemas import HeadlineResponse

router = APIRouter(prefix="/headlines", tags=["Headlines"])


def get_db():
    """Dependency to get database session."""
    with get_session() as session:
        yield session


@router.get("/recent", response_model=list[HeadlineResponse])
def recent_headlines(
    source: str | None = Query(None, description="Filter by source: reuters, ap, bbc, aljazeera"),
    days_back: int = Query(2, ge=1, le=30, description="How many days of headlines to return"),
    limit: int = Query(100, ge=1, le=500, description="Maximum headlines to return"),
    db: Session = Depends(get_db),
):
    """
    Get recent news headlines from RSS feeds.

    Returns headlines from the last N days, newest first.
    Optionally filter by source (reuters, ap, bbc, aljazeera).
    """
    end_date = date.today()
    start_date = end_date - timedelta(days=days_back)

    headlines = get_headlines_by_date_range(db, start_date, end_date, source=source)
    return headlines[:limit]


@router.post("/score")
def trigger_sentiment_scoring(db: Session = Depends(get_db)):
    """
    Score sentiment on all unprocessed headlines.

    Runs the sentiment model on headlines that don't have scores yet.
    First call may be slow (~5s) as the model downloads/loads.
    """
    n_scored = score_unprocessed_headlines(db)
    return {"headlines_scored": n_scored}
