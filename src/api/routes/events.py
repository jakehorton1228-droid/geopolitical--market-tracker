"""
Events API Router.

Endpoints for querying geopolitical events from GDELT.

USAGE:
------
    GET /api/events - List events with filters
    GET /api/events/{id} - Get single event by ID
    GET /api/events/countries - Get event counts by country
    GET /api/events/types - Get event counts by type
"""

from datetime import date, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func

from src.db.connection import get_session
from src.db.models import Event
from src.config.constants import EVENT_GROUPS, CAMEO_CATEGORIES, get_event_group
from src.api.schemas import EventResponse, EventQuery

router = APIRouter(prefix="/events", tags=["Events"])


def get_db():
    """Dependency to get database session."""
    with get_session() as session:
        yield session


@router.get("", response_model=list[EventResponse])
def list_events(
    start_date: date | None = Query(None, description="Filter from date"),
    end_date: date | None = Query(None, description="Filter to date"),
    country_code: str | None = Query(None, description="3-letter ISO country code", max_length=3),
    event_root_codes: str | None = Query(None, description="Comma-separated CAMEO codes (e.g., '18,19,20')"),
    event_group: str | None = Query(None, description="Event group: violent_conflict, material_conflict, etc."),
    min_goldstein: float | None = Query(None, description="Minimum absolute Goldstein score"),
    min_mentions: int | None = Query(None, description="Minimum media mentions"),
    limit: int = Query(100, ge=1, le=1000, description="Max results"),
    offset: int = Query(0, ge=0, description="Skip results"),
    db: Session = Depends(get_db),
):
    """
    List geopolitical events with optional filters.

    **Examples:**
    - Get recent conflict events: `?event_group=violent_conflict&min_mentions=10`
    - Get events in Russia: `?country_code=RUS`
    - Get high-impact events: `?min_goldstein=5&min_mentions=20`
    """
    # Default date range if not specified
    if not end_date:
        end_date = date.today()
    if not start_date:
        start_date = end_date - timedelta(days=30)

    query = db.query(Event).filter(
        Event.event_date >= start_date,
        Event.event_date <= end_date,
    )

    # Filter by country
    if country_code:
        query = query.filter(
            (Event.actor1_country_code == country_code.upper()) |
            (Event.actor2_country_code == country_code.upper()) |
            (Event.action_geo_country_code == country_code.upper())
        )

    # Filter by event codes
    if event_root_codes:
        codes = [c.strip() for c in event_root_codes.split(",")]
        query = query.filter(Event.event_root_code.in_(codes))
    elif event_group:
        # Map group to codes
        codes = EVENT_GROUPS.get(event_group, [])
        if codes:
            query = query.filter(Event.event_root_code.in_(codes))

    # Filter by Goldstein scale
    if min_goldstein is not None:
        query = query.filter(func.abs(Event.goldstein_scale) >= min_goldstein)

    # Filter by mentions
    if min_mentions is not None:
        query = query.filter(Event.num_mentions >= min_mentions)

    # Order and paginate
    events = query.order_by(
        Event.event_date.desc(),
        Event.num_mentions.desc(),
    ).offset(offset).limit(limit).all()

    return events


@router.get("/count", response_model=dict)
def count_events(
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
    db: Session = Depends(get_db),
):
    """Get total count of events in date range."""
    if not end_date:
        end_date = date.today()
    if not start_date:
        start_date = end_date - timedelta(days=30)

    count = db.query(func.count(Event.id)).filter(
        Event.event_date >= start_date,
        Event.event_date <= end_date,
    ).scalar()

    return {"count": count, "start_date": start_date, "end_date": end_date}


@router.get("/by-country", response_model=list[dict])
def events_by_country(
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Get event counts grouped by country."""
    if not end_date:
        end_date = date.today()
    if not start_date:
        start_date = end_date - timedelta(days=30)

    results = db.query(
        Event.action_geo_country_code,
        func.count(Event.id).label("count"),
    ).filter(
        Event.event_date >= start_date,
        Event.event_date <= end_date,
        Event.action_geo_country_code.isnot(None),
        Event.action_geo_country_code != "",
    ).group_by(
        Event.action_geo_country_code,
    ).order_by(
        func.count(Event.id).desc(),
    ).limit(limit).all()

    return [{"country_code": r[0], "count": r[1]} for r in results]


@router.get("/by-type", response_model=list[dict])
def events_by_type(
    start_date: date | None = Query(None),
    end_date: date | None = Query(None),
    db: Session = Depends(get_db),
):
    """Get event counts grouped by CAMEO type."""
    if not end_date:
        end_date = date.today()
    if not start_date:
        start_date = end_date - timedelta(days=30)

    results = db.query(
        Event.event_root_code,
        func.count(Event.id).label("count"),
    ).filter(
        Event.event_date >= start_date,
        Event.event_date <= end_date,
    ).group_by(
        Event.event_root_code,
    ).order_by(
        Event.event_root_code,
    ).all()

    return [
        {
            "code": r[0],
            "name": CAMEO_CATEGORIES.get(str(r[0]).zfill(2), "Unknown"),
            "group": get_event_group(r[0]),
            "count": r[1],
        }
        for r in results
    ]


@router.get("/{event_id}", response_model=EventResponse)
def get_event(
    event_id: int,
    db: Session = Depends(get_db),
):
    """Get a single event by ID."""
    event = db.query(Event).filter(Event.id == event_id).first()

    if not event:
        raise HTTPException(status_code=404, detail=f"Event {event_id} not found")

    return event
