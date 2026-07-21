"""Shared FastAPI dependencies.

Single source of truth for the pieces every router needs — a database
session and the start/end date-range parsing that was previously copy-pasted
into ~10 endpoints. Import from here rather than redefining locally.
"""

from datetime import date, timedelta

from fastapi import Query

from src.db.connection import get_session


def get_db():
    """Yield a database session for the lifetime of a request.

    Wraps the ``get_session`` context manager so the session is committed on
    success and rolled back on error, then always closed. Use as::

        @router.get(...)
        def handler(db: Session = Depends(get_db)):
            ...
    """
    with get_session() as session:
        yield session


class DateRange:
    """Parse optional ``start_date``/``end_date`` query params with defaults.

    Instantiate with the default look-back window and use as a dependency::

        @router.get(...)
        def handler(dates: tuple[date, date] = Depends(DateRange(365))):
            start_date, end_date = dates

    When ``end_date`` is omitted it defaults to today; when ``start_date`` is
    omitted it defaults to ``default_days`` before ``end_date``.
    """

    def __init__(self, default_days: int = 365):
        self.default_days = default_days

    def __call__(
        self,
        start_date: date | None = Query(None, description="Start of date range (ISO 8601)"),
        end_date: date | None = Query(None, description="End of date range (ISO 8601)"),
    ) -> tuple[date, date]:
        end = end_date or date.today()
        start = start_date or (end - timedelta(days=self.default_days))
        return start, end
