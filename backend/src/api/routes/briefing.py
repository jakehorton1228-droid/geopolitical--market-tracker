"""
Briefing API Router.

Endpoints for intelligence briefing context using RAG.

USAGE:
------
    GET /api/briefing/summary - Get RAG-retrieved situational context
"""

import logging

from fastapi import APIRouter, Query

from src.rag.context import ContextBuilder

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/briefing", tags=["Briefing"])


@router.get("/summary")
def get_briefing_summary(
    days_back: int = Query(3, ge=1, le=30, description="Days of data to analyze"),
):
    """
    Get a situational awareness briefing context using RAG.

    Retrieves recent headlines and events across 5 geopolitical themes
    and returns structured intelligence context.
    """
    builder = ContextBuilder()
    context = builder.build_briefing_context(days_back=days_back)

    if not context:
        return {
            "summary": "Insufficient data for briefing. Run the ingestion pipeline to populate headlines and events.",
            "source": "system",
        }

    return {
        "summary": context,
        "source": "rag_context",
    }
