"""
Briefing API Router.

Endpoints for AI-generated intelligence summaries using RAG.

USAGE:
------
    GET /api/briefing/summary - Generate AI situational summary
"""

import logging

from fastapi import APIRouter, Query

from src.config.settings import OLLAMA_MODEL
from src.rag.context import ContextBuilder
from src.analysis.synthesis import generate_briefing, is_available

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/briefing", tags=["Briefing"])


@router.get("/summary")
def get_briefing_summary(
    days_back: int = Query(3, ge=1, le=30, description="Days of data to analyze"),
):
    """
    Generate an AI situational awareness summary using RAG.

    1. Retrieves recent headlines and events across 5 geopolitical themes
    2. Feeds them as context to a locally-hosted Llama model via Ollama
    3. Returns a structured intelligence briefing summary
    """
    builder = ContextBuilder()
    context = builder.build_briefing_context(days_back=days_back)

    if not context:
        return {
            "summary": "Insufficient data for briefing. Run the ingestion pipeline to populate headlines and events.",
            "source": "system",
        }

    if not is_available():
        return {
            "summary": context,
            "source": "rag_context",
            "note": "Ollama model not available. Showing raw RAG context.",
        }

    summary = generate_briefing(context)

    return {
        "summary": summary,
        "source": "ai_generated",
        "model": OLLAMA_MODEL,
    }
