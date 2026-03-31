"""
Briefing API Router.

Endpoints for AI-generated intelligence summaries using RAG.

USAGE:
------
    GET /api/briefing/summary - Generate AI situational summary
"""

import logging

from fastapi import APIRouter, Query, HTTPException

from src.config.settings import ANTHROPIC_API_KEY, AGENT_MODEL, AGENT_MAX_TOKENS
from src.rag.context import ContextBuilder

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/briefing", tags=["Briefing"])


@router.get("/summary")
def get_briefing_summary(
    days_back: int = Query(3, ge=1, le=30, description="Days of data to analyze"),
):
    """
    Generate an AI situational awareness summary using RAG.

    1. Retrieves recent headlines and events across 5 geopolitical themes
    2. Feeds them as context to Claude
    3. Returns a structured intelligence briefing summary

    Returns a pre-built RAG context if no API key is configured,
    or a full AI-generated summary if the key is available.
    """
    builder = ContextBuilder()
    context = builder.build_briefing_context(days_back=days_back)

    if not context:
        return {
            "summary": "Insufficient data for briefing. Run the ingestion pipeline to populate headlines and events.",
            "source": "system",
        }

    # If no API key, return the raw context as a structured summary
    if not ANTHROPIC_API_KEY:
        return {
            "summary": context,
            "source": "rag_context",
            "note": "Set ANTHROPIC_API_KEY for AI-generated summaries.",
        }

    # Generate AI summary using Claude
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        response = client.messages.create(
            model=AGENT_MODEL,
            max_tokens=1024,
            system=(
                "You are an intelligence analyst writing a daily situational awareness briefing. "
                "You have been given relevant headlines and events retrieved from a database. "
                "Write a concise briefing (3-5 paragraphs) covering: "
                "1) Top threats and conflicts, "
                "2) Diplomatic developments, "
                "3) Economic/market implications, "
                "4) Key watchpoints for the next 24-48 hours. "
                "Be factual, cite specific events from the context, and avoid speculation. "
                "Write in a professional intelligence briefing style."
            ),
            messages=[
                {
                    "role": "user",
                    "content": f"Generate a situational awareness briefing based on this intelligence:\n\n{context}",
                }
            ],
        )

        summary = response.content[0].text

        return {
            "summary": summary,
            "source": "ai_generated",
            "model": AGENT_MODEL,
        }

    except Exception as e:
        logger.error(f"AI summary generation failed: {e}")
        # Fallback to raw context
        return {
            "summary": context,
            "source": "rag_context_fallback",
            "error": str(e),
        }
