"""Agent API route — chat endpoint for the AI analyst."""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.config.settings import ANTHROPIC_API_KEY

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["Agent"])


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


class ChatResponse(BaseModel):
    response: str
    tool_calls: list[dict] = []
    model: str = ""


@router.post("/chat", response_model=ChatResponse)
def agent_chat(req: ChatRequest):
    """
    Chat with the AI geopolitical market analyst.

    The agent uses Claude with tool use to query events, correlations,
    patterns, predictions, and anomalies from the database.

    Returns 503 if ANTHROPIC_API_KEY is not configured.
    """
    if not ANTHROPIC_API_KEY:
        raise HTTPException(
            status_code=503,
            detail=(
                "AI agent not configured. Set ANTHROPIC_API_KEY in your "
                ".env file to enable the AI analyst."
            ),
        )

    try:
        from src.agent.service import AgentService

        service = AgentService()
        result = service.chat(req.message, req.history or [])
        return ChatResponse(**result)
    except Exception as e:
        logger.exception("Agent chat error")
        raise HTTPException(status_code=500, detail=str(e))
