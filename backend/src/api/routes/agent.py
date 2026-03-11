"""Agent API routes — single-agent and multi-agent chat endpoints."""

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


class MultiAgentResponse(BaseModel):
    """Response from the multi-agent intelligence graph."""
    response: str
    agents_used: list[str] = []
    iterations: int = 0


def _check_api_key():
    if not ANTHROPIC_API_KEY:
        raise HTTPException(
            status_code=503,
            detail=(
                "AI agent not configured. Set ANTHROPIC_API_KEY in your "
                ".env file to enable the AI analyst."
            ),
        )


@router.post("/chat", response_model=ChatResponse)
def agent_chat(req: ChatRequest):
    """
    Chat with the AI geopolitical market analyst (single agent).

    Uses a single Claude agent with all 15 tools.
    """
    _check_api_key()

    try:
        from src.agent.service import AgentService

        service = AgentService()
        result = service.chat(req.message, req.history or [])
        return ChatResponse(**result)
    except Exception as e:
        logger.exception("Agent chat error")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/multi", response_model=MultiAgentResponse)
def agent_chat_multi(req: ChatRequest):
    """
    Chat with the multi-agent intelligence system (LangGraph).

    Routes the query through specialized agents:
      Supervisor → Collection → Analysis → Dissemination

    Each agent has a focused tool set and system prompt.
    The Supervisor orchestrates the flow based on shared state.
    """
    _check_api_key()

    try:
        from src.agent.graph import intelligence_graph
        from langchain_core.messages import HumanMessage

        # Build initial state
        initial_state = {
            "messages": [HumanMessage(content=req.message)],
            "query": req.message,
        }

        # Run the graph
        result = intelligence_graph.invoke(initial_state)

        # Extract which agents ran from the message history
        agents_used = []
        for msg in result.get("messages", []):
            content = msg.content if hasattr(msg, "content") else str(msg)
            if "[Collection Agent]" in content and "collection" not in agents_used:
                agents_used.append("collection")
            elif "[Analysis Agent]" in content and "analysis" not in agents_used:
                agents_used.append("analysis")

        return MultiAgentResponse(
            response=result.get("final_response", "No response generated."),
            agents_used=agents_used,
            iterations=result.get("iteration", 0),
        )
    except Exception as e:
        logger.exception("Multi-agent chat error")
        raise HTTPException(status_code=500, detail=str(e))
