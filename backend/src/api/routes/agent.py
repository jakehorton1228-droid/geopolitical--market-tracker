"""Agent API route — multi-agent intelligence pipeline (LangGraph)."""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.analysis.synthesis import is_available

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["Agent"])


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


class ChatResponse(BaseModel):
    """Response from the multi-agent intelligence graph."""
    response: str
    agents_used: list[str] = []
    iterations: int = 0


def _check_ollama():
    if not is_available():
        raise HTTPException(
            status_code=503,
            detail=(
                "Ollama model not available. Ensure Ollama is running "
                "and the configured model is pulled."
            ),
        )


@router.post("/chat", response_model=ChatResponse)
def agent_chat(req: ChatRequest):
    """
    Chat with the multi-agent intelligence system (LangGraph).

    Routes the query through specialized agents:
      Supervisor -> Collection -> Analysis -> Dissemination

    Collection and Analysis are deterministic. Dissemination uses
    a local Llama model via Ollama for synthesis.
    """
    _check_ollama()

    try:
        from src.agent.graph import intelligence_graph
        from langchain_core.messages import HumanMessage

        initial_state = {
            "messages": [HumanMessage(content=req.message)],
            "query": req.message,
        }

        result = intelligence_graph.invoke(initial_state)

        agents_used = []
        for msg in result.get("messages", []):
            content = msg.content if hasattr(msg, "content") else str(msg)
            if "[Collection]" in content and "collection" not in agents_used:
                agents_used.append("collection")
            elif "[Analysis]" in content and "analysis" not in agents_used:
                agents_used.append("analysis")

        return ChatResponse(
            response=result.get("final_response", "No response generated."),
            agents_used=agents_used,
            iterations=result.get("iteration", 0),
        )
    except Exception as e:
        logger.exception("Agent chat error")
        raise HTTPException(status_code=500, detail=str(e))
