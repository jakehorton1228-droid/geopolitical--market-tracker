"""Multi-agent state schema for LangGraph.

This is the shared "tracker board" that all agents read from and write to.
The Supervisor checks this state to decide which agent runs next.

Architecture mirrors the military intelligence cycle:
  Collection → Analysis → Dissemination

State fields use LangGraph reducers:
  - Annotated[list, operator.add] = append reducer (messages accumulate)
  - Plain types = overwrite reducer (last write wins)
"""

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from langgraph.graph import MessagesState


# Agent identifiers — used for routing
AgentName = Literal["collection", "analysis", "dissemination", "supervisor", "END"]


@dataclass
class CollectedData:
    """Raw data gathered by the Collection Agent.

    Each field is optional — Collection only populates what's relevant
    to the user's query.
    """
    events: list[dict] = field(default_factory=list)
    market_data: list[dict] = field(default_factory=list)
    headlines: list[dict] = field(default_factory=list)
    correlations: list[dict] = field(default_factory=list)
    predictions: list[dict] = field(default_factory=list)
    rag_context: str = ""


@dataclass
class AnalysisResults:
    """Structured insights produced by the Analysis Agent."""
    sentiment_summary: dict = field(default_factory=dict)
    patterns: list[dict] = field(default_factory=list)
    anomalies: list[dict] = field(default_factory=list)
    risk_assessment: str = ""
    key_findings: list[str] = field(default_factory=list)


class AgentState(MessagesState):
    """Shared state for the multi-agent intelligence graph.

    Extends MessagesState which gives us a `messages` field with
    an append reducer (messages accumulate across agent turns).

    The Supervisor reads `next_agent` to route, and each specialized
    agent writes to its own section (collected_data or analysis_results)
    so there's no collision.
    """

    # The original user question (set once at the start)
    query: str = ""

    # Collection Agent writes here
    collected_data: CollectedData = field(default_factory=CollectedData)

    # Analysis Agent writes here
    analysis_results: AnalysisResults = field(default_factory=AnalysisResults)

    # Dissemination Agent writes the final formatted answer
    final_response: str = ""

    # Routing: Supervisor sets next_agent, graph reads it for edges
    current_agent: AgentName = "supervisor"
    next_agent: AgentName = "supervisor"

    # Safety counter — prevents infinite loops (like MAX_TOOL_ROUNDS)
    iteration: int = 0
    max_iterations: int = 10
