"""LangGraph multi-agent graph — deterministic Supervisor orchestration.

Wires Collection, Analysis, and Dissemination nodes into a directed
graph with conditional routing. The Supervisor uses deterministic
state-based logic to decide which agent runs next — no LLM needed
for routing decisions.

Graph flow:
  START -> supervisor -> collection -> supervisor -> analysis -> supervisor
        -> dissemination -> supervisor -> END
"""

import logging

from langgraph.graph import StateGraph, END

from src.agent.state import AgentState, CollectedData, AnalysisResults
from src.agent.nodes import collection_node, analysis_node, dissemination_node

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Supervisor — deterministic routing based on state
# ---------------------------------------------------------------------------

def supervisor_node(state: AgentState) -> dict:
    """Supervisor: decide which agent runs next using deterministic logic.

    Rules:
    - If no data collected yet -> collection
    - If data collected but not analyzed -> analysis
    - If analyzed but no final response -> dissemination
    - If final response exists -> END
    """
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 10)

    if iteration >= max_iter:
        logger.warning("Supervisor hit max iterations, forcing END")
        return {"next_agent": "END", "iteration": iteration + 1}

    collected = state.get("collected_data", CollectedData())
    analysis = state.get("analysis_results", AnalysisResults())
    final = state.get("final_response", "")

    has_collection = bool(
        collected.events or collected.market_data
        or collected.headlines or collected.rag_context
    )
    has_analysis = bool(
        analysis.key_findings or analysis.sentiment_summary
        or analysis.patterns or analysis.anomalies
    )
    has_final = bool(final)

    if has_final:
        decision = "END"
    elif has_analysis:
        decision = "dissemination"
    elif has_collection:
        decision = "analysis"
    else:
        decision = "collection"

    logger.info(
        f"[Supervisor] Iteration {iteration}: "
        f"collected={has_collection}, analyzed={has_analysis}, "
        f"final={has_final} -> {decision}"
    )

    return {"next_agent": decision, "iteration": iteration + 1}


# ---------------------------------------------------------------------------
# Route function — reads next_agent from state for conditional edges
# ---------------------------------------------------------------------------

def route_next(state: AgentState) -> str:
    """Conditional edge function: return the node name to run next."""
    next_agent = state.get("next_agent", "END")
    if next_agent == "END":
        return END
    return next_agent


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Construct and compile the multi-agent intelligence graph.

    Returns a compiled graph ready to .invoke() or .stream().
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("collection", collection_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("dissemination", dissemination_node)

    # Entry point: always start with the Supervisor
    graph.set_entry_point("supervisor")

    # After each agent finishes, go back to the Supervisor
    graph.add_edge("collection", "supervisor")
    graph.add_edge("analysis", "supervisor")
    graph.add_edge("dissemination", "supervisor")

    # Supervisor uses conditional edge to pick the next agent (or END)
    graph.add_conditional_edges(
        "supervisor",
        route_next,
        {
            "collection": "collection",
            "analysis": "analysis",
            "dissemination": "dissemination",
            END: END,
        },
    )

    return graph.compile()


# Module-level compiled graph (import and use directly)
intelligence_graph = build_graph()
