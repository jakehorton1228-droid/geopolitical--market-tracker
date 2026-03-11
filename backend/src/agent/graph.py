"""LangGraph multi-agent graph — the Supervisor orchestration layer.

Wires Collection, Analysis, and Dissemination nodes into a directed
graph with conditional routing. The Supervisor decides which agent
runs next based on the current state.

Graph flow:
  START → supervisor → collection → supervisor → analysis → supervisor
        → dissemination → supervisor → END
"""

import logging

import anthropic
from langgraph.graph import StateGraph, END

from src.config.settings import ANTHROPIC_API_KEY, AGENT_MODEL
from src.agent.state import AgentState, CollectedData, AnalysisResults
from src.agent.nodes import collection_node, analysis_node, dissemination_node

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supervisor — the routing brain
# ---------------------------------------------------------------------------

_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


def supervisor_node(state: AgentState) -> dict:
    """Supervisor: decide which agent runs next.

    Reads the current state and uses Claude to make a routing decision.
    Returns a state update with `next_agent` set.
    """
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 10)

    # Safety: if we've looped too many times, force end
    if iteration >= max_iter:
        logger.warning("Supervisor hit max iterations, forcing END")
        return {"next_agent": "END", "iteration": iteration + 1}

    query = state.get("query", "")
    collected = state.get("collected_data", CollectedData())
    analysis = state.get("analysis_results", AnalysisResults())
    final = state.get("final_response", "")
    current = state.get("current_agent", "supervisor")

    # Build a concise status summary for the Supervisor's decision
    has_collection = bool(
        collected.events or collected.market_data
        or collected.headlines or collected.rag_context
    )
    has_analysis = bool(
        analysis.key_findings or analysis.sentiment_summary
        or analysis.patterns or analysis.anomalies
    )
    has_final = bool(final)

    status = (
        f"Query: {query}\n"
        f"Current agent just finished: {current}\n"
        f"Data collected: {'YES' if has_collection else 'NO'}\n"
        f"Analysis done: {'YES' if has_analysis else 'NO'}\n"
        f"Final response written: {'YES' if has_final else 'NO'}\n"
        f"Iteration: {iteration}/{max_iter}"
    )

    # Use Claude to decide routing
    client = _get_client()
    response = client.messages.create(
        model=AGENT_MODEL,
        max_tokens=50,
        system=(
            "You are a routing supervisor for an intelligence analysis system. "
            "Based on the current state, decide which agent should run next.\n\n"
            "Rules:\n"
            "- If no data has been collected yet, route to 'collection'\n"
            "- If data is collected but not analyzed, route to 'analysis'\n"
            "- If data is collected AND analyzed but no final response, route to 'dissemination'\n"
            "- If a final response exists, route to 'END'\n\n"
            "Respond with ONLY one word: collection, analysis, dissemination, or END"
        ),
        messages=[{"role": "user", "content": status}],
    )

    decision = response.content[0].text.strip().lower()

    # Validate the decision
    valid = {"collection", "analysis", "dissemination", "end", "END"}
    if decision not in valid:
        logger.warning(f"Supervisor returned invalid decision '{decision}', defaulting to END")
        decision = "END"

    # Normalize
    decision = decision.upper() if decision.lower() == "end" else decision

    logger.info(f"[Supervisor] Iteration {iteration}: {current} → {decision}")

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
