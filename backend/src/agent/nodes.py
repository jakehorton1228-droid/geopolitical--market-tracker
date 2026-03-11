"""LangGraph agent nodes — one function per specialized agent.

Each node is a function that:
  1. Receives the shared AgentState
  2. Uses Claude with its specialized tools to do work
  3. Returns a state update dict (LangGraph merges it into state)

Architecture:
  Collection Agent  → gathers raw data → writes state.collected_data
  Analysis Agent    → produces insights → writes state.analysis_results
  Dissemination Agent → formats final answer → writes state.final_response
  Supervisor        → reads state, decides next_agent
"""

import json
import logging
from dataclasses import asdict

import anthropic

from src.config.settings import ANTHROPIC_API_KEY, AGENT_MODEL, AGENT_MAX_TOKENS
from src.agent.state import AgentState, CollectedData, AnalysisResults
from src.agent.tools import TOOL_DEFINITIONS, execute_tool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool distribution — each agent gets only the tools it needs
# ---------------------------------------------------------------------------

COLLECTION_TOOLS = [
    t for t in TOOL_DEFINITIONS
    if t["name"] in {
        "get_recent_events",
        "get_event_summary",
        "get_market_data",
        "get_headline_sentiment",
        "search_similar_content",
        "rag_search",
        "list_symbols",
        "get_symbol_countries",
    }
]

ANALYSIS_TOOLS = [
    t for t in TOOL_DEFINITIONS
    if t["name"] in {
        "get_correlations",
        "get_top_correlations",
        "get_historical_patterns",
        "run_prediction",
        "detect_anomalies",
        "get_sentiment_summary",
    }
]

# Dissemination Agent gets no data tools — it reads from state and writes prose

# ---------------------------------------------------------------------------
# System prompts — tight, focused instructions for each agent
# ---------------------------------------------------------------------------

COLLECTION_SYSTEM_PROMPT = """\
You are the Collection Agent in a geopolitical intelligence system.

Your ONLY job is to gather raw data relevant to the user's question.
You do NOT analyze, interpret, or draw conclusions — just collect.

You have access to:
- GDELT geopolitical events (by country, date, type)
- Financial market data (33 instruments: commodities, currencies, ETFs)
- News headlines with sentiment scores
- Semantic search across headlines and events
- RAG retrieval for broad topic research
- Symbol/country reference lookups

Instructions:
- Read the user's question and determine what data to fetch.
- Use multiple tools if the question spans different data types.
- For country-related questions, use get_symbol_countries first to find relevant symbols.
- For broad topics, use rag_search to cast a wide net.
- Be thorough — the Analysis Agent can only work with what you collect.
- Return a brief summary of what you collected (counts, date ranges, sources)."""

ANALYSIS_SYSTEM_PROMPT = """\
You are the Analysis Agent in a geopolitical intelligence system.

Your ONLY job is to analyze data and produce structured insights.
The Collection Agent has already gathered raw data — it's provided below.
You add analytical depth: correlations, patterns, anomalies, sentiment trends.

You have access to:
- Correlation analysis (event metrics vs market returns)
- Historical pattern analysis (frequency-based: "When X happens, Y goes up Z%")
- Logistic regression predictions (next-day market direction)
- Anomaly detection (unusual market-event mismatches)
- Sentiment summary (aggregate mood statistics)

Instructions:
- Review the collected data provided in the context.
- Run analytical tools that are relevant to the user's question.
- Identify the key findings — what's significant, what's unusual.
- Be specific: cite numbers, p-values, sample sizes.
- Return structured findings as a list of key bullet points."""

DISSEMINATION_SYSTEM_PROMPT = """\
You are the Dissemination Agent in a geopolitical intelligence system.

Your job is to synthesize collected data and analysis into a clear,
actionable intelligence briefing for the user.

You have NO data tools — you work entirely from the collected data
and analysis results provided below. Your output IS the final answer.

Instructions:
- Write in a concise, professional intelligence briefing style.
- Lead with the bottom line (BLUF — Bottom Line Up Front).
- Support with specific data points from the collection and analysis.
- Use bullet points and structure for readability.
- If data is insufficient or results are not significant, say so clearly.
- End with a brief disclaimer: this is for educational/research purposes, not financial advice.
- Do NOT fabricate data — only reference what was actually collected and analyzed."""

# ---------------------------------------------------------------------------
# Shared helper: run Claude with tools until it finishes
# ---------------------------------------------------------------------------

MAX_TOOL_ROUNDS = 6  # Per-agent limit (lower than the old monolithic 10)

_client = None

def _get_client() -> anthropic.Anthropic:
    """Lazy-init the Anthropic client (shared across agents)."""
    global _client
    if _client is None:
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not configured")
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


def _run_agent_with_tools(
    system_prompt: str,
    user_message: str,
    tools: list[dict],
) -> tuple[str, list[dict]]:
    """Run a Claude agent loop with tools until it produces a text response.

    Returns:
        (response_text, tool_calls_made)
    """
    client = _get_client()
    messages = [{"role": "user", "content": user_message}]
    tool_calls_made = []

    for round_num in range(MAX_TOOL_ROUNDS):
        kwargs = {
            "model": AGENT_MODEL,
            "max_tokens": AGENT_MAX_TOKENS,
            "system": system_prompt,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        response = client.messages.create(**kwargs)

        if response.stop_reason == "tool_use":
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            tool_results = []
            for block in assistant_content:
                if block.type == "tool_use":
                    logger.info(f"  Tool: {block.name}({block.input})")
                    result_str = execute_tool(block.name, block.input)
                    tool_calls_made.append({
                        "tool": block.name,
                        "input": block.input,
                    })
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_str,
                    })

            messages.append({"role": "user", "content": tool_results})
        else:
            text_parts = [
                block.text for block in response.content
                if hasattr(block, "text")
            ]
            return "\n".join(text_parts), tool_calls_made

    logger.warning("Agent hit max tool rounds")
    return "Reached processing limit.", tool_calls_made


# ---------------------------------------------------------------------------
# Node functions — these are what LangGraph calls
# ---------------------------------------------------------------------------

def collection_node(state: AgentState) -> dict:
    """Collection Agent: gather raw data relevant to the query."""
    logger.info(f"[Collection] Processing: {state['query']}")

    prompt = (
        f"User question: {state['query']}\n\n"
        "Gather all relevant data for this question. Use multiple tools "
        "if needed. Summarize what you collected at the end."
    )

    response_text, tool_calls = _run_agent_with_tools(
        COLLECTION_SYSTEM_PROMPT, prompt, COLLECTION_TOOLS
    )

    # Parse what was collected from the tool calls into structured data
    collected = CollectedData()
    for tc in tool_calls:
        tool_name = tc["tool"]
        # The actual data is in the tool results fed back to Claude,
        # but we also track which categories were populated
        if tool_name == "get_recent_events":
            collected.events.append(tc["input"])
        elif tool_name == "get_market_data":
            collected.market_data.append(tc["input"])
        elif tool_name in ("get_headline_sentiment", "search_similar_content"):
            collected.headlines.append(tc["input"])
        elif tool_name == "rag_search":
            collected.rag_context = tc["input"].get("query", "")

    return {
        "collected_data": collected,
        "current_agent": "collection",
        "messages": [{"role": "assistant", "content": f"[Collection Agent]\n{response_text}"}],
    }


def analysis_node(state: AgentState) -> dict:
    """Analysis Agent: produce insights from collected data."""
    logger.info(f"[Analysis] Processing: {state['query']}")

    # Build context from what Collection gathered
    collected = state.get("collected_data", CollectedData())
    context = f"Collected data summary:\n{json.dumps(asdict(collected), default=str, indent=2)}"

    # Include Collection Agent's response for additional context
    collection_response = ""
    for msg in reversed(state.get("messages", [])):
        content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
        if "[Collection Agent]" in content:
            collection_response = content
            break

    prompt = (
        f"User question: {state['query']}\n\n"
        f"{context}\n\n"
        f"Collection Agent's notes:\n{collection_response}\n\n"
        "Analyze this data. Run relevant analytical tools to add depth. "
        "Return your key findings as structured bullet points."
    )

    response_text, tool_calls = _run_agent_with_tools(
        ANALYSIS_SYSTEM_PROMPT, prompt, ANALYSIS_TOOLS
    )

    # Build analysis results
    results = AnalysisResults()
    results.key_findings = [
        line.strip("- •").strip()
        for line in response_text.split("\n")
        if line.strip().startswith(("-", "•", "*"))
    ]

    for tc in tool_calls:
        if tc["tool"] == "get_sentiment_summary":
            results.sentiment_summary = tc["input"]
        elif tc["tool"] == "get_historical_patterns":
            results.patterns.append(tc["input"])
        elif tc["tool"] == "detect_anomalies":
            results.anomalies.append(tc["input"])

    return {
        "analysis_results": results,
        "current_agent": "analysis",
        "messages": [{"role": "assistant", "content": f"[Analysis Agent]\n{response_text}"}],
    }


def dissemination_node(state: AgentState) -> dict:
    """Dissemination Agent: synthesize everything into a final briefing."""
    logger.info(f"[Dissemination] Processing: {state['query']}")

    collected = state.get("collected_data", CollectedData())
    analysis = state.get("analysis_results", AnalysisResults())

    # Build the full context from both prior agents
    agent_notes = []
    for msg in state.get("messages", []):
        content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
        if "[Collection Agent]" in content or "[Analysis Agent]" in content:
            agent_notes.append(content)

    prompt = (
        f"User question: {state['query']}\n\n"
        f"=== COLLECTED DATA ===\n{json.dumps(asdict(collected), default=str, indent=2)}\n\n"
        f"=== ANALYSIS RESULTS ===\n{json.dumps(asdict(analysis), default=str, indent=2)}\n\n"
        f"=== AGENT NOTES ===\n{'---'.join(agent_notes)}\n\n"
        "Synthesize this into a clear, actionable intelligence briefing. "
        "Lead with the bottom line (BLUF)."
    )

    response_text, _ = _run_agent_with_tools(
        DISSEMINATION_SYSTEM_PROMPT, prompt, []  # No tools
    )

    return {
        "final_response": response_text,
        "current_agent": "dissemination",
        "messages": [{"role": "assistant", "content": response_text}],
    }
