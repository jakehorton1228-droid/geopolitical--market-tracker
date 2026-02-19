"""Agent service — Claude API with tool use loop.

Handles the agentic conversation loop:
1. User sends a message
2. Claude decides which tools to call
3. Tools are executed, results fed back
4. Claude synthesizes a final response
"""

import logging
from typing import Optional

import anthropic

from src.config.settings import ANTHROPIC_API_KEY, AGENT_MODEL, AGENT_MAX_TOKENS
from src.agent.tools import TOOL_DEFINITIONS, execute_tool

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a geopolitical market analyst AI with access to 10 years of GDELT event data (2016–present) and daily market data for 33 financial instruments (commodities, currencies, ETFs, bonds, volatility).

Your capabilities:
- Query geopolitical events by country, date, type, and significance
- Analyze correlations between event metrics and market returns
- Examine historical frequency patterns ("When X happens, Y goes UP Z% of the time")
- Run logistic regression predictions for market direction
- Detect anomalies — unusual market moves not explained by events

Guidelines:
- Always use your tools to look up data before answering. Never guess at numbers.
- Cite specific dates, values, and sample sizes in your answers.
- When discussing correlations, mention the p-value and whether it's statistically significant.
- Be concise but thorough. Use tables and bullet points for readability.
- If data is insufficient or results are not significant, say so clearly.
- Always include a brief disclaimer that this is for educational/research purposes, not financial advice.
- When a user asks about a country, look up which symbols are sensitive to that country using the get_symbol_countries tool.
- For broad questions like "what's happening in the market", start with get_top_correlations and recent events."""

MAX_TOOL_ROUNDS = 10  # Safety limit on agentic loop iterations


class AgentService:
    """Manages Claude API interactions with tool use."""

    def __init__(self):
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not configured")
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def chat(
        self,
        message: str,
        history: Optional[list[dict]] = None,
    ) -> dict:
        """
        Send a message to the agent and get a response.

        Args:
            message: User's message
            history: Previous conversation messages [{role, content}, ...]

        Returns:
            {
                "response": str,       # Agent's text response
                "tool_calls": list,    # Tools the agent used
                "model": str,          # Model used
            }
        """
        # Build messages from history + new user message
        messages = []
        if history:
            for msg in history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })
        messages.append({"role": "user", "content": message})

        tool_calls_made = []

        # Agentic loop: keep going until Claude returns a text response
        for round_num in range(MAX_TOOL_ROUNDS):
            logger.info(f"Agent round {round_num + 1}, messages: {len(messages)}")

            response = self.client.messages.create(
                model=AGENT_MODEL,
                max_tokens=AGENT_MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=messages,
            )

            # Check if Claude wants to use tools
            if response.stop_reason == "tool_use":
                # Process all tool use blocks in this response
                assistant_content = response.content
                messages.append({"role": "assistant", "content": assistant_content})

                tool_results = []
                for block in assistant_content:
                    if block.type == "tool_use":
                        tool_name = block.name
                        tool_input = block.input
                        tool_id = block.id

                        logger.info(f"Executing tool: {tool_name}({tool_input})")
                        result_str = execute_tool(tool_name, tool_input)

                        tool_calls_made.append({
                            "tool": tool_name,
                            "input": tool_input,
                        })

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": result_str,
                        })

                messages.append({"role": "user", "content": tool_results})

            else:
                # Claude returned a final text response
                text_parts = []
                for block in response.content:
                    if hasattr(block, "text"):
                        text_parts.append(block.text)

                return {
                    "response": "\n".join(text_parts),
                    "tool_calls": tool_calls_made,
                    "model": AGENT_MODEL,
                }

        # Safety: if we hit max rounds, return what we have
        logger.warning(f"Agent hit max tool rounds ({MAX_TOOL_ROUNDS})")
        return {
            "response": "I used several tools to analyze your question but reached the processing limit. Please try a more specific question.",
            "tool_calls": tool_calls_made,
            "model": AGENT_MODEL,
        }
