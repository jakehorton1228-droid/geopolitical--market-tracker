"""LLM synthesis layer — generates intelligence assessments from structured data.

Uses a locally-hosted Llama model via Ollama to transform structured analysis
results into readable intelligence briefings. This is the only LLM call in the
pipeline — everything upstream is deterministic.

The model's job is synthesis, not analysis. It reads pre-computed results
(correlations, anomalies, patterns, sentiment, events) and writes a structured
assessment following a fixed intelligence product format.
"""

import logging

import ollama

from src.config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_MAX_TOKENS

logger = logging.getLogger(__name__)

ASSESSMENT_SYSTEM_PROMPT = """\
You are a geopolitical intelligence analyst. You receive structured data from
an automated analysis pipeline and synthesize it into a clear, actionable
intelligence assessment.

Your input is pre-computed — do NOT fabricate data or statistics. Only reference
what is provided. If data is insufficient, say so.

Write in a professional intelligence briefing style:
- Lead with the bottom line (BLUF — Bottom Line Up Front)
- Support with specific data points from the input
- Use bullet points for readability
- Be concise and direct

Output format:

BOTTOM LINE: [1-2 sentence summary of what matters most]

KEY FINDINGS:
- [Finding with supporting data]
- [Finding with supporting data]

MARKET IMPLICATIONS:
- [Asset/sector + expected direction + reasoning]

HISTORICAL CONTEXT:
- [Relevant analogs or patterns from the data]

WATCHLIST:
- [What to monitor next and why]

CONFIDENCE: [High/Medium/Low] — [brief justification]

End with: "This assessment is for educational/research purposes only. Not financial advice."
"""

BRIEFING_SYSTEM_PROMPT = """\
You are an intelligence analyst writing a daily situational awareness briefing.
You have been given relevant headlines and events retrieved from a database.

Write a concise briefing (3-5 paragraphs) covering:
1) Top threats and conflicts
2) Diplomatic developments
3) Economic/market implications
4) Key watchpoints for the next 24-48 hours

Be factual, cite specific events from the context, and avoid speculation.
Write in a professional intelligence briefing style.
"""


def _get_client() -> ollama.Client:
    """Create an Ollama client pointing at the configured endpoint."""
    return ollama.Client(host=OLLAMA_BASE_URL)


def generate_assessment(structured_data: str) -> str:
    """Generate an intelligence assessment from structured analysis data.

    Args:
        structured_data: JSON or formatted string of analysis results
            (events, market data, correlations, anomalies, patterns, sentiment)

    Returns:
        Formatted intelligence assessment string
    """
    client = _get_client()

    try:
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": ASSESSMENT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Generate an intelligence assessment based on this "
                        f"analysis data:\n\n{structured_data}"
                    ),
                },
            ],
            options={"num_predict": OLLAMA_MAX_TOKENS},
        )
        return response["message"]["content"]
    except Exception as e:
        logger.error(f"Ollama assessment generation failed: {e}")
        return f"Assessment generation unavailable: {e}"


def generate_briefing(rag_context: str) -> str:
    """Generate a situational awareness briefing from RAG context.

    Args:
        rag_context: Formatted context from ContextBuilder with headlines and events

    Returns:
        Formatted briefing string
    """
    client = _get_client()

    try:
        response = client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": BRIEFING_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Generate a situational awareness briefing based on "
                        f"this intelligence:\n\n{rag_context}"
                    ),
                },
            ],
            options={"num_predict": 1024},
        )
        return response["message"]["content"]
    except Exception as e:
        logger.error(f"Ollama briefing generation failed: {e}")
        return f"Briefing generation unavailable: {e}"


def is_available() -> bool:
    """Check if Ollama is reachable and the configured model is loaded."""
    try:
        client = _get_client()
        models = client.list()
        model_names = [m.model for m in models.models]
        return any(OLLAMA_MODEL in name for name in model_names)
    except Exception:
        return False
