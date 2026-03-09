"""
Sentiment analysis for news headlines using HuggingFace transformers.

HOW IT WORKS:
-------------
1. Load FinBERT (BERT fine-tuned on financial news) via HuggingFace pipeline
2. Feed headline text through: tokenizer → transformer network → softmax → label
3. Model outputs: label (positive/negative/neutral) + confidence score (0.0-1.0)
4. We convert to our scale: -1.0 (very negative) to +1.0 (very positive)

WHY THIS MODEL:
- ProsusAI/finbert is trained on FINANCIAL NEWS — understands that "strikes" is negative
  and "ceasefire" is positive in a geopolitical context (unlike SST-2 which is movie reviews)
- 3 classes: positive, negative, neutral (SST-2 only has 2)
- ~420MB, still fast on CPU for headline-length text
- Based on BERT, fine-tuned on ~4,500 financial news articles from Reuters

USAGE:
    analyzer = SentimentAnalyzer()  # Loads model once
    results = analyzer.analyze_batch(["Oil prices surge", "War breaks out"])
    # [SentimentResult(score=0.82, label='positive'), SentimentResult(score=-0.91, label='negative')]
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Thresholds for converting continuous score to discrete label
POSITIVE_THRESHOLD = 0.1
NEGATIVE_THRESHOLD = -0.1


@dataclass
class SentimentResult:
    """Sentiment analysis output for a single text."""
    score: float      # -1.0 (very negative) to +1.0 (very positive)
    label: str        # "positive", "negative", or "neutral"
    confidence: float  # 0.0-1.0, how sure the model is


class SentimentAnalyzer:
    """
    Batch sentiment analyzer using HuggingFace transformers pipeline.

    The model is loaded lazily on first use and cached for reuse.
    This avoids the ~5s load time when importing the module.

    The pipeline handles tokenization, inference, and decoding internally.
    We just feed it strings and get back labels + scores.
    """

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.model_name = model_name
        self._pipeline = None

    def _load_pipeline(self):
        """Lazy-load the sentiment pipeline (downloads model on first run)."""
        if self._pipeline is not None:
            return

        logger.info(f"Loading sentiment model: {self.model_name}")

        # Import here to avoid slow import at module level
        from transformers import pipeline

        self._pipeline = pipeline(
            "sentiment-analysis",
            model=self.model_name,
            # truncation=True handles headlines that exceed 512 tokens (rare but safe)
            truncation=True,
            # Use CPU — for our batch sizes, GPU isn't needed
            device=-1,
        )
        logger.info("Sentiment model loaded successfully")

    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment of a single text string."""
        results = self.analyze_batch([text])
        return results[0]

    def analyze_batch(self, texts: list[str], batch_size: int = 32) -> list[SentimentResult]:
        """
        Analyze sentiment of multiple texts efficiently.

        The pipeline batches internally for GPU/CPU efficiency.
        batch_size controls how many texts are processed at once —
        higher = faster but more memory.

        Args:
            texts: List of strings to analyze
            batch_size: Number of texts per inference batch

        Returns:
            List of SentimentResult, one per input text
        """
        self._load_pipeline()

        if not texts:
            return []

        # Pipeline returns [{"label": "POSITIVE", "score": 0.9998}, ...]
        raw_results = self._pipeline(texts, batch_size=batch_size)

        results = []
        for raw in raw_results:
            # FinBERT returns: label = "positive" | "negative" | "neutral"
            # score = confidence (0.0-1.0)
            #
            # Convert to our -1 to +1 scale:
            #   positive 0.85 → +0.85
            #   negative 0.90 → -0.90
            #   neutral  0.70 →  0.00 (neutral always maps to 0)
            confidence = raw["score"]
            raw_label = raw["label"].lower()

            if raw_label == "positive":
                score = confidence
            elif raw_label == "negative":
                score = -confidence
            else:
                # Neutral: score is 0, confidence still tracks how sure
                score = 0.0

            # Assign discrete label based on thresholds
            if score > POSITIVE_THRESHOLD:
                label = "positive"
            elif score < NEGATIVE_THRESHOLD:
                label = "negative"
            else:
                label = "neutral"

            results.append(SentimentResult(
                score=round(score, 4),
                label=label,
                confidence=round(confidence, 4),
            ))

        return results


def score_unprocessed_headlines(session, batch_size: int = 100) -> int:
    """
    Score all headlines that don't have sentiment yet.

    This is the main entry point for the sentiment pipeline.
    Call it from a Prefect flow or manually to process new headlines.

    Args:
        session: SQLAlchemy session
        batch_size: Number of headlines to process at once

    Returns:
        Number of headlines scored
    """
    from src.db.models import NewsHeadline

    # Get headlines without sentiment scores
    unscored = (
        session.query(NewsHeadline)
        .filter(NewsHeadline.sentiment_score.is_(None))
        .order_by(NewsHeadline.published_at.desc())
        .all()
    )

    if not unscored:
        logger.info("No unscored headlines found")
        return 0

    logger.info(f"Scoring {len(unscored)} headlines...")

    analyzer = SentimentAnalyzer()
    texts = [h.headline for h in unscored]
    results = analyzer.analyze_batch(texts, batch_size=batch_size)

    # Update each headline with its sentiment
    for headline, result in zip(unscored, results):
        headline.sentiment_score = result.score
        headline.sentiment_label = result.label

    session.commit()
    logger.info(f"Scored {len(unscored)} headlines")
    return len(unscored)
