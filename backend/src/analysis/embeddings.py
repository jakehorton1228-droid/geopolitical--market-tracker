"""
Embedding generation for headlines and events using sentence-transformers.

HOW IT WORKS:
-------------
1. Load all-MiniLM-L6-v2 model (~80MB, runs locally, no API key)
2. Feed text through: tokenizer → transformer → mean pooling → 384-dim vector
3. Store the vector in the `embedding` column (pgvector type)
4. Later, use cosine distance (<=> in SQL) to find semantically similar rows

WHY EMBEDDINGS:
- Sentiment tells you HOW a headline feels (positive/negative)
- Embeddings tell you WHAT it's about (meaning as a 384-number vector)
- "Russian troops advance" and "Moscow military offensive" have different words
  but nearly identical embeddings — they mean the same thing
- This powers semantic search: "find headlines about nuclear escalation"
  without needing the exact words "nuclear" or "escalation"

WHY THIS MODEL:
- all-MiniLM-L6-v2: optimized for semantic similarity of short texts
- 384 dimensions (small, fast) vs 768+ for larger models
- ~80MB download, fast inference on CPU
- Trained on 1B+ sentence pairs — excellent general-purpose embeddings

USAGE:
    generator = EmbeddingGenerator()
    vectors = generator.encode(["Oil prices surge", "War breaks out in Syria"])
    # [array(384,), array(384,)] — each is a 384-dim numpy array
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Must match EMBEDDING_DIM in models.py and the Vector(384) column
EMBEDDING_DIM = 384
DEFAULT_MODEL = "all-MiniLM-L6-v2"


class EmbeddingGenerator:
    """
    Generates 384-dim embeddings using sentence-transformers.

    Like SentimentAnalyzer, the model is lazy-loaded on first use.
    sentence-transformers wraps HuggingFace models with a simpler API
    optimized for generating embeddings (vs. classification like FinBERT).
    The key difference: FinBERT outputs labels, this outputs vectors.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Lazy-load the sentence-transformers model."""
        if self._model is not None:
            return

        logger.info(f"Loading embedding model: {self.model_name}")

        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name)
        logger.info(f"Embedding model loaded (dim={self._model.get_sentence_embedding_dimension()})")

    def encode(self, texts: list[str], batch_size: int = 64) -> list[np.ndarray]:
        """
        Convert texts to embedding vectors.

        Under the hood, for each text:
        1. Tokenize: "oil prices" → [101, 3514, 4597, 102]
        2. Feed through 6 transformer layers (MiniLM has 6, BERT has 12)
        3. Mean pool all token embeddings → one 384-dim vector per text
        4. Normalize to unit length (so cosine distance works correctly)

        Args:
            texts: List of strings to embed
            batch_size: Texts per batch (higher = faster, more memory)

        Returns:
            List of numpy arrays, each shape (384,)
        """
        self._load_model()

        if not texts:
            return []

        # encode() returns a numpy array of shape (n_texts, 384)
        # normalize_embeddings=True makes cosine distance equivalent to dot product
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,
        )

        return [emb for emb in embeddings]


def _event_to_text(event) -> str:
    """
    Convert an Event ORM object to a descriptive text string for embedding.

    We compose a natural-language summary from the structured fields because
    the embedding model works on text, not numbers. The richer the text,
    the better the embedding captures the event's meaning.

    Example output: "USA military action against IRN in Tehran, Iraq.
    Goldstein: -7.0, Mentions: 450"
    """
    parts = []

    # Actors
    actor1 = event.actor1_name or event.actor1_code or "Unknown"
    actor2 = event.actor2_name or event.actor2_code or "Unknown"
    parts.append(f"{actor1} action against {actor2}")

    # Location
    if event.action_geo_name:
        parts.append(f"in {event.action_geo_name}")
    elif event.action_geo_country_code:
        parts.append(f"in {event.action_geo_country_code}")

    # Metrics that carry meaning
    if event.goldstein_scale is not None:
        scale = event.goldstein_scale
        tone = "cooperative" if scale > 2 else "conflictual" if scale < -2 else "neutral"
        parts.append(f"({tone}, Goldstein: {scale:.1f})")

    if event.num_mentions and event.num_mentions > 10:
        parts.append(f"Mentions: {event.num_mentions}")

    return ". ".join(parts)


def embed_unprocessed_headlines(session, batch_size: int = 100) -> int:
    """
    Generate embeddings for all headlines that don't have one yet.

    Mirrors score_unprocessed_headlines() — queries for NULL embeddings,
    generates vectors, writes them back.

    Args:
        session: SQLAlchemy session
        batch_size: Headlines per encoding batch

    Returns:
        Number of headlines embedded
    """
    from src.db.models import NewsHeadline

    unembedded = (
        session.query(NewsHeadline)
        .filter(NewsHeadline.embedding.is_(None))
        .order_by(NewsHeadline.published_at.desc())
        .all()
    )

    if not unembedded:
        logger.info("No unembedded headlines found")
        return 0

    logger.info(f"Embedding {len(unembedded)} headlines...")

    generator = EmbeddingGenerator()
    texts = [h.headline for h in unembedded]
    vectors = generator.encode(texts, batch_size=batch_size)

    for headline, vector in zip(unembedded, vectors):
        headline.embedding = vector.tolist()

    session.commit()
    logger.info(f"Embedded {len(unembedded)} headlines")
    return len(unembedded)


def embed_unprocessed_events(session, batch_size: int = 100) -> int:
    """
    Generate embeddings for all events that don't have one yet.

    Events don't have a single text field, so we compose a summary
    string from structured fields (actors, location, metrics) using
    _event_to_text().

    Args:
        session: SQLAlchemy session
        batch_size: Events per encoding batch

    Returns:
        Number of events embedded
    """
    from src.db.models import Event

    unembedded = (
        session.query(Event)
        .filter(Event.embedding.is_(None))
        .order_by(Event.event_date.desc())
        .limit(5000)  # Cap per run — events table can be huge
        .all()
    )

    if not unembedded:
        logger.info("No unembedded events found")
        return 0

    logger.info(f"Embedding {len(unembedded)} events...")

    generator = EmbeddingGenerator()
    texts = [_event_to_text(e) for e in unembedded]
    vectors = generator.encode(texts, batch_size=batch_size)

    for event, vector in zip(unembedded, vectors):
        event.embedding = vector.tolist()

    session.commit()
    logger.info(f"Embedded {len(unembedded)} events")
    return len(unembedded)
