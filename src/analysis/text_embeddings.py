"""
Text Embeddings using Sentence Transformers.

================================================================================
WHAT ARE EMBEDDINGS? (The Core Idea)
================================================================================

An embedding is a way to represent something (text, images, etc.) as a vector
of numbers. The magic is that SIMILAR things have SIMILAR vectors.

Example with 3D vectors (real embeddings use 384+ dimensions):

    "Russia attacks Ukraine"  → [0.8, -0.7, 0.2]
    "Military strikes in Kyiv" → [0.75, -0.65, 0.25]  # Similar!
    "Apple releases new iPhone" → [-0.3, 0.1, 0.9]    # Very different

WHY IS THIS USEFUL?
- Similar events cluster together automatically
- We can find related events without keyword matching
- Numbers can be fed into ML models as features
- Enables semantic search ("find events similar to X")

================================================================================
HOW DO TRANSFORMERS CREATE EMBEDDINGS?
================================================================================

Traditional approaches (before 2017):
- Word2Vec: Each word gets a fixed vector
- Problem: "bank" (river) and "bank" (financial) have SAME vector!

Transformer approach (2017+):
- Considers the ENTIRE sentence when encoding each word
- "bank" gets different vectors in different contexts
- Uses "attention" to understand which words relate to which

THE ATTENTION MECHANISM (Simplified):
When encoding "Russia attacks Ukraine", the model asks:
- For "attacks": How relevant is "Russia"? (very) How relevant is "Ukraine"? (very)
- This context-awareness creates much better representations

SENTENCE TRANSFORMERS:
- Takes a full sentence and produces ONE vector (not one per word)
- Pre-trained on millions of sentence pairs
- Fine-tuned so similar sentences have similar vectors
- Model we use: "all-MiniLM-L6-v2" (fast, 384 dimensions, good quality)

================================================================================
TRANSFER LEARNING
================================================================================

We're using a PRE-TRAINED model. This is called "transfer learning":

1. Someone else trained this model on MASSIVE text data (billions of sentences)
2. The model learned general language understanding
3. We download and use it directly - no training needed!
4. It works well on our geopolitical events even though it wasn't trained on them

This is like hiring an expert who already knows language - you don't need to
teach them from scratch.

================================================================================
COSINE SIMILARITY
================================================================================

To compare two embeddings, we use COSINE SIMILARITY:

                    A · B           (dot product)
    cos(θ) = ─────────────────
              ||A|| × ||B||      (product of magnitudes)

- Result is between -1 and 1
- 1.0 = identical direction (very similar)
- 0.0 = perpendicular (unrelated)
- -1.0 = opposite direction (opposite meaning)

For text embeddings, typical thresholds:
- > 0.8: Very similar (near duplicates)
- > 0.5: Related topics
- < 0.3: Unrelated

================================================================================
USAGE
================================================================================

    from src.analysis.text_embeddings import EventEmbedder

    embedder = EventEmbedder()

    # Embed a single event
    text = "Russia launches military operation in Ukraine"
    embedding = embedder.embed_text(text)
    print(embedding.shape)  # (384,)

    # Find similar events
    events = ["Ukraine conflict escalates", "Apple stock rises", "NATO responds"]
    similarities = embedder.find_similar(text, events)

    # Use embeddings as ML features
    event_vectors = embedder.embed_events(event_list)
"""

from dataclasses import dataclass
from typing import Optional
import logging

import numpy as np

# Sentence Transformers is built on top of Hugging Face's transformers library
# It provides easy-to-use pre-trained models for generating embeddings
from sentence_transformers import SentenceTransformer

# For efficient similarity calculations
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Model choice explanation:
# - "all-MiniLM-L6-v2": Good balance of speed and quality
#   - 384 dimensions (smaller = faster, but less expressive)
#   - 6 transformer layers (L6)
#   - ~90MB download
#   - ~14,000 sentences/second on GPU, ~1,000 on CPU
#
# Other options (trade-offs):
# - "all-mpnet-base-v2": Better quality, but slower (768 dims)
# - "paraphrase-MiniLM-L3-v2": Faster, but lower quality (384 dims, only 3 layers)
DEFAULT_MODEL = "all-MiniLM-L6-v2"

# CAMEO event codes to human-readable descriptions
# We use these to create text from structured event data
CAMEO_DESCRIPTIONS = {
    "01": "made a public statement",
    "02": "made an appeal",
    "03": "expressed intent to cooperate",
    "04": "consulted",
    "05": "engaged in diplomatic cooperation",
    "06": "engaged in material cooperation",
    "07": "provided aid",
    "08": "made a concession",
    "09": "investigated",
    "10": "demanded",
    "11": "disapproved",
    "12": "rejected",
    "13": "threatened",
    "14": "protested",
    "15": "exhibited military posture",
    "16": "reduced relations",
    "17": "coerced",
    "18": "assaulted",
    "19": "fought",
    "20": "engaged in mass violence",
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class EmbeddingResult:
    """
    Container for an embedding result.

    WHY STORE THE TEXT TOO?
    Useful for debugging and understanding what was embedded.
    The vector alone doesn't tell you what it represents.
    """
    text: str
    embedding: np.ndarray
    model_name: str

    def __post_init__(self):
        """Validate embedding shape after initialization."""
        if len(self.embedding.shape) != 1:
            raise ValueError(f"Embedding must be 1D, got shape {self.embedding.shape}")


@dataclass
class SimilarityResult:
    """Container for similarity search results."""
    query_text: str
    similar_text: str
    similarity_score: float
    rank: int


# =============================================================================
# MAIN EMBEDDER CLASS
# =============================================================================

class EventEmbedder:
    """
    Creates text embeddings for geopolitical events.

    This class:
    1. Converts structured event data to descriptive text
    2. Uses a pre-trained transformer model to create embeddings
    3. Provides similarity search functionality
    4. Generates embeddings that can be used as ML features

    The first time you instantiate this class, it will download the model
    (~90MB) from Hugging Face. Subsequent uses load from cache.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        Initialize the embedder with a pre-trained model.

        Args:
            model_name: Name of the sentence-transformers model to use.
                       See https://www.sbert.net/docs/pretrained_models.html

        NOTE ON FIRST RUN:
        The model will be downloaded and cached in ~/.cache/huggingface/
        This only happens once. After that, loading is fast (~1 second).
        """
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None

        logger.info(f"EventEmbedder initialized with model: {model_name}")

    @property
    def model(self) -> SentenceTransformer:
        """
        Lazy load the model (only when first needed).

        WHY LAZY LOADING?
        Loading a model takes ~1 second. If you're just importing the class
        but not using it yet, you don't want to wait. This defers the load
        until you actually need to create an embedding.
        """
        if self._model is None:
            logger.info(f"Loading model {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        return self._model

    @property
    def embedding_dim(self) -> int:
        """
        Get the dimensionality of embeddings this model produces.

        For all-MiniLM-L6-v2, this is 384.
        Higher dimensions can capture more nuance but:
        - Take more memory
        - Slower to compute similarities
        - May need more data to train downstream models
        """
        return self.model.get_sentence_embedding_dimension()

    # =========================================================================
    # CORE EMBEDDING METHODS
    # =========================================================================

    def embed_text(self, text: str) -> np.ndarray:
        """
        Create an embedding for a single text string.

        HOW IT WORKS (inside the model):
        1. Tokenize: "Russia attacks" → ["Russia", "attack", "s"]
        2. Token IDs: [1234, 5678, 9012]  (model's vocabulary indices)
        3. Initial embeddings: Each token gets a learned vector
        4. Transformer layers: Tokens attend to each other, refining representations
        5. Pooling: Combine all token vectors into ONE sentence vector
           (typically by averaging, called "mean pooling")

        Args:
            text: Any string to embed

        Returns:
            numpy array of shape (embedding_dim,), e.g., (384,)
        """
        # The encode() method handles all the complexity
        # - Tokenization
        # - Padding/truncation to model's max length
        # - Running through transformer
        # - Pooling to single vector
        embedding = self.model.encode(text, convert_to_numpy=True)

        return embedding

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Create embeddings for multiple texts efficiently.

        WHY BATCHING?
        Processing texts one-by-one is slow because:
        - Each call has overhead (moving data to GPU, etc.)
        - GPUs are optimized for parallel computation

        Batching sends multiple texts through the model at once,
        which is MUCH faster (often 10-100x on GPU).

        Args:
            texts: List of strings to embed
            batch_size: How many texts to process at once
                       (larger = faster, but uses more memory)

        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])

        # show_progress_bar is nice for large batches
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            convert_to_numpy=True,
        )

        return embeddings

    # =========================================================================
    # EVENT-SPECIFIC METHODS
    # =========================================================================

    def event_to_text(self, event: dict) -> str:
        """
        Convert a structured event to descriptive text.

        Since we don't have the full article text, we CREATE a description
        from the structured fields. This captures the key information.

        Example:
        Input: {
            "actor1_name": "United States",
            "actor2_name": "Russia",
            "event_root_code": "13",
            "action_geo_name": "Washington",
            "avg_tone": -3.5
        }
        Output: "United States threatened Russia in Washington (negative tone)"

        WHY THIS APPROACH?
        - We have structured data, not raw text
        - Creating a sentence lets us use language models
        - The model understands natural language better than raw features
        - We capture relationships (who did what to whom)
        """
        # Extract components with defaults
        actor1 = event.get("actor1_name") or event.get("actor1_code") or "Unknown actor"
        actor2 = event.get("actor2_name") or event.get("actor2_code") or "another party"
        event_code = event.get("event_root_code", "01")
        location = event.get("action_geo_name") or event.get("action_geo_country_code") or ""
        tone = event.get("avg_tone", 0)

        # Get human-readable action
        action = CAMEO_DESCRIPTIONS.get(event_code, "interacted with")

        # Build the sentence
        text = f"{actor1} {action} {actor2}"

        if location:
            text += f" in {location}"

        # Add tone indicator
        if tone < -2:
            text += " (negative tone)"
        elif tone > 2:
            text += " (positive tone)"

        return text

    def embed_event(self, event: dict) -> EmbeddingResult:
        """
        Create an embedding for a single event.

        Args:
            event: Dictionary with event fields (from database or GDELT)

        Returns:
            EmbeddingResult with text, embedding, and model info
        """
        text = self.event_to_text(event)
        embedding = self.embed_text(text)

        return EmbeddingResult(
            text=text,
            embedding=embedding,
            model_name=self.model_name,
        )

    def embed_events(self, events: list[dict], batch_size: int = 32) -> list[EmbeddingResult]:
        """
        Create embeddings for multiple events efficiently.

        Args:
            events: List of event dictionaries
            batch_size: Batch size for efficient processing

        Returns:
            List of EmbeddingResults (same order as input)
        """
        if not events:
            return []

        # Convert all events to text first
        texts = [self.event_to_text(event) for event in events]

        # Batch embed
        embeddings = self.embed_batch(texts, batch_size=batch_size)

        # Create result objects
        results = [
            EmbeddingResult(text=text, embedding=emb, model_name=self.model_name)
            for text, emb in zip(texts, embeddings)
        ]

        return results

    # =========================================================================
    # SIMILARITY METHODS
    # =========================================================================

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        COSINE SIMILARITY INTUITION:
        Think of embeddings as arrows in high-dimensional space.
        Cosine similarity measures the angle between them:
        - Same direction (angle = 0°) → similarity = 1.0
        - Perpendicular (angle = 90°) → similarity = 0.0
        - Opposite (angle = 180°) → similarity = -1.0

        WHY COSINE OVER EUCLIDEAN DISTANCE?
        Cosine ignores magnitude, only considers direction.
        This is good because:
        - Different texts might have different "confidence" (magnitude)
        - We care about WHAT is being said, not HOW STRONGLY

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score between -1 and 1
        """
        # Reshape for sklearn's cosine_similarity (expects 2D)
        e1 = embedding1.reshape(1, -1)
        e2 = embedding2.reshape(1, -1)

        # Returns a 2D array, we want the single value
        similarity = cosine_similarity(e1, e2)[0, 0]

        return float(similarity)

    def find_similar(
        self,
        query: str,
        candidates: list[str],
        top_k: int = 5,
    ) -> list[SimilarityResult]:
        """
        Find the most similar texts to a query.

        This is the basis of SEMANTIC SEARCH:
        - Traditional search: Match keywords exactly
        - Semantic search: Match by meaning

        Example:
        Query: "military conflict in Eastern Europe"
        Might find: "Russian forces advance in Ukraine"
        Even though they share few exact words!

        Args:
            query: The text to find similar items for
            candidates: List of texts to search through
            top_k: How many results to return

        Returns:
            List of SimilarityResults, sorted by similarity (highest first)
        """
        if not candidates:
            return []

        # Embed query and all candidates
        query_embedding = self.embed_text(query)
        candidate_embeddings = self.embed_batch(candidates)

        # Compute all similarities at once (vectorized, fast)
        # Shape: (1, n_candidates)
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            candidate_embeddings
        )[0]

        # Get top_k indices (argsort gives ascending, so we reverse)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = [
            SimilarityResult(
                query_text=query,
                similar_text=candidates[idx],
                similarity_score=float(similarities[idx]),
                rank=rank + 1,
            )
            for rank, idx in enumerate(top_indices)
        ]

        return results

    def find_similar_events(
        self,
        query_event: dict,
        candidate_events: list[dict],
        top_k: int = 5,
    ) -> list[tuple[dict, float]]:
        """
        Find events most similar to a query event.

        Args:
            query_event: The event to find similar ones for
            candidate_events: List of events to search through
            top_k: How many to return

        Returns:
            List of (event, similarity_score) tuples
        """
        if not candidate_events:
            return []

        # Embed query
        query_result = self.embed_event(query_event)

        # Embed candidates
        candidate_results = self.embed_events(candidate_events)
        candidate_embeddings = np.array([r.embedding for r in candidate_results])

        # Compute similarities
        similarities = cosine_similarity(
            query_result.embedding.reshape(1, -1),
            candidate_embeddings
        )[0]

        # Get top_k
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = [
            (candidate_events[idx], float(similarities[idx]))
            for idx in top_indices
        ]

        return results

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def cluster_events(
        self,
        events: list[dict],
        n_clusters: int = 5,
    ) -> list[list[dict]]:
        """
        Group similar events into clusters.

        This uses K-MEANS CLUSTERING:
        1. Place n_clusters "centroids" randomly in embedding space
        2. Assign each event to nearest centroid
        3. Move centroids to center of their assigned events
        4. Repeat until stable

        Result: Events with similar meaning end up in same cluster,
        even if they use different words!

        Args:
            events: List of event dictionaries
            n_clusters: Number of groups to create

        Returns:
            List of lists, where each inner list is a cluster of events
        """
        from sklearn.cluster import KMeans

        if len(events) < n_clusters:
            return [events]  # Not enough events to cluster

        # Embed all events
        results = self.embed_events(events)
        embeddings = np.array([r.embedding for r in results])

        # Cluster
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        # Group events by cluster
        clusters = [[] for _ in range(n_clusters)]
        for event, label in zip(events, labels):
            clusters[label].append(event)

        return clusters

    def get_event_embedding_for_ml(self, event: dict) -> np.ndarray:
        """
        Get embedding in a format ready for ML model input.

        Use this when you want to add embeddings as features to your
        gradient boosting or LSTM models.

        Returns:
            1D numpy array of shape (embedding_dim,)
        """
        return self.embed_event(event).embedding

    def aggregate_event_embeddings(
        self,
        events: list[dict],
        method: str = "mean"
    ) -> np.ndarray:
        """
        Combine multiple event embeddings into one.

        Useful when you have multiple events per day and want a single
        "daily event embedding" for your model.

        AGGREGATION METHODS:
        - "mean": Average of all embeddings (most common)
        - "max": Element-wise maximum (captures strongest signals)
        - "min": Element-wise minimum

        Args:
            events: List of events to aggregate
            method: How to combine ("mean", "max", "min")

        Returns:
            Single embedding representing all events
        """
        if not events:
            # Return zeros if no events
            return np.zeros(self.embedding_dim)

        results = self.embed_events(events)
        embeddings = np.array([r.embedding for r in results])

        if method == "mean":
            return embeddings.mean(axis=0)
        elif method == "max":
            return embeddings.max(axis=0)
        elif method == "min":
            return embeddings.min(axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """
    Example: Create embeddings and find similar events.

    Run this file directly:
        python -m src.analysis.text_embeddings

    First run will download the model (~90MB).
    """
    import logging
    logging.basicConfig(level=logging.INFO)

    # Create embedder (first run downloads model)
    print("Creating embedder...")
    embedder = EventEmbedder()

    # Example 1: Embed raw text
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Text Embedding")
    print("="*60)

    text1 = "Russia launches military operation in Ukraine"
    text2 = "Russian forces attack Ukrainian cities"
    text3 = "Apple announces new iPhone model"

    emb1 = embedder.embed_text(text1)
    emb2 = embedder.embed_text(text2)
    emb3 = embedder.embed_text(text3)

    print(f"\nEmbedding shape: {emb1.shape}")
    print(f"\nSimilarity scores:")
    print(f"  '{text1[:40]}...'")
    print(f"  vs '{text2[:40]}...': {embedder.compute_similarity(emb1, emb2):.3f}")
    print(f"  vs '{text3[:40]}...': {embedder.compute_similarity(emb1, emb3):.3f}")

    # Example 2: Embed structured events
    print("\n" + "="*60)
    print("EXAMPLE 2: Event Embedding")
    print("="*60)

    events = [
        {
            "actor1_name": "Russia",
            "actor2_name": "Ukraine",
            "event_root_code": "19",  # fought
            "action_geo_name": "Kyiv",
            "avg_tone": -5.2,
        },
        {
            "actor1_name": "United States",
            "actor2_name": "China",
            "event_root_code": "13",  # threatened
            "action_geo_name": "Washington",
            "avg_tone": -2.1,
        },
        {
            "actor1_name": "Germany",
            "actor2_name": "France",
            "event_root_code": "05",  # diplomatic cooperation
            "action_geo_name": "Berlin",
            "avg_tone": 3.5,
        },
    ]

    print("\nEvent texts generated:")
    for event in events:
        text = embedder.event_to_text(event)
        print(f"  - {text}")

    # Find similar events
    print("\n" + "="*60)
    print("EXAMPLE 3: Find Similar Events")
    print("="*60)

    query_event = {
        "actor1_name": "NATO",
        "actor2_name": "Russia",
        "event_root_code": "15",  # military posture
        "action_geo_name": "Brussels",
        "avg_tone": -1.5,
    }

    print(f"\nQuery: {embedder.event_to_text(query_event)}")
    print("\nMost similar events:")

    similar = embedder.find_similar_events(query_event, events, top_k=3)
    for event, score in similar:
        print(f"  {score:.3f}: {embedder.event_to_text(event)}")

    # Example 4: Aggregate daily embeddings
    print("\n" + "="*60)
    print("EXAMPLE 4: Aggregate Events (for ML features)")
    print("="*60)

    daily_embedding = embedder.aggregate_event_embeddings(events, method="mean")
    print(f"\nAggregated embedding shape: {daily_embedding.shape}")
    print(f"This can be used as a feature vector in your ML models!")
