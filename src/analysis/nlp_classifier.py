"""
NLP Event Classifier using zero-shot classification.

Classifies geopolitical events into defense-relevant categories
using a pre-trained transformer model, without requiring labeled
training data.

HOW ZERO-SHOT CLASSIFICATION WORKS:
------------------------------------
Instead of training on labeled examples, zero-shot classification
frames classification as a natural language inference (NLI) problem:

    Premise: "Russia deploys troops near Ukraine border"
    Hypothesis: "This text is about military escalation"
    → Model predicts: ENTAILMENT (yes) with high confidence

The model evaluates each candidate label as a hypothesis and
returns confidence scores for each.

USAGE:
------
    from src.analysis.nlp_classifier import EventClassifier

    classifier = EventClassifier()
    result = classifier.classify_event(event_dict)
    print(result.category)  # "military_escalation"
    print(result.confidence)  # 0.87
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Optional
import logging

import numpy as np
import pandas as pd

from src.db.connection import get_session
from src.db.queries import get_events_by_date_range

logger = logging.getLogger(__name__)

# Defense-relevant event categories
DEFENSE_CATEGORIES = [
    "military escalation",
    "economic sanctions",
    "arms deal or military trade",
    "diplomatic alliance shift",
    "cyber attack or information warfare",
    "humanitarian crisis",
    "nuclear or WMD proliferation",
    "territorial dispute",
    "peace negotiation or ceasefire",
    "political instability or coup",
]


@dataclass
class ClassificationResult:
    """Result of classifying a single event."""
    event_id: int
    event_text: str
    category: str
    confidence: float
    all_scores: dict[str, float]
    event_date: Optional[date] = None


@dataclass
class ClassificationReport:
    """Summary of batch classification."""
    total_events: int
    category_counts: dict[str, int]
    avg_confidence: float
    results: list[ClassificationResult]


class EventClassifier:
    """
    Zero-shot event classifier using Hugging Face transformers.

    Uses facebook/bart-large-mnli for inference without training data.
    Falls back to a simpler keyword-based approach if transformers
    is not available.
    """

    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        self.model_name = model_name
        self._pipeline = None
        self._embedder = None

    def _get_pipeline(self):
        """Lazy-load the classification pipeline."""
        if self._pipeline is not None:
            return self._pipeline

        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=-1,  # CPU
            )
            logger.info(f"Loaded zero-shot classifier: {self.model_name}")
            return self._pipeline

        except ImportError:
            logger.warning(
                "transformers not available. Install: pip install transformers"
            )
            return None
        except Exception as e:
            logger.error(f"Failed to load classifier: {e}")
            return None

    def _get_embedder(self):
        """Get the event embedder for text generation."""
        if self._embedder is not None:
            return self._embedder

        try:
            from src.analysis.text_embeddings import EventEmbedder

            self._embedder = EventEmbedder()
            return self._embedder
        except ImportError:
            return None

    def _event_to_text(self, event: dict) -> str:
        """Convert event dict to natural language text."""
        embedder = self._get_embedder()
        if embedder:
            return embedder.event_to_text(event)

        # Fallback: basic text construction
        actor1 = event.get("actor1_name") or event.get("actor1_code") or "Unknown"
        actor2 = event.get("actor2_name") or event.get("actor2_code") or "another party"
        return f"{actor1} interacted with {actor2}"

    def classify_event(
        self,
        event: dict,
        categories: list[str] = None,
    ) -> ClassificationResult:
        """
        Classify a single event into defense-relevant categories.

        Args:
            event: Event dictionary with actor, code, and location fields
            categories: Custom categories (default: DEFENSE_CATEGORIES)

        Returns:
            ClassificationResult with category, confidence, and all scores
        """
        if categories is None:
            categories = DEFENSE_CATEGORIES

        text = self._event_to_text(event)
        pipe = self._get_pipeline()

        if pipe is not None:
            result = pipe(text, candidate_labels=categories, multi_label=False)
            scores = dict(zip(result["labels"], result["scores"]))
            top_category = result["labels"][0]
            top_confidence = result["scores"][0]
        else:
            # Keyword fallback when transformers isn't available
            scores, top_category, top_confidence = self._keyword_classify(
                text, event, categories
            )

        return ClassificationResult(
            event_id=event.get("id", 0),
            event_text=text,
            category=top_category,
            confidence=top_confidence,
            all_scores=scores,
            event_date=event.get("event_date"),
        )

    def _keyword_classify(
        self,
        text: str,
        event: dict,
        categories: list[str],
    ) -> tuple[dict, str, float]:
        """Keyword-based fallback classifier."""
        text_lower = text.lower()
        root_code = str(event.get("event_root_code", "01"))

        scores = {cat: 0.1 for cat in categories}

        # Map CAMEO codes to categories
        code_map = {
            "14": "military escalation",
            "15": "military escalation",
            "18": "military escalation",
            "19": "military escalation",
            "20": "military escalation",
            "12": "economic sanctions",
            "13": "diplomatic alliance shift",
            "03": "peace negotiation or ceasefire",
            "04": "peace negotiation or ceasefire",
            "05": "diplomatic alliance shift",
            "06": "diplomatic alliance shift",
            "10": "political instability or coup",
        }

        if root_code in code_map:
            category = code_map[root_code]
            if category in scores:
                scores[category] = 0.7

        # Keyword boosts
        keyword_map = {
            "military": "military escalation",
            "troops": "military escalation",
            "attack": "military escalation",
            "sanction": "economic sanctions",
            "embargo": "economic sanctions",
            "arms": "arms deal or military trade",
            "weapon": "arms deal or military trade",
            "cyber": "cyber attack or information warfare",
            "nuclear": "nuclear or WMD proliferation",
            "humanitarian": "humanitarian crisis",
            "refugee": "humanitarian crisis",
            "territory": "territorial dispute",
            "border": "territorial dispute",
            "peace": "peace negotiation or ceasefire",
            "ceasefire": "peace negotiation or ceasefire",
            "coup": "political instability or coup",
            "protest": "political instability or coup",
        }

        for keyword, category in keyword_map.items():
            if keyword in text_lower and category in scores:
                scores[category] = max(scores[category], 0.6)

        top_category = max(scores, key=scores.get)
        top_confidence = scores[top_category]

        return scores, top_category, top_confidence

    def classify_batch(
        self,
        events: list[dict],
        categories: list[str] = None,
    ) -> ClassificationReport:
        """
        Classify multiple events and generate a report.

        Args:
            events: List of event dictionaries
            categories: Custom categories (default: DEFENSE_CATEGORIES)

        Returns:
            ClassificationReport with counts, averages, and individual results
        """
        results = []
        for event in events:
            try:
                result = self.classify_event(event, categories)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to classify event {event.get('id')}: {e}")

        if not results:
            return ClassificationReport(
                total_events=0,
                category_counts={},
                avg_confidence=0.0,
                results=[],
            )

        # Count categories
        category_counts: dict[str, int] = {}
        for r in results:
            category_counts[r.category] = category_counts.get(r.category, 0) + 1

        avg_confidence = sum(r.confidence for r in results) / len(results)

        return ClassificationReport(
            total_events=len(results),
            category_counts=category_counts,
            avg_confidence=avg_confidence,
            results=results,
        )

    def classify_date_range(
        self,
        start_date: date,
        end_date: date,
        limit: int = 100,
    ) -> ClassificationReport:
        """
        Classify events from the database for a date range.

        Fetches events from the database and classifies them.
        """
        with get_session() as session:
            events_raw = get_events_by_date_range(
                session, start_date, end_date, limit=limit
            )

        if not events_raw:
            return ClassificationReport(
                total_events=0,
                category_counts={},
                avg_confidence=0.0,
                results=[],
            )

        # Convert ORM objects to dicts
        events = []
        for e in events_raw:
            events.append({
                "id": e.id,
                "event_date": e.event_date,
                "event_root_code": e.event_root_code,
                "actor1_name": e.actor1_name,
                "actor1_code": e.actor1_code,
                "actor2_name": e.actor2_name,
                "actor2_code": e.actor2_code,
                "action_geo_name": getattr(e, "action_geo_name", None),
                "action_geo_country_code": getattr(e, "action_geo_country_code", None),
                "goldstein_scale": e.goldstein_scale,
                "avg_tone": e.avg_tone,
                "num_mentions": e.num_mentions,
            })

        return self.classify_batch(events)
