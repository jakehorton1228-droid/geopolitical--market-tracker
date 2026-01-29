"""Tests for the NLP classifier module."""

import sys
from pathlib import Path
from datetime import date, timedelta
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _event_dict(**overrides) -> dict:
    """Create a plain dict event suitable for EventClassifier."""
    base = {
        "id": 1,
        "event_date": date.today() - timedelta(days=5),
        "event_root_code": "14",
        "goldstein_scale": -3.0,
        "num_mentions": 25,
        "avg_tone": -2.5,
        "actor1_code": "USA",
        "actor2_code": "RUS",
        "actor1_name": "United States",
        "actor2_name": "Russia",
        "action_geo_country_code": "UKR",
        "action_geo_name": None,
    }
    base.update(overrides)
    return base


class TestEventClassifier:
    """Tests for the EventClassifier class."""

    def _make_classifier(self, use_transformer=False):
        from src.analysis.nlp_classifier import EventClassifier
        classifier = EventClassifier()
        if not use_transformer:
            # Force keyword fallback by making _get_pipeline return None
            classifier._get_pipeline = lambda: None
        return classifier

    def test_keyword_classify_military_event(self):
        """Should classify military conflicts correctly via keywords."""
        classifier = self._make_classifier()

        event = _event_dict(
            event_root_code="19",
            goldstein_scale=-8.0,
            actor1_name="Russia",
            actor2_name="Ukraine",
        )

        result = classifier.classify_event(event)

        assert result is not None
        assert result.category == "military escalation"
        assert result.confidence > 0

    def test_keyword_classify_sanctions(self):
        """Should classify sanctions events correctly."""
        classifier = self._make_classifier()

        event = _event_dict(
            event_root_code="12",
            goldstein_scale=-5.0,
            actor1_name="United States",
            actor2_name="Iran",
        )

        result = classifier.classify_event(event)

        assert result is not None
        assert result.category == "economic sanctions"

    def test_keyword_classify_cooperation(self):
        """Should classify cooperation/peace events correctly."""
        classifier = self._make_classifier()

        event = _event_dict(
            event_root_code="04",
            goldstein_scale=5.0,
            actor1_name="Israel",
            actor2_name="Saudi Arabia",
        )

        result = classifier.classify_event(event)

        assert result is not None
        assert result.category == "peace negotiation or ceasefire"

    def test_classify_batch(self):
        """Should classify a batch of events."""
        classifier = self._make_classifier()

        events = [
            _event_dict(id=1, event_root_code="19", goldstein_scale=-7.0),
            _event_dict(id=2, event_root_code="04", goldstein_scale=4.0),
            _event_dict(id=3, event_root_code="12", goldstein_scale=-3.0),
        ]

        report = classifier.classify_batch(events)

        assert report.total_events == 3
        assert len(report.results) == 3
        categories = [r.category for r in report.results]
        assert "military escalation" in categories

    @patch("src.analysis.nlp_classifier.get_session")
    @patch("src.analysis.nlp_classifier.get_events_by_date_range")
    def test_classify_date_range(self, mock_events, mock_session):
        """Should classify events in a date range and return a report."""
        mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_session.return_value.__exit__ = MagicMock(return_value=False)

        from tests.conftest import make_event_series
        mock_events.return_value = make_event_series(10)

        classifier = self._make_classifier()
        report = classifier.classify_date_range(
            date.today() - timedelta(days=10),
            date.today(),
            limit=10,
        )

        assert report is not None
        assert report.total_events == 10
        assert report.avg_confidence > 0
        assert len(report.category_counts) > 0

    def test_classify_date_range_no_events(self):
        """Should handle empty date range."""
        classifier = self._make_classifier()

        with patch("src.analysis.nlp_classifier.get_session") as mock_session, \
             patch("src.analysis.nlp_classifier.get_events_by_date_range") as mock_events:
            mock_session.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_session.return_value.__exit__ = MagicMock(return_value=False)
            mock_events.return_value = []

            report = classifier.classify_date_range(
                date.today() - timedelta(days=10),
                date.today(),
            )

            assert report.total_events == 0


class TestClassificationResult:
    """Tests for the ClassificationResult dataclass."""

    def test_result_creation(self):
        from src.analysis.nlp_classifier import ClassificationResult

        result = ClassificationResult(
            event_id=1,
            event_text="Russia attacks Ukraine",
            event_date=date.today(),
            category="military_escalation",
            confidence=0.95,
            all_scores={"military_escalation": 0.95, "sanctions": 0.03},
        )

        assert result.category == "military_escalation"
        assert result.confidence == 0.95
        assert len(result.all_scores) == 2
