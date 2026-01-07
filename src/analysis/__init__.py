"""
Analysis Module.

This module provides tools for analyzing relationships between
geopolitical events and market movements.

FOUR MAIN ANALYSES:
-------------------
1. Event Study: "Did this event move the market?"
   - Calculates Cumulative Abnormal Returns (CAR)
   - Tests statistical significance

2. Anomaly Detection: "Why did the market move?"
   - Finds unexplained market moves
   - Finds muted responses to major events

3. Regression: "What event characteristics predict market moves?"
   - Multiple regression analysis
   - Identifies which factors matter most

4. Classification: "Will the market go UP or DOWN?"
   - Logistic regression for direction prediction
   - Covers all 33 tracked markets
   - Provides interpretable predictions with confidence scores

USAGE:
------
    from src.analysis import EventStudy, AnomalyDetector, EventRegression, MarketClassifier

    # Event Study
    study = EventStudy()
    result = study.analyze_event(event_id, symbol, event_date)

    # Anomaly Detection
    detector = AnomalyDetector()
    anomalies = detector.detect_unexplained_moves(symbol, start, end)

    # Regression
    reg = EventRegression()
    result = reg.analyze_event_impact(symbol, start, end)

    # Classification
    classifier = MarketClassifier()
    results = classifier.train_all_markets(start_date, end_date)
    prediction = classifier.predict("CL=F", features)
"""

from src.analysis.event_study import EventStudy, EventStudyResult, explain_result
from src.analysis.anomaly_detection import AnomalyDetector, Anomaly, explain_anomaly
from src.analysis.regression import EventRegression, RegressionResult, explain_regression
from src.analysis.classification import (
    MarketClassifier,
    ClassificationResult,
    ModelMetrics,
    explain_classification,
)

__all__ = [
    # Event Study
    "EventStudy",
    "EventStudyResult",
    "explain_result",
    # Anomaly Detection
    "AnomalyDetector",
    "Anomaly",
    "explain_anomaly",
    # Regression
    "EventRegression",
    "RegressionResult",
    "explain_regression",
    # Classification
    "MarketClassifier",
    "ClassificationResult",
    "ModelMetrics",
    "explain_classification",
]
