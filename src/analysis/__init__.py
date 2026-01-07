"""
Analysis Module.

This module provides tools for analyzing relationships between
geopolitical events and market movements.

TWO VERSIONS AVAILABLE:
-----------------------
1. LEARNING VERSIONS (educational, show the math):
   - EventStudy, EventRegression, MarketClassifier
   - Written from scratch to understand the algorithms
   - Great for learning and interviews

2. PRODUCTION VERSIONS (industry-standard libraries):
   - ProductionEventStudy, ProductionRegression, ProductionClassifier
   - Use sklearn, statsmodels, scipy
   - What you'd use in a real job

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

USAGE (Learning Versions):
--------------------------
    from src.analysis import EventStudy, EventRegression, MarketClassifier

    # These show all the math step-by-step
    study = EventStudy()
    result = study.analyze_event(event_id, symbol, event_date)

USAGE (Production Versions):
----------------------------
    from src.analysis import ProductionEventStudy, ProductionRegression, ProductionClassifier

    # These use sklearn/statsmodels - what you'd use at work
    classifier = ProductionClassifier()
    classifier.train_all_markets(start_date, end_date)

    reg = ProductionRegression()
    result = reg.analyze(symbol, start_date, end_date)
    print(result.summary)  # Beautiful statsmodels output
"""

# ═══════════════════════════════════════════════════════════════════════════════
# LEARNING VERSIONS (Educational - show the math)
# ═══════════════════════════════════════════════════════════════════════════════
from src.analysis.event_study import EventStudy, EventStudyResult, explain_result
from src.analysis.anomaly_detection import AnomalyDetector, Anomaly, explain_anomaly
from src.analysis.regression import EventRegression, RegressionResult, explain_regression
from src.analysis.classification import (
    MarketClassifier,
    ClassificationResult,
    ModelMetrics,
    explain_classification,
)

# ═══════════════════════════════════════════════════════════════════════════════
# PRODUCTION VERSIONS (Industry-standard libraries)
# ═══════════════════════════════════════════════════════════════════════════════
from src.analysis.production_classifier import (
    ProductionClassifier,
    ProdClassificationResult,
    ProdModelMetrics,
)
from src.analysis.production_regression import (
    ProductionRegression,
    ProdRegressionResult,
    print_interpretation,
)
from src.analysis.production_event_study import (
    ProductionEventStudy,
    ProdEventStudyResult,
    run_quick_event_study,
)

__all__ = [
    # ═══════════════════════════════════════════════════════════════════════════
    # LEARNING VERSIONS
    # ═══════════════════════════════════════════════════════════════════════════
    # Event Study (learning)
    "EventStudy",
    "EventStudyResult",
    "explain_result",
    # Anomaly Detection
    "AnomalyDetector",
    "Anomaly",
    "explain_anomaly",
    # Regression (learning)
    "EventRegression",
    "RegressionResult",
    "explain_regression",
    # Classification (learning)
    "MarketClassifier",
    "ClassificationResult",
    "ModelMetrics",
    "explain_classification",
    # ═══════════════════════════════════════════════════════════════════════════
    # PRODUCTION VERSIONS
    # ═══════════════════════════════════════════════════════════════════════════
    # Event Study (production)
    "ProductionEventStudy",
    "ProdEventStudyResult",
    "run_quick_event_study",
    # Regression (production)
    "ProductionRegression",
    "ProdRegressionResult",
    "print_interpretation",
    # Classification (production)
    "ProductionClassifier",
    "ProdClassificationResult",
    "ProdModelMetrics",
]
