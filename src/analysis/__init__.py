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
   - Gradient boosted models for direction prediction
   - Covers all 33 tracked markets
   - Provides interpretable predictions with confidence scores

USAGE:
------
    from src.analysis import ProductionEventStudy, ProductionRegression

    study = ProductionEventStudy()
    result = study.analyze_event(event_id, symbol, event_date)

    reg = ProductionRegression()
    result = reg.analyze(symbol, start_date, end_date)
    print(result.summary)
"""

# ═══════════════════════════════════════════════════════════════════════════════
# PRODUCTION ANALYSIS MODULES
# ═══════════════════════════════════════════════════════════════════════════════
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
from src.analysis.production_anomaly import (
    ProductionAnomalyDetector,
    ProdAnomaly,
    AnomalyReport,
    run_quick_anomaly_detection,
)

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════
from src.analysis.logging_config import (
    setup_logging,
    setup_debug_logging,
    setup_quiet_logging,
    setup_file_logging,
    get_analysis_logger,
    AnalysisTimer,
)

__all__ = [
    # Event Study
    "ProductionEventStudy",
    "ProdEventStudyResult",
    "run_quick_event_study",
    # Regression
    "ProductionRegression",
    "ProdRegressionResult",
    "print_interpretation",
    # Anomaly Detection
    "ProductionAnomalyDetector",
    "ProdAnomaly",
    "AnomalyReport",
    "run_quick_anomaly_detection",
    # Logging
    "setup_logging",
    "setup_debug_logging",
    "setup_quiet_logging",
    "setup_file_logging",
    "get_analysis_logger",
    "AnalysisTimer",
]
