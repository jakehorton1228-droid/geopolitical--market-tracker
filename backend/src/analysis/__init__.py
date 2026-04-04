"""
Analysis Module.

This module provides tools for analyzing relationships between
geopolitical events and market movements.

ANALYSES:
---------
1. Event Study: "Did this event move the market?"
   - Calculates Cumulative Abnormal Returns (CAR)
   - Tests statistical significance

2. Anomaly Detection: "Why did the market move?"
   - Finds unexplained market moves
   - Finds muted responses to major events

3. Regression: "What event characteristics predict market moves?"
   - OLS and logistic regression
   - Identifies which factors matter most

4. Correlation: "How do events and prices move together?"
   - Pearson/Spearman correlation analysis
   - Rolling correlation windows

5. Historical Patterns: "What happened historically?"
   - Conditional probability lookups
   - "When X happens, Y goes UP Z% of the time"

6. ML Feature Pipeline: "Train models to predict market impact"
   - Flat features for tree models (XGBoost, LightGBM, RF, LogReg, MLP)
   - Windowed sequences for neural nets (CNN, LSTM)
   - Time-series aware train/val/test split
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

# ═══════════════════════════════════════════════════════════════════════════════
# ML FEATURE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
from src.analysis.ml_features import MLFeaturePipeline

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
    # ML Feature Pipeline
    "MLFeaturePipeline",
    # Logging
    "setup_logging",
    "setup_debug_logging",
    "setup_quiet_logging",
    "setup_file_logging",
    "get_analysis_logger",
    "AnalysisTimer",
]
