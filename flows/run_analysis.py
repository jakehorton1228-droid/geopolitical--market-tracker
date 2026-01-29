"""
Analysis Flow.

Runs statistical analysis on ingested data:
- Event studies (CAR calculation)
- Anomaly detection
- ML predictions (Gradient Boosting & LSTM)
- Drift detection (feature distribution shifts)
- Data quality checks

USAGE:
------
    # Run directly
    python flows/run_analysis.py

    # Run via Prefect CLI
    prefect deployment run 'run-analysis/daily'

    # Run with ML predictions enabled
    python flows/run_analysis.py --ml

CONCEPTS:
---------
    Subflows: Flows can call other flows for modularity
    Artifacts: Store analysis outputs for later review
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datetime import date, timedelta
from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact

from src.db.connection import get_session
from src.db.models import Event, MarketData, AnalysisResult
from src.analysis.production_event_study import ProductionEventStudy
from src.analysis.production_anomaly import ProductionAnomalyDetector
from src.config.constants import TRACKED_SYMBOLS

# Advanced ML imports (lazy loaded to avoid slow startup)
_gradient_boost_classifier = None
_lstm_trainer = None


@task(name="get-recent-events", description="Fetch events for analysis window")
def get_recent_events(start_date: date, end_date: date, min_mentions: int = 10) -> list[dict]:
    """Fetch significant events for analysis."""
    logger = get_run_logger()

    with get_session() as session:
        events = session.query(Event).filter(
            Event.event_date >= start_date,
            Event.event_date <= end_date,
            Event.num_mentions >= min_mentions,
        ).order_by(Event.event_date).all()

        event_dicts = [
            {
                "id": e.id,
                "event_date": e.event_date,
                "event_root_code": e.event_root_code,
                "goldstein_scale": e.goldstein_scale,
                "num_mentions": e.num_mentions,
                "actor1_code": e.actor1_code,
                "actor2_code": e.actor2_code,
                "action_geo_country_code": e.action_geo_country_code,
            }
            for e in events
        ]

    logger.info(f"Found {len(event_dicts)} significant events")
    return event_dicts


@task(name="get-market-data", description="Fetch market data for analysis")
def get_market_data_for_analysis(
    symbols: list[str],
    start_date: date,
    end_date: date,
) -> dict[str, list[dict]]:
    """Fetch market data organized by symbol."""
    logger = get_run_logger()

    # Extend date range for event study windows
    extended_start = start_date - timedelta(days=30)
    extended_end = end_date + timedelta(days=10)

    with get_session() as session:
        records = session.query(MarketData).filter(
            MarketData.symbol.in_(symbols),
            MarketData.date >= extended_start,
            MarketData.date <= extended_end,
        ).order_by(MarketData.symbol, MarketData.date).all()

        by_symbol = {}
        for r in records:
            if r.symbol not in by_symbol:
                by_symbol[r.symbol] = []
            by_symbol[r.symbol].append({
                "date": r.date,
                "close": r.close,
                "daily_return": r.daily_return,
            })

    logger.info(f"Loaded market data for {len(by_symbol)} symbols")
    return by_symbol


@task(name="run-event-studies", description="Calculate CAR for events")
def run_event_studies(
    events: list[dict],
    market_data: dict[str, list[dict]],
    symbols: list[str],
) -> list[dict]:
    """
    Run event study analysis.

    For each significant event, calculate the Cumulative Abnormal Return (CAR)
    around the event date for each tracked symbol.
    """
    logger = get_run_logger()
    results = []

    study = ProductionEventStudy()

    for event in events[:50]:  # Limit to avoid excessive processing
        event_date = event["event_date"]

        for symbol in symbols[:10]:  # Analyze top 10 symbols per event
            if symbol not in market_data:
                continue

            prices = market_data[symbol]
            if len(prices) < 30:  # Need enough history
                continue

            try:
                result = study.calculate_car(
                    prices=prices,
                    event_date=event_date,
                    estimation_window=20,
                    event_window=(-2, 5),
                )

                if result:
                    results.append({
                        "event_id": event["id"],
                        "symbol": symbol,
                        "analysis_type": "event_study",
                        "car": result.get("car"),
                        "t_stat": result.get("t_statistic"),
                        "p_value": result.get("p_value"),
                        "is_significant": result.get("p_value", 1) < 0.05,
                    })
            except Exception as e:
                logger.warning(f"Event study failed for {symbol}/{event_date}: {e}")

    logger.info(f"Completed {len(results)} event studies")
    return results


@task(name="run-anomaly-detection", description="Detect market anomalies")
def run_anomaly_detection(
    events: list[dict],
    market_data: dict[str, list[dict]],
    symbols: list[str],
) -> list[dict]:
    """
    Detect anomalies in market data.

    Anomaly types:
    - Unexplained moves: Big market move with no corresponding event
    - Muted responses: Big event with surprisingly small market reaction
    """
    logger = get_run_logger()
    results = []

    detector = ProductionAnomalyDetector()

    for symbol in symbols[:10]:
        if symbol not in market_data:
            continue

        prices = market_data[symbol]
        if len(prices) < 30:
            continue

        try:
            anomalies = detector.detect_anomalies(
                prices=prices,
                events=events,
                z_threshold=2.5,
            )

            for anomaly in anomalies:
                results.append({
                    "symbol": symbol,
                    "date": anomaly.get("date"),
                    "analysis_type": "anomaly_detection",
                    "anomaly_type": anomaly.get("type"),
                    "z_score": anomaly.get("z_score"),
                    "is_anomaly": True,
                })
        except Exception as e:
            logger.warning(f"Anomaly detection failed for {symbol}: {e}")

    logger.info(f"Detected {len(results)} anomalies")
    return results


# =============================================================================
# ADVANCED ML TASKS
# =============================================================================

@task(name="run-gradient-boost", description="Train and predict with XGBoost/LightGBM")
def run_gradient_boost_predictions(
    symbols: list[str],
    start_date: date,
    end_date: date,
) -> list[dict]:
    """
    Train gradient boosting models and generate predictions.

    This trains both XGBoost and LightGBM classifiers for each symbol
    and returns predictions along with model comparison metrics.
    """
    logger = get_run_logger()
    results = []

    try:
        from src.analysis.gradient_boost_classifier import GradientBoostClassifier

        classifier = GradientBoostClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
        )

        for symbol in symbols[:10]:  # Limit symbols for performance
            try:
                comparison = classifier.train_and_compare(symbol, start_date, end_date)

                if comparison:
                    # Record model performance
                    results.append({
                        "symbol": symbol,
                        "analysis_type": "gradient_boost",
                        "xgb_cv_accuracy": comparison.xgboost_metrics.cv_accuracy if comparison.xgboost_metrics else None,
                        "lgb_cv_accuracy": comparison.lightgbm_metrics.cv_accuracy if comparison.lightgbm_metrics else None,
                        "winner": comparison.winner,
                        "top_features": list(comparison.feature_importance_xgb.keys())[:3],
                    })

                    logger.info(
                        f"{symbol}: XGB={comparison.xgboost_metrics.cv_accuracy:.2%}, "
                        f"LGB={comparison.lightgbm_metrics.cv_accuracy:.2%}, "
                        f"Winner={comparison.winner}"
                    )
            except Exception as e:
                logger.warning(f"Gradient boost failed for {symbol}: {e}")

    except ImportError as e:
        logger.warning(f"Gradient boost not available: {e}. Install xgboost and lightgbm.")

    logger.info(f"Completed gradient boost analysis for {len(results)} symbols")
    return results


@task(name="run-lstm-predictions", description="Train LSTM and predict market direction")
def run_lstm_predictions(
    symbols: list[str],
    start_date: date,
    end_date: date,
) -> list[dict]:
    """
    Train LSTM models and generate sequence-based predictions.

    This creates sequences from historical data, trains an LSTM,
    and evaluates its predictive performance.
    """
    logger = get_run_logger()
    results = []

    try:
        from src.analysis.sequence_dataset import MarketSequenceDataset
        from src.analysis.lstm_model import MarketLSTM, LSTMTrainer

        for symbol in symbols[:5]:  # LSTM is slower, limit symbols
            try:
                # Create dataset
                dataset = MarketSequenceDataset(
                    symbol=symbol,
                    sequence_length=20,
                    start_date=start_date,
                    end_date=end_date,
                    test_ratio=0.2,
                )

                stats = dataset.get_stats()
                if stats.train_size < 50:
                    logger.warning(f"Insufficient data for LSTM on {symbol}: {stats.train_size} samples")
                    continue

                # Create and train model
                model = MarketLSTM(
                    input_size=stats.n_features,
                    hidden_size=64,
                    num_layers=2,
                    dropout=0.2,
                )

                trainer = LSTMTrainer(model, learning_rate=0.001)
                train_loader, test_loader = dataset.get_dataloaders(batch_size=32)

                # Train (fewer epochs for flow efficiency)
                history = trainer.fit(
                    train_loader,
                    test_loader,
                    epochs=30,
                    early_stopping_patience=5,
                    verbose=False,
                )

                # Evaluate
                test_results = trainer.predict(test_loader)

                results.append({
                    "symbol": symbol,
                    "analysis_type": "lstm",
                    "test_accuracy": test_results.accuracy,
                    "test_f1": test_results.f1,
                    "test_auc": test_results.auc,
                    "best_epoch": history.best_epoch,
                    "train_samples": stats.train_size,
                    "test_samples": stats.test_size,
                })

                logger.info(
                    f"{symbol} LSTM: Acc={test_results.accuracy:.2%}, "
                    f"F1={test_results.f1:.2%}, AUC={test_results.auc:.3f}"
                )

            except Exception as e:
                logger.warning(f"LSTM failed for {symbol}: {e}")

    except ImportError as e:
        logger.warning(f"LSTM not available: {e}. Install torch.")

    logger.info(f"Completed LSTM analysis for {len(results)} symbols")
    return results


@task(name="run-drift-detection", description="Detect feature distribution drift")
def run_drift_detection(
    symbols: list[str],
    start_date: date,
    end_date: date,
) -> list[dict]:
    """
    Run drift detection across symbols.

    Checks if feature distributions have shifted significantly
    compared to a historical baseline, which may indicate the
    need for model retraining.
    """
    logger = get_run_logger()
    results = []

    try:
        from src.analysis.drift_detection import DriftDetector

        detector = DriftDetector(baseline_window=60, test_window=14)

        for symbol in symbols[:10]:
            try:
                report = detector.analyze(symbol, start_date, end_date)
                if report is None:
                    continue

                results.append({
                    "symbol": symbol,
                    "analysis_type": "drift_detection",
                    "drift_detected": report.drift_detected,
                    "severity": report.overall_severity,
                    "drifted_features": report.drifted_features,
                    "total_features": report.total_features,
                })

                if report.drift_detected:
                    logger.warning(
                        f"{symbol}: Drift detected! Severity={report.overall_severity}, "
                        f"{report.drifted_features}/{report.total_features} features drifted"
                    )
            except Exception as e:
                logger.warning(f"Drift detection failed for {symbol}: {e}")

    except ImportError as e:
        logger.warning(f"Drift detection not available: {e}")

    logger.info(f"Drift detection complete: {len(results)} symbols checked")
    return results


@task(name="run-data-quality", description="Check data quality across sources")
def run_data_quality_checks(
    symbols: list[str],
    start_date: date,
    end_date: date,
) -> dict:
    """
    Run data quality checks and return a summary.

    Checks completeness, freshness, and validity of both
    market data and event data.
    """
    logger = get_run_logger()

    try:
        from src.analysis.data_quality import DataQualityChecker

        checker = DataQualityChecker()
        report = checker.check_all(start_date, end_date, symbols=symbols[:10])

        summary = {
            "overall_score": report.overall_score,
            "symbols_checked": report.symbols_checked,
            "symbols_with_issues": report.symbols_with_issues,
            "total_issues": report.total_issues,
            "high_severity": report.high_severity_issues,
            "medium_severity": report.medium_severity_issues,
            "low_severity": report.low_severity_issues,
        }

        if report.high_severity_issues > 0:
            logger.warning(
                f"Data quality: {report.high_severity_issues} high-severity issues found! "
                f"Overall score: {report.overall_score:.0%}"
            )
        else:
            logger.info(f"Data quality score: {report.overall_score:.0%}")

        return summary

    except ImportError as e:
        logger.warning(f"Data quality checks not available: {e}")
        return {}
    except Exception as e:
        logger.warning(f"Data quality checks failed: {e}")
        return {}


@task(name="store-results", description="Store analysis results in database")
def store_analysis_results(results: list[dict]) -> int:
    """Store analysis results in the database."""
    logger = get_run_logger()

    if not results:
        logger.info("No results to store")
        return 0

    with get_session() as session:
        for r in results:
            result = AnalysisResult(
                event_id=r.get("event_id"),
                symbol=r.get("symbol"),
                analysis_type=r.get("analysis_type"),
                result_value=r.get("car") or r.get("z_score"),
                p_value=r.get("p_value"),
                is_significant=r.get("is_significant", False),
                is_anomaly=r.get("is_anomaly", False),
                metadata_json={
                    "t_stat": r.get("t_stat"),
                    "anomaly_type": r.get("anomaly_type"),
                },
            )
            session.add(result)
        session.commit()

    logger.info(f"Stored {len(results)} analysis results")
    return len(results)


@task(name="create-analysis-report", description="Generate markdown report")
def create_analysis_report(
    event_study_results: list[dict],
    anomaly_results: list[dict],
    ml_results: list[dict] = None,
) -> str:
    """Create a markdown summary report."""
    logger = get_run_logger()

    ml_results = ml_results or []

    # Calculate statistics
    significant_cars = [r for r in event_study_results if r.get("is_significant")]
    avg_car = sum(r.get("car", 0) for r in significant_cars) / max(len(significant_cars), 1)

    report = f"""
# Analysis Report

## Event Study Results
- Total studies: {len(event_study_results)}
- Significant (p < 0.05): {len(significant_cars)}
- Average CAR (significant only): {avg_car:.4f}

## Top Significant Results
| Symbol | CAR | t-stat | p-value |
|--------|-----|--------|---------|
"""

    for r in sorted(significant_cars, key=lambda x: abs(x.get("car", 0)), reverse=True)[:10]:
        report += f"| {r.get('symbol')} | {r.get('car', 0):.4f} | {r.get('t_stat', 0):.2f} | {r.get('p_value', 0):.4f} |\n"

    report += f"""

## Anomaly Detection
- Total anomalies detected: {len(anomaly_results)}

## Anomalies by Type
"""

    anomaly_types = {}
    for a in anomaly_results:
        atype = a.get("anomaly_type", "unknown")
        anomaly_types[atype] = anomaly_types.get(atype, 0) + 1

    for atype, count in anomaly_types.items():
        report += f"- {atype}: {count}\n"

    # ML Results section
    if ml_results:
        gb_results = [r for r in ml_results if r.get("analysis_type") == "gradient_boost"]
        lstm_results = [r for r in ml_results if r.get("analysis_type") == "lstm"]

        if gb_results:
            report += f"""

## Gradient Boosting Predictions
| Symbol | XGBoost CV | LightGBM CV | Winner | Top Features |
|--------|-----------|-------------|--------|--------------|
"""
            for r in sorted(gb_results, key=lambda x: x.get("xgb_cv_accuracy", 0), reverse=True):
                xgb_acc = r.get("xgb_cv_accuracy", 0)
                lgb_acc = r.get("lgb_cv_accuracy", 0)
                features = ", ".join(r.get("top_features", [])[:3])
                report += f"| {r.get('symbol')} | {xgb_acc:.1%} | {lgb_acc:.1%} | {r.get('winner')} | {features} |\n"

            avg_xgb = sum(r.get("xgb_cv_accuracy", 0) for r in gb_results) / max(len(gb_results), 1)
            avg_lgb = sum(r.get("lgb_cv_accuracy", 0) for r in gb_results) / max(len(gb_results), 1)
            report += f"\n**Average CV Accuracy:** XGBoost={avg_xgb:.1%}, LightGBM={avg_lgb:.1%}\n"

        if lstm_results:
            report += f"""

## LSTM Sequence Predictions
| Symbol | Accuracy | F1 Score | AUC | Samples |
|--------|----------|----------|-----|---------|
"""
            for r in sorted(lstm_results, key=lambda x: x.get("test_accuracy", 0), reverse=True):
                report += f"| {r.get('symbol')} | {r.get('test_accuracy', 0):.1%} | {r.get('test_f1', 0):.1%} | {r.get('test_auc', 0):.3f} | {r.get('train_samples', 0)} |\n"

            avg_acc = sum(r.get("test_accuracy", 0) for r in lstm_results) / max(len(lstm_results), 1)
            avg_auc = sum(r.get("test_auc", 0) for r in lstm_results) / max(len(lstm_results), 1)
            report += f"\n**Average Performance:** Accuracy={avg_acc:.1%}, AUC={avg_auc:.3f}\n"
            report += f"**Baseline (random):** Accuracy=50%, AUC=0.500\n"

    logger.info("Generated analysis report")
    return report


@flow(
    name="run-analysis",
    description="Run statistical analysis on market and event data",
    version="2.0.0",
)
def run_analysis(
    days_back: int = 30,
    end_date: date | None = None,
    symbols: list[str] | None = None,
    store_results: bool = True,
    run_ml: bool = True,
) -> dict:
    """
    Main analysis flow.

    Runs event studies, anomaly detection, and ML predictions on recent data,
    then stores results and creates a report artifact.

    Parameters
    ----------
    days_back : int
        Analysis window in days (default: 30)
    end_date : date, optional
        End date for analysis (default: today)
    symbols : list[str], optional
        Symbols to analyze (default: all tracked)
    store_results : bool
        Whether to store results in database (default: True)
    run_ml : bool
        Whether to run ML predictions (gradient boost + LSTM) (default: True)

    Returns
    -------
    dict
        Summary of analysis results
    """
    logger = get_run_logger()

    # Defaults
    if end_date is None:
        end_date = date.today()
    if symbols is None:
        symbols = TRACKED_SYMBOLS

    start_date = end_date - timedelta(days=days_back)

    logger.info(f"Starting analysis: {start_date} to {end_date}")
    logger.info(f"ML predictions: {'enabled' if run_ml else 'disabled'}")

    # Fetch data
    events = get_recent_events(start_date, end_date)
    market_data = get_market_data_for_analysis(symbols, start_date, end_date)

    if not events:
        logger.warning("No events found for analysis")
        return {"status": "no_data", "message": "No events found"}

    # Run traditional analyses
    event_study_results = run_event_studies(events, market_data, symbols)
    anomaly_results = run_anomaly_detection(events, market_data, symbols)

    # Run ML predictions (if enabled)
    ml_results = []
    if run_ml:
        logger.info("Running ML predictions...")

        # Gradient Boosting (XGBoost + LightGBM)
        gb_results = run_gradient_boost_predictions(symbols, start_date, end_date)
        ml_results.extend(gb_results)

        # LSTM sequence predictions
        lstm_results = run_lstm_predictions(symbols, start_date, end_date)
        ml_results.extend(lstm_results)

    # Run monitoring tasks
    logger.info("Running monitoring checks...")
    drift_results = run_drift_detection(symbols, start_date, end_date)
    dq_summary = run_data_quality_checks(symbols, start_date, end_date)

    # Store results
    if store_results:
        all_results = event_study_results + anomaly_results
        stored_count = store_analysis_results(all_results)
    else:
        stored_count = 0

    # Create report artifact (now includes ML results and monitoring)
    report = create_analysis_report(event_study_results, anomaly_results, ml_results)

    create_markdown_artifact(
        key="analysis-report",
        markdown=report,
        description=f"Analysis report for {start_date} to {end_date}",
    )

    # Summary statistics
    gb_results = [r for r in ml_results if r.get("analysis_type") == "gradient_boost"]
    lstm_results = [r for r in ml_results if r.get("analysis_type") == "lstm"]

    summary = {
        "date_range": f"{start_date} to {end_date}",
        "events_analyzed": len(events),
        "event_studies": len(event_study_results),
        "significant_results": sum(1 for r in event_study_results if r.get("is_significant")),
        "anomalies_detected": len(anomaly_results),
        "results_stored": stored_count,
        "ml_predictions": {
            "gradient_boost_symbols": len(gb_results),
            "lstm_symbols": len(lstm_results),
            "avg_gb_accuracy": sum(r.get("xgb_cv_accuracy", 0) for r in gb_results) / max(len(gb_results), 1) if gb_results else None,
            "avg_lstm_accuracy": sum(r.get("test_accuracy", 0) for r in lstm_results) / max(len(lstm_results), 1) if lstm_results else None,
        } if run_ml else None,
        "monitoring": {
            "drift_symbols_checked": len(drift_results),
            "drift_detected_count": sum(1 for d in drift_results if d.get("drift_detected")),
            "data_quality_score": dq_summary.get("overall_score"),
            "data_quality_issues": dq_summary.get("total_issues", 0),
        },
    }

    logger.info(f"Analysis complete: {summary}")
    return summary


def create_deployment():
    """Create Prefect deployment for scheduled runs."""
    from prefect.deployments import Deployment
    from prefect.server.schemas.schedules import CronSchedule

    deployment = Deployment.build_from_flow(
        flow=run_analysis,
        name="daily",
        version="1.0.0",
        description="Daily analysis run - after data ingestion completes",
        schedule=CronSchedule(cron="0 20 * * *"),  # 8 PM UTC daily
        parameters={"days_back": 7, "store_results": True},
        tags=["analysis", "daily"],
    )

    deployment.apply()
    print("Deployment created: run-analysis/daily")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Analysis")
    parser.add_argument("--deploy", action="store_true", help="Create Prefect deployment")
    parser.add_argument("--days", type=int, default=30, help="Days to analyze")
    parser.add_argument("--no-store", action="store_true", help="Don't store results")
    parser.add_argument("--ml", action="store_true", default=True, help="Run ML predictions (default: True)")
    parser.add_argument("--no-ml", action="store_true", help="Disable ML predictions")

    args = parser.parse_args()

    if args.deploy:
        create_deployment()
    else:
        result = run_analysis(
            days_back=args.days,
            store_results=not args.no_store,
            run_ml=not args.no_ml,
        )
        print(f"Result: {result}")
