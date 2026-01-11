"""
Analysis Flow.

Runs statistical analysis on ingested data:
- Event studies (CAR calculation)
- Anomaly detection
- Regression analysis

USAGE:
------
    # Run directly
    python flows/run_analysis.py

    # Run via Prefect CLI
    prefect deployment run 'run-analysis/daily'

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
) -> str:
    """Create a markdown summary report."""
    logger = get_run_logger()

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

    logger.info("Generated analysis report")
    return report


@flow(
    name="run-analysis",
    description="Run statistical analysis on market and event data",
    version="1.0.0",
)
def run_analysis(
    days_back: int = 30,
    end_date: date | None = None,
    symbols: list[str] | None = None,
    store_results: bool = True,
) -> dict:
    """
    Main analysis flow.

    Runs event studies and anomaly detection on recent data,
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

    # Fetch data
    events = get_recent_events(start_date, end_date)
    market_data = get_market_data_for_analysis(symbols, start_date, end_date)

    if not events:
        logger.warning("No events found for analysis")
        return {"status": "no_data", "message": "No events found"}

    # Run analyses
    event_study_results = run_event_studies(events, market_data, symbols)
    anomaly_results = run_anomaly_detection(events, market_data, symbols)

    # Store results
    if store_results:
        all_results = event_study_results + anomaly_results
        stored_count = store_analysis_results(all_results)
    else:
        stored_count = 0

    # Create report artifact
    report = create_analysis_report(event_study_results, anomaly_results)

    create_markdown_artifact(
        key="analysis-report",
        markdown=report,
        description=f"Analysis report for {start_date} to {end_date}",
    )

    summary = {
        "date_range": f"{start_date} to {end_date}",
        "events_analyzed": len(events),
        "event_studies": len(event_study_results),
        "significant_results": sum(1 for r in event_study_results if r.get("is_significant")),
        "anomalies_detected": len(anomaly_results),
        "results_stored": stored_count,
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

    args = parser.parse_args()

    if args.deploy:
        create_deployment()
    else:
        result = run_analysis(
            days_back=args.days,
            store_results=not args.no_store,
        )
        print(f"Result: {result}")
