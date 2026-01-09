#!/usr/bin/env python3
"""
CLI for running analyses from the command line.

This script provides a simple interface to run the various analysis modules
without writing Python code. Great for quick exploration and automation.

USAGE:
------
    # Run event study
    python scripts/run_analysis.py event-study --symbol CL=F --date 2024-01-15

    # Run regression analysis
    python scripts/run_analysis.py regression --symbol SPY --days 90

    # Run anomaly detection
    python scripts/run_analysis.py anomalies --symbol GC=F --days 60

    # Train classifier
    python scripts/run_analysis.py classify --symbol CL=F --days 180

    # Compare all markets
    python scripts/run_analysis.py compare --days 90

EXAMPLES:
---------
    # Quick event study for oil after a specific date
    python scripts/run_analysis.py event-study -s CL=F -d 2024-02-24

    # Regression with verbose output
    python scripts/run_analysis.py regression -s SPY --days 180 -v

    # Find anomalies in gold
    python scripts/run_analysis.py anomalies -s GC=F --days 90
"""

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.analysis.logging_config import setup_logging


def run_event_study(args):
    """Run event study analysis."""
    from src.analysis.production_event_study import ProductionEventStudy

    study = ProductionEventStudy()

    # Parse date
    if args.date:
        event_date = date.fromisoformat(args.date)
    else:
        event_date = date.today() - timedelta(days=7)

    print(f"\nRunning event study for {args.symbol} on {event_date}...")
    result = study.analyze_event(
        event_id=0,
        symbol=args.symbol,
        event_date=event_date,
    )

    if result:
        print(result.summary)
    else:
        print("No result - insufficient data")


def run_regression(args):
    """Run regression analysis."""
    from src.analysis.production_regression import ProductionRegression, print_interpretation

    reg = ProductionRegression()

    end_date = date.today()
    start_date = end_date - timedelta(days=args.days)

    print(f"\nRunning regression for {args.symbol} ({args.days} days)...")
    result = reg.analyze(args.symbol, start_date, end_date)

    if result:
        if args.verbose:
            print(result.summary)
        else:
            print_interpretation(result)
    else:
        print("No result - insufficient data")


def run_anomaly_detection(args):
    """Run anomaly detection."""
    from src.analysis.production_anomaly import ProductionAnomalyDetector

    detector = ProductionAnomalyDetector()

    end_date = date.today()
    start_date = end_date - timedelta(days=args.days)

    print(f"\nRunning anomaly detection for {args.symbol} ({args.days} days)...")
    anomalies = detector.detect_all(args.symbol, start_date, end_date)
    report = detector.get_anomaly_report(anomalies, args.symbol, start_date, end_date)

    print(report.summary)


def run_classification(args):
    """Train and evaluate classifier."""
    from src.analysis.production_classifier import ProductionClassifier

    classifier = ProductionClassifier()

    end_date = date.today()
    start_date = end_date - timedelta(days=args.days)

    print(f"\nTraining classifier for {args.symbol} ({args.days} days)...")
    metrics = classifier.train(args.symbol, start_date, end_date)

    if metrics:
        print(f"\nResults for {args.symbol}:")
        print(f"  Accuracy: {metrics.accuracy:.2%}")
        print(f"  Precision: {metrics.precision:.2%}")
        print(f"  Recall: {metrics.recall:.2%}")
        print(f"  F1 Score: {metrics.f1_score:.2%}")
        print(f"  Cross-validation: {metrics.cv_accuracy:.2%} (+/- {metrics.cv_std*2:.2%})")
        print(f"  Samples: {metrics.n_samples}")

        if args.verbose:
            print("\nFeature Importance:")
            for name, imp in classifier.get_feature_importance(args.symbol).items():
                print(f"  {name}: {imp:.4f}")
    else:
        print("Training failed - insufficient data")


def run_comparison(args):
    """Compare analysis across all markets."""
    from src.analysis.production_anomaly import ProductionAnomalyDetector
    from src.config.constants import get_all_symbols

    detector = ProductionAnomalyDetector()

    end_date = date.today()
    start_date = end_date - timedelta(days=args.days)

    symbols = get_all_symbols()
    if args.limit:
        symbols = symbols[:args.limit]

    print(f"\nComparing {len(symbols)} markets ({args.days} days)...")
    comparison = detector.compare_symbols(symbols, start_date, end_date)

    if not comparison.empty:
        print("\nAnomaly rates by market:")
        print("=" * 60)
        for _, row in comparison.head(20).iterrows():
            print(
                f"{row['symbol']:<12} "
                f"rate={row['anomaly_rate']*100:.1f}%  "
                f"unexplained={row['unexplained_moves']}  "
                f"muted={row['muted_responses']}  "
                f"outliers={row['statistical_outliers']}"
            )
    else:
        print("No comparison data available")


def main():
    parser = argparse.ArgumentParser(
        description="Run analysis modules from the command line",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Global options
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    subparsers = parser.add_subparsers(dest="command", help="Analysis type")

    # Event study command
    event_parser = subparsers.add_parser(
        "event-study",
        help="Run event study analysis",
    )
    event_parser.add_argument(
        "-s", "--symbol",
        required=True,
        help="Ticker symbol (e.g., CL=F, SPY, GC=F)",
    )
    event_parser.add_argument(
        "-d", "--date",
        help="Event date (YYYY-MM-DD). Default: 7 days ago",
    )

    # Regression command
    reg_parser = subparsers.add_parser(
        "regression",
        help="Run regression analysis",
    )
    reg_parser.add_argument(
        "-s", "--symbol",
        required=True,
        help="Ticker symbol",
    )
    reg_parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Number of days to analyze (default: 90)",
    )

    # Anomaly detection command
    anomaly_parser = subparsers.add_parser(
        "anomalies",
        help="Run anomaly detection",
    )
    anomaly_parser.add_argument(
        "-s", "--symbol",
        required=True,
        help="Ticker symbol",
    )
    anomaly_parser.add_argument(
        "--days",
        type=int,
        default=60,
        help="Number of days to analyze (default: 60)",
    )

    # Classification command
    classify_parser = subparsers.add_parser(
        "classify",
        help="Train and evaluate classifier",
    )
    classify_parser.add_argument(
        "-s", "--symbol",
        required=True,
        help="Ticker symbol",
    )
    classify_parser.add_argument(
        "--days",
        type=int,
        default=180,
        help="Number of days for training (default: 180)",
    )

    # Comparison command
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare anomaly rates across markets",
    )
    compare_parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Number of days to analyze (default: 90)",
    )
    compare_parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of symbols to compare",
    )

    args = parser.parse_args()

    # Setup logging based on verbosity
    if args.verbose:
        setup_logging(level="DEBUG")
    else:
        setup_logging(level="INFO")

    # Route to appropriate function
    if args.command == "event-study":
        run_event_study(args)
    elif args.command == "regression":
        run_regression(args)
    elif args.command == "anomalies":
        run_anomaly_detection(args)
    elif args.command == "classify":
        run_classification(args)
    elif args.command == "compare":
        run_comparison(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
