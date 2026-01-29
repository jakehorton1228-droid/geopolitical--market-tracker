"""
Data Quality Monitoring Module.

Checks for data integrity issues that could degrade model performance
or produce misleading analysis results. Defense systems require high
data reliability - this module ensures data meets quality standards.

CHECKS PERFORMED:
-----------------
1. Missing Data: Gaps in market data or GDELT ingestion
2. Staleness: How recently data was last updated
3. Range Validation: Values within expected bounds
4. Consistency: Cross-checks between related data sources
5. Completeness: Coverage across tracked symbols

USAGE:
------
    from src.analysis.data_quality import DataQualityChecker

    checker = DataQualityChecker()
    report = checker.check_all("SPY", start_date, end_date)
    print(f"Quality score: {report.quality_score:.0%}")
"""

from dataclasses import dataclass, field
from datetime import date, timedelta, datetime
from typing import Optional
import logging

import numpy as np
import pandas as pd

from src.config.constants import TRACKED_SYMBOLS
from src.db.connection import get_session
from src.db.queries import (
    get_market_data,
    get_events_by_date_range,
    get_latest_market_date,
)

logger = logging.getLogger(__name__)


@dataclass
class QualityIssue:
    """A single data quality issue."""
    issue_type: str  # "missing_data", "stale_data", "out_of_range", "outlier", "gap"
    severity: str  # "low", "medium", "high"
    description: str
    metric_name: str
    metric_value: float
    threshold: float
    affected_date: Optional[date] = None
    affected_symbol: Optional[str] = None


@dataclass
class SymbolQualityReport:
    """Quality report for a single symbol."""
    symbol: str
    start_date: date
    end_date: date

    # Scores (0-1, higher is better)
    quality_score: float
    completeness_score: float
    freshness_score: float
    validity_score: float

    # Counts
    total_records: int
    expected_records: int
    missing_days: int
    outlier_count: int

    # Staleness
    last_update: Optional[date]
    days_stale: int

    # Issues found
    issues: list[QualityIssue] = field(default_factory=list)


@dataclass
class DataQualityReport:
    """Overall data quality report across all symbols."""
    check_date: date
    start_date: date
    end_date: date

    # Overall scores
    overall_score: float
    symbols_checked: int
    symbols_with_issues: int

    # Per-symbol reports
    symbol_reports: list[SymbolQualityReport]

    # Aggregate issues
    total_issues: int
    high_severity_issues: int
    medium_severity_issues: int
    low_severity_issues: int

    # Summary
    summary: str = ""


class DataQualityChecker:
    """
    Checks data quality across market data and event data.

    Produces quality scores and identifies specific issues that
    could affect model performance or analysis accuracy.
    """

    # Validation bounds
    GOLDSTEIN_MIN = -10.0
    GOLDSTEIN_MAX = 10.0
    MAX_DAILY_RETURN = 0.25  # 25% daily move is suspicious
    MAX_STALE_DAYS = 3  # Data older than this is considered stale (weekends excluded)

    def __init__(self):
        pass

    def _count_business_days(self, start_date: date, end_date: date) -> int:
        """Count business days between two dates (approximate)."""
        total_days = (end_date - start_date).days
        weeks = total_days // 7
        remainder = total_days % 7
        business_days = weeks * 5 + min(remainder, 5)
        return max(business_days, 1)

    def check_completeness(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> tuple[float, list[QualityIssue]]:
        """
        Check for missing data points.

        Returns (completeness_score, issues).
        """
        issues = []

        with get_session() as session:
            market_data = get_market_data(session, symbol, start_date, end_date)

        if not market_data:
            issues.append(QualityIssue(
                issue_type="missing_data",
                severity="high",
                description=f"No market data found for {symbol}",
                metric_name="record_count",
                metric_value=0,
                threshold=1,
                affected_symbol=symbol,
            ))
            return 0.0, issues

        # Check against expected business days
        expected = self._count_business_days(start_date, end_date)
        actual = len(market_data)
        completeness = min(actual / max(expected, 1), 1.0)

        if completeness < 0.8:
            severity = "high" if completeness < 0.5 else "medium"
            issues.append(QualityIssue(
                issue_type="missing_data",
                severity=severity,
                description=(
                    f"{symbol}: {actual}/{expected} expected trading days "
                    f"({completeness:.0%} complete)"
                ),
                metric_name="completeness",
                metric_value=completeness,
                threshold=0.8,
                affected_symbol=symbol,
            ))

        # Check for gaps (consecutive missing days > 5)
        dates = sorted([m.date for m in market_data])
        for i in range(1, len(dates)):
            gap = (dates[i] - dates[i - 1]).days
            if gap > 5:  # More than a week gap (accounting for weekends)
                issues.append(QualityIssue(
                    issue_type="gap",
                    severity="medium",
                    description=f"{symbol}: {gap}-day gap between {dates[i-1]} and {dates[i]}",
                    metric_name="gap_days",
                    metric_value=gap,
                    threshold=5,
                    affected_date=dates[i - 1],
                    affected_symbol=symbol,
                ))

        return completeness, issues

    def check_freshness(
        self,
        symbol: str,
    ) -> tuple[float, int, Optional[date], list[QualityIssue]]:
        """
        Check how recent the data is.

        Returns (freshness_score, days_stale, last_update, issues).
        """
        issues = []

        with get_session() as session:
            last_date = get_latest_market_date(session, symbol)

        if last_date is None:
            issues.append(QualityIssue(
                issue_type="stale_data",
                severity="high",
                description=f"No data found for {symbol}",
                metric_name="days_stale",
                metric_value=999,
                threshold=self.MAX_STALE_DAYS,
                affected_symbol=symbol,
            ))
            return 0.0, 999, None, issues

        today = date.today()
        days_stale = (today - last_date).days

        # Account for weekends: if today is Monday, 3 days stale is OK
        weekday = today.weekday()
        expected_stale = 1 if weekday < 5 else (weekday - 4)

        adjusted_stale = max(0, days_stale - expected_stale)
        freshness = max(0.0, 1.0 - adjusted_stale / 10.0)

        if adjusted_stale > self.MAX_STALE_DAYS:
            severity = "high" if adjusted_stale > 7 else "medium"
            issues.append(QualityIssue(
                issue_type="stale_data",
                severity=severity,
                description=f"{symbol}: Last data is {days_stale} days old (as of {last_date})",
                metric_name="days_stale",
                metric_value=days_stale,
                threshold=self.MAX_STALE_DAYS,
                affected_date=last_date,
                affected_symbol=symbol,
            ))

        return freshness, days_stale, last_date, issues

    def check_validity(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> tuple[float, int, list[QualityIssue]]:
        """
        Validate that data values are within expected ranges.

        Returns (validity_score, outlier_count, issues).
        """
        issues = []
        outlier_count = 0

        with get_session() as session:
            market_data = get_market_data(session, symbol, start_date, end_date)

        if not market_data:
            return 1.0, 0, issues

        total_checks = 0
        passed_checks = 0

        for m in market_data:
            total_checks += 1

            # Check for null close prices
            if m.close is None:
                issues.append(QualityIssue(
                    issue_type="out_of_range",
                    severity="high",
                    description=f"{symbol}: Null close price on {m.date}",
                    metric_name="close_price",
                    metric_value=0,
                    threshold=0,
                    affected_date=m.date,
                    affected_symbol=symbol,
                ))
                continue

            # Check for negative close prices
            if float(m.close) <= 0:
                issues.append(QualityIssue(
                    issue_type="out_of_range",
                    severity="high",
                    description=f"{symbol}: Non-positive close price ({m.close}) on {m.date}",
                    metric_name="close_price",
                    metric_value=float(m.close),
                    threshold=0,
                    affected_date=m.date,
                    affected_symbol=symbol,
                ))
                continue

            # Check for extreme daily returns
            if m.daily_return is not None and abs(m.daily_return) > self.MAX_DAILY_RETURN:
                outlier_count += 1
                issues.append(QualityIssue(
                    issue_type="outlier",
                    severity="medium",
                    description=(
                        f"{symbol}: Extreme daily return ({m.daily_return*100:.1f}%) on {m.date}"
                    ),
                    metric_name="daily_return",
                    metric_value=abs(m.daily_return),
                    threshold=self.MAX_DAILY_RETURN,
                    affected_date=m.date,
                    affected_symbol=symbol,
                ))
            else:
                passed_checks += 1

        validity = passed_checks / max(total_checks, 1)
        return validity, outlier_count, issues

    def check_event_quality(
        self,
        start_date: date,
        end_date: date,
    ) -> tuple[float, list[QualityIssue]]:
        """
        Check quality of event data (GDELT).

        Returns (quality_score, issues).
        """
        issues = []

        with get_session() as session:
            events = get_events_by_date_range(session, start_date, end_date)

        if not events:
            issues.append(QualityIssue(
                issue_type="missing_data",
                severity="high",
                description="No events found in date range",
                metric_name="event_count",
                metric_value=0,
                threshold=1,
            ))
            return 0.0, issues

        total = len(events)
        valid = 0
        invalid_goldstein = 0

        for e in events:
            is_valid = True

            # Check Goldstein range
            if e.goldstein_scale is not None:
                if e.goldstein_scale < self.GOLDSTEIN_MIN or e.goldstein_scale > self.GOLDSTEIN_MAX:
                    invalid_goldstein += 1
                    is_valid = False

            if is_valid:
                valid += 1

        if invalid_goldstein > 0:
            issues.append(QualityIssue(
                issue_type="out_of_range",
                severity="medium",
                description=f"{invalid_goldstein} events with Goldstein scale outside [-10, 10]",
                metric_name="invalid_goldstein_count",
                metric_value=invalid_goldstein,
                threshold=0,
            ))

        # Check for event gaps (days with no events)
        event_dates = set(e.event_date for e in events)
        total_days = (end_date - start_date).days
        gap_days = 0
        for i in range(total_days):
            check_date = start_date + timedelta(days=i)
            if check_date.weekday() < 5 and check_date not in event_dates:
                gap_days += 1

        if gap_days > 5:
            issues.append(QualityIssue(
                issue_type="gap",
                severity="medium" if gap_days < 10 else "high",
                description=f"{gap_days} business days with no events ingested",
                metric_name="event_gap_days",
                metric_value=gap_days,
                threshold=5,
            ))

        quality = valid / max(total, 1)
        return quality, issues

    def check_symbol(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> SymbolQualityReport:
        """Run all quality checks for a single symbol."""
        completeness, comp_issues = self.check_completeness(symbol, start_date, end_date)
        freshness, days_stale, last_update, fresh_issues = self.check_freshness(symbol)
        validity, outlier_count, val_issues = self.check_validity(symbol, start_date, end_date)

        all_issues = comp_issues + fresh_issues + val_issues

        # Overall score: weighted average
        quality_score = (
            completeness * 0.4 +
            freshness * 0.3 +
            validity * 0.3
        )

        # Count records
        with get_session() as session:
            market_data = get_market_data(session, symbol, start_date, end_date)
        total_records = len(market_data) if market_data else 0
        expected_records = self._count_business_days(start_date, end_date)
        missing_days = max(0, expected_records - total_records)

        return SymbolQualityReport(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            quality_score=quality_score,
            completeness_score=completeness,
            freshness_score=freshness,
            validity_score=validity,
            total_records=total_records,
            expected_records=expected_records,
            missing_days=missing_days,
            outlier_count=outlier_count,
            last_update=last_update,
            days_stale=days_stale,
            issues=all_issues,
        )

    def check_all(
        self,
        start_date: date,
        end_date: date,
        symbols: Optional[list[str]] = None,
    ) -> DataQualityReport:
        """
        Run quality checks across all tracked symbols.

        This is the main entry point for comprehensive data quality assessment.
        """
        if symbols is None:
            symbols = TRACKED_SYMBOLS

        logger.info(f"Running data quality checks for {len(symbols)} symbols")

        symbol_reports = []
        for symbol in symbols:
            try:
                report = self.check_symbol(symbol, start_date, end_date)
                symbol_reports.append(report)
            except Exception as e:
                logger.error(f"Quality check failed for {symbol}: {e}")

        # Also check event data quality
        event_score, event_issues = self.check_event_quality(start_date, end_date)

        # Aggregate
        all_issues = []
        for r in symbol_reports:
            all_issues.extend(r.issues)
        all_issues.extend(event_issues)

        high = sum(1 for i in all_issues if i.severity == "high")
        medium = sum(1 for i in all_issues if i.severity == "medium")
        low = sum(1 for i in all_issues if i.severity == "low")

        symbols_with_issues = sum(1 for r in symbol_reports if r.issues)

        # Overall score
        if symbol_reports:
            overall_score = (
                sum(r.quality_score for r in symbol_reports) / len(symbol_reports)
                * 0.8  # 80% weight on market data
                + event_score * 0.2  # 20% weight on event data
            )
        else:
            overall_score = 0.0

        # Build summary
        summary_lines = [
            "",
            "=" * 60,
            "DATA QUALITY REPORT",
            "=" * 60,
            "",
            f"Check date: {date.today()}",
            f"Analysis period: {start_date} to {end_date}",
            f"Symbols checked: {len(symbol_reports)}",
            "",
            "OVERALL SCORES:",
            f"  Overall quality: {overall_score:.0%}",
            f"  Event data quality: {event_score:.0%}",
            "",
            "ISSUES FOUND:",
            f"  Total: {len(all_issues)}",
            f"  High severity: {high}",
            f"  Medium severity: {medium}",
            f"  Low severity: {low}",
            "",
        ]

        if symbol_reports:
            summary_lines.extend([
                "PER-SYMBOL QUALITY:",
                "-" * 40,
            ])
            for r in sorted(symbol_reports, key=lambda x: x.quality_score):
                status = "OK" if r.quality_score >= 0.8 else "WARN" if r.quality_score >= 0.5 else "FAIL"
                summary_lines.append(
                    f"  [{status}] {r.symbol}: {r.quality_score:.0%} "
                    f"(complete={r.completeness_score:.0%}, "
                    f"fresh={r.freshness_score:.0%}, "
                    f"valid={r.validity_score:.0%})"
                )

        if high > 0:
            summary_lines.extend([
                "",
                "HIGH SEVERITY ISSUES:",
                "-" * 40,
            ])
            for i in all_issues:
                if i.severity == "high":
                    summary_lines.append(f"  [{i.issue_type}] {i.description}")

        summary_lines.append("=" * 60)

        return DataQualityReport(
            check_date=date.today(),
            start_date=start_date,
            end_date=end_date,
            overall_score=overall_score,
            symbols_checked=len(symbol_reports),
            symbols_with_issues=symbols_with_issues,
            symbol_reports=symbol_reports,
            total_issues=len(all_issues),
            high_severity_issues=high,
            medium_severity_issues=medium,
            low_severity_issues=low,
            summary="\n".join(summary_lines),
        )
