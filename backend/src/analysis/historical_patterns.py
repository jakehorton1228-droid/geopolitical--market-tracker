"""
Historical frequency analysis: "When X happens, Y goes UP Z% of the time."

This is Level 1 prediction -- pure conditional probability from historical data.
No model training, just counting occurrences and computing statistics.
"""

from dataclasses import dataclass
from datetime import date
from typing import Optional
import logging

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import select

from src.analysis.feature_engineering import FeatureEngineering
from src.config.constants import (
    EVENT_GROUPS,
    SYMBOL_COUNTRY_MAP,
    CAMEO_CATEGORIES,
)
from src.db.connection import get_session
from src.db.models import Event
from src.db.queries import get_events_by_date_range

logger = logging.getLogger(__name__)


@dataclass
class HistoricalPattern:
    """Result of historical frequency analysis."""
    symbol: str
    event_filter: str
    total_occurrences: int
    up_count: int
    down_count: int
    up_percentage: float
    avg_return_up: float
    avg_return_down: float
    avg_return_all: float
    median_return: float
    t_statistic: float
    p_value: float


class HistoricalPatternAnalyzer:
    """Analyzes historical conditional probabilities of market moves given events."""

    def __init__(self):
        self.fe = FeatureEngineering()

    def _fetch_event_date_counts(
        self,
        start_date: date,
        end_date: date,
        event_root_codes: Optional[list[str]] = None,
        country_code: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Get event counts per date using SQL aggregation.

        Much faster than loading millions of rows — the database does
        the filtering and grouping, returning only ~3600 rows (one per date).
        """
        from sqlalchemy import func, or_

        with get_session() as session:
            query = session.query(
                Event.event_date.label("date"),
                func.count(Event.id).label("event_count"),
            ).filter(
                Event.event_date >= start_date,
                Event.event_date <= end_date,
            )

            if event_root_codes:
                query = query.filter(Event.event_root_code.in_(event_root_codes))

            if country_code:
                query = query.filter(or_(
                    Event.actor1_country_code == country_code,
                    Event.actor2_country_code == country_code,
                    Event.action_geo_country_code == country_code,
                ))

            query = query.group_by(Event.event_date)
            rows = query.all()

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows, columns=["date", "event_count"])

    def _compute_pattern_from_dates(
        self,
        market_df: pd.DataFrame,
        event_dates_df: pd.DataFrame,
        symbol: str,
        event_filter: str,
    ) -> Optional[HistoricalPattern]:
        """Compute a single pattern by joining event dates with market returns."""
        if event_dates_df.empty:
            return None

        # Merge with market data — only keep dates where events occurred
        merged = pd.merge(market_df, event_dates_df[["date"]], on="date", how="inner")
        merged = merged.dropna(subset=["log_return"])

        if merged.empty or len(merged) < 5:
            return None

        returns = merged["log_return"].values
        up_returns = returns[returns > 0]
        down_returns = returns[returns <= 0]

        t_stat, p_value = stats.ttest_1samp(returns, 0)

        return HistoricalPattern(
            symbol=symbol,
            event_filter=event_filter,
            total_occurrences=len(returns),
            up_count=len(up_returns),
            down_count=len(down_returns),
            up_percentage=len(up_returns) / len(returns) * 100,
            avg_return_up=float(np.mean(up_returns) * 100) if len(up_returns) > 0 else 0.0,
            avg_return_down=float(np.mean(down_returns) * 100) if len(down_returns) > 0 else 0.0,
            avg_return_all=float(np.mean(returns) * 100),
            median_return=float(np.median(returns) * 100),
            t_statistic=float(t_stat),
            p_value=float(p_value),
        )

    def _get_event_day_returns(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        event_group: Optional[str] = None,
        country_code: Optional[str] = None,
        min_event_count: int = 1,
    ) -> pd.DataFrame:
        """
        Get market returns for days that match event criteria.

        Returns DataFrame with date, log_return, and event info for
        days that had qualifying events.
        """
        market_df = self.fe.fetch_market_data(symbol, start_date, end_date)
        if market_df.empty:
            return pd.DataFrame()

        # Get events with optional filters
        with get_session() as session:
            event_root_codes = None
            if event_group and event_group in EVENT_GROUPS:
                event_root_codes = EVENT_GROUPS[event_group]

            events = get_events_by_date_range(
                session,
                start_date,
                end_date,
                country_code=country_code,
                event_root_codes=event_root_codes,
            )

        if not events:
            return pd.DataFrame()

        # Aggregate events by date
        event_rows = []
        for e in events:
            event_rows.append({
                "date": e.event_date,
                "goldstein_scale": e.goldstein_scale or 0,
                "num_mentions": e.num_mentions or 0,
            })

        events_df = pd.DataFrame(event_rows)
        event_agg = events_df.groupby("date").agg(
            event_count=("goldstein_scale", "count"),
            avg_goldstein=("goldstein_scale", "mean"),
            total_mentions=("num_mentions", "sum"),
        ).reset_index()

        # Filter by minimum event count
        event_agg = event_agg[event_agg["event_count"] >= min_event_count]

        # Merge with market data
        merged = pd.merge(market_df, event_agg, on="date", how="inner")
        merged = merged.dropna(subset=["log_return"])

        return merged

    def analyze_event_type_pattern(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        event_group: Optional[str] = None,
        country_code: Optional[str] = None,
        min_event_count: int = 1,
    ) -> Optional[HistoricalPattern]:
        """
        For days when qualifying events occur, what is the distribution
        of returns for the given symbol?

        Returns HistoricalPattern with up/down counts, percentages, and t-test.
        """
        merged = self._get_event_day_returns(
            symbol, start_date, end_date,
            event_group=event_group,
            country_code=country_code,
            min_event_count=min_event_count,
        )

        if merged.empty or len(merged) < 5:
            return None

        returns = merged["log_return"].values
        up_returns = returns[returns > 0]
        down_returns = returns[returns <= 0]

        # t-test: is the mean return significantly different from 0?
        t_stat, p_value = stats.ttest_1samp(returns, 0)

        # Build filter description
        filter_parts = []
        if event_group:
            filter_parts.append(event_group.replace("_", " "))
        if country_code:
            filter_parts.append(f"in {country_code}")
        event_filter = " ".join(filter_parts) if filter_parts else "all events"

        return HistoricalPattern(
            symbol=symbol,
            event_filter=event_filter,
            total_occurrences=len(returns),
            up_count=len(up_returns),
            down_count=len(down_returns),
            up_percentage=len(up_returns) / len(returns) * 100,
            avg_return_up=float(np.mean(up_returns) * 100) if len(up_returns) > 0 else 0.0,
            avg_return_down=float(np.mean(down_returns) * 100) if len(down_returns) > 0 else 0.0,
            avg_return_all=float(np.mean(returns) * 100),
            median_return=float(np.median(returns) * 100),
            t_statistic=float(t_stat),
            p_value=float(p_value),
        )

    def all_patterns_for_symbol(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        min_occurrences: int = 10,
    ) -> list[HistoricalPattern]:
        """
        Compute patterns for all event groups for a symbol.

        Uses SQL-level aggregation (GROUP BY event_date) for each filter
        combination instead of loading millions of event rows into Python.
        Each query returns only ~3600 rows (one per date with events).
        """
        # Fetch market data once
        market_df = self.fe.fetch_market_data(symbol, start_date, end_date)
        if market_df.empty:
            return []

        patterns = []

        # Pattern for each event group (SQL aggregation per group)
        for group_name, codes in EVENT_GROUPS.items():
            event_dates = self._fetch_event_date_counts(
                start_date, end_date, event_root_codes=codes,
            )
            filter_label = group_name.replace("_", " ")
            pattern = self._compute_pattern_from_dates(
                market_df, event_dates, symbol, filter_label,
            )
            if pattern and pattern.total_occurrences >= min_occurrences:
                patterns.append(pattern)

        # Country-specific patterns
        relevant_countries = SYMBOL_COUNTRY_MAP.get(symbol, [])
        for country in relevant_countries:
            # All events for country
            event_dates = self._fetch_event_date_counts(
                start_date, end_date, country_code=country,
            )
            pattern = self._compute_pattern_from_dates(
                market_df, event_dates, symbol, f"in {country}",
            )
            if pattern and pattern.total_occurrences >= min_occurrences:
                patterns.append(pattern)

            # Country + conflict combo
            for group_name in ["violent_conflict", "material_conflict"]:
                codes = EVENT_GROUPS[group_name]
                event_dates = self._fetch_event_date_counts(
                    start_date, end_date,
                    event_root_codes=codes,
                    country_code=country,
                )
                filter_label = f"{group_name.replace('_', ' ')} in {country}"
                pattern = self._compute_pattern_from_dates(
                    market_df, event_dates, symbol, filter_label,
                )
                if pattern and pattern.total_occurrences >= min_occurrences:
                    patterns.append(pattern)

        # Sort by statistical significance
        patterns.sort(key=lambda p: p.p_value)

        return patterns
