"""
Anomaly Detection Module.

WHAT IS ANOMALY DETECTION?
--------------------------
While event studies ask "did this event move markets?", anomaly detection
asks the opposite: "the market moved unusually - what caused it?"

We detect two types of anomalies:

1. UNEXPLAINED MOVES: Market moved significantly, but no major event happened.
   This could indicate insider trading, unreported news, or other hidden factors.

2. MUTED RESPONSES: A major event happened, but the market didn't react as
   expected. This could mean the event was already priced in, or the market
   disagrees with its importance.

HOW IT WORKS:
-------------
1. Calculate daily Z-scores for market returns:
   Z = (return - mean) / std_dev

2. Flag days where |Z| > threshold (default: 2.0 = 2 standard deviations)

3. Cross-reference with events:
   - Big move + no event = "unexplained_move"
   - Big event + small move = "muted_response"

USAGE:
------
    from src.analysis.anomaly_detection import AnomalyDetector

    detector = AnomalyDetector()

    # Find unexplained market moves
    anomalies = detector.detect_unexplained_moves("CL=F", start_date, end_date)

    # Find muted responses to events
    muted = detector.detect_muted_responses(start_date, end_date)
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional
import logging

import numpy as np
import pandas as pd

from src.config.settings import (
    ANOMALY_ZSCORE_THRESHOLD,
    ANOMALY_GOLDSTEIN_THRESHOLD,
    ANOMALY_MOVE_THRESHOLD_PCT,
)
from src.config.constants import COUNTRY_ASSET_MAP, EVENT_ASSET_MAP, get_event_group
from src.db.connection import get_session
from src.db.models import Event, MarketData, AnalysisResult
from src.db.queries import (
    get_market_data,
    get_events_by_date_range,
    save_analysis_result,
)

logger = logging.getLogger(__name__)


@dataclass
class Anomaly:
    """Container for a detected anomaly."""

    date: date
    symbol: str
    anomaly_type: str  # "unexplained_move" or "muted_response"

    # Market data
    actual_return: float
    expected_return: float
    z_score: float

    # Event data (if applicable)
    event_id: Optional[int] = None
    event_goldstein: Optional[float] = None
    event_description: Optional[str] = None

    # Scoring
    anomaly_score: float = 0.0  # Higher = more anomalous


class AnomalyDetector:
    """
    Detects anomalous market behavior that may or may not be explained by events.
    """

    def __init__(
        self,
        zscore_threshold: float = None,
        goldstein_threshold: float = None,
        move_threshold_pct: float = None,
        lookback_days: int = 30,
    ):
        """
        Initialize the anomaly detector.

        Args:
            zscore_threshold: Z-score threshold for "unusual" moves (default 2.0)
            goldstein_threshold: Min absolute Goldstein score for "big" events
            move_threshold_pct: Expected minimum market reaction to big events
            lookback_days: Days to use for calculating mean/std
        """
        self.zscore_threshold = zscore_threshold or ANOMALY_ZSCORE_THRESHOLD
        self.goldstein_threshold = goldstein_threshold or ANOMALY_GOLDSTEIN_THRESHOLD
        self.move_threshold_pct = move_threshold_pct or ANOMALY_MOVE_THRESHOLD_PCT
        self.lookback_days = lookback_days

    def _calculate_zscore(self, returns: pd.Series) -> pd.Series:
        """
        Calculate rolling Z-scores for returns.

        Z-SCORE EXPLAINED:
        ------------------
        Z = (x - mean) / std_dev

        A Z-score tells you how many standard deviations a value is from the mean.
        - Z = 0: exactly average
        - Z = 1: one std dev above average
        - Z = 2: two std devs above (unusual, ~2.5% probability)
        - Z = 3: three std devs above (rare, ~0.1% probability)

        Args:
            returns: Series of returns

        Returns:
            Series of Z-scores
        """
        # Rolling mean and std
        rolling_mean = returns.rolling(window=self.lookback_days, min_periods=10).mean()
        rolling_std = returns.rolling(window=self.lookback_days, min_periods=10).std()

        # Z-score
        zscore = (returns - rolling_mean) / rolling_std

        return zscore

    def _get_price_data_with_zscore(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Fetch price data and calculate Z-scores.

        Args:
            symbol: Ticker symbol
            start_date: Start date (will fetch extra for lookback)
            end_date: End date

        Returns:
            DataFrame with date, return, z_score columns
        """
        # Fetch extra days for lookback calculation
        fetch_start = start_date - timedelta(days=self.lookback_days + 10)

        with get_session() as session:
            data = get_market_data(session, symbol, fetch_start, end_date)

            if not data:
                return pd.DataFrame()

            df = pd.DataFrame([
                {
                    "date": d.date,
                    "close": float(d.close),
                    "log_return": d.log_return,
                }
                for d in data
            ])

        if df.empty:
            return df

        df = df.sort_values("date").reset_index(drop=True)

        # Calculate Z-scores
        df["z_score"] = self._calculate_zscore(df["log_return"])

        # Calculate rolling mean for "expected return"
        df["expected_return"] = df["log_return"].rolling(
            window=self.lookback_days, min_periods=10
        ).mean()

        # Filter to requested date range
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

        return df

    def detect_unexplained_moves(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> list[Anomaly]:
        """
        Find days with large market moves but no major events.

        An "unexplained move" is when:
        1. |Z-score| > threshold (market moved a lot)
        2. No significant events happened that day

        Args:
            symbol: Ticker symbol
            start_date: Start of analysis period
            end_date: End of analysis period

        Returns:
            List of Anomaly objects for unexplained moves
        """
        logger.info(f"Detecting unexplained moves for {symbol}")

        # Get price data with Z-scores
        price_df = self._get_price_data_with_zscore(symbol, start_date, end_date)

        if price_df.empty:
            logger.warning(f"No price data for {symbol}")
            return []

        # Find days with unusual moves
        unusual_days = price_df[abs(price_df["z_score"]) > self.zscore_threshold].copy()

        if unusual_days.empty:
            logger.info(f"No unusual moves detected for {symbol}")
            return []

        # Get events for the period
        with get_session() as session:
            events = get_events_by_date_range(
                session,
                start_date,
                end_date,
                min_goldstein=self.goldstein_threshold,
            )

        # Create a set of dates with significant events
        event_dates = {e.event_date for e in events}

        # Find unexplained moves (unusual move but no event)
        anomalies = []
        for _, row in unusual_days.iterrows():
            if row["date"] not in event_dates:
                anomaly = Anomaly(
                    date=row["date"],
                    symbol=symbol,
                    anomaly_type="unexplained_move",
                    actual_return=row["log_return"],
                    expected_return=row["expected_return"] if pd.notna(row["expected_return"]) else 0,
                    z_score=row["z_score"],
                    anomaly_score=abs(row["z_score"]),  # Higher Z = more anomalous
                )
                anomalies.append(anomaly)

        logger.info(f"Found {len(anomalies)} unexplained moves for {symbol}")
        return anomalies

    def detect_muted_responses(
        self,
        start_date: date,
        end_date: date,
        symbol: str = None,
    ) -> list[Anomaly]:
        """
        Find events that should have moved markets but didn't.

        A "muted response" is when:
        1. A significant event happened (|Goldstein| > threshold)
        2. Related market didn't move much (|return| < expected)

        Args:
            start_date: Start of analysis period
            end_date: End of analysis period
            symbol: Optional specific symbol to check (otherwise uses mappings)

        Returns:
            List of Anomaly objects for muted responses
        """
        logger.info("Detecting muted responses to events")

        # Get significant events
        with get_session() as session:
            events = get_events_by_date_range(
                session,
                start_date,
                end_date,
                min_goldstein=self.goldstein_threshold,
                min_mentions=10,  # Well-covered events
            )

        if not events:
            logger.info("No significant events found")
            return []

        anomalies = []

        for event in events:
            # Determine which symbols should be affected
            if symbol:
                related_symbols = [symbol]
            else:
                related_symbols = self._get_related_symbols(event)

            if not related_symbols:
                continue

            for sym in related_symbols:
                # Get market data for event day
                with get_session() as session:
                    market_data = get_market_data(
                        session,
                        sym,
                        event.event_date,
                        event.event_date,
                    )

                if not market_data:
                    continue

                daily_return = market_data[0].log_return
                if daily_return is None:
                    continue

                # Check if response was muted
                # Expected move based on Goldstein score
                expected_move = abs(event.goldstein_scale) / 10 * (self.move_threshold_pct / 100)
                actual_move = abs(daily_return)

                # If actual move is less than half the expected, it's muted
                if actual_move < expected_move * 0.5:
                    anomaly = Anomaly(
                        date=event.event_date,
                        symbol=sym,
                        anomaly_type="muted_response",
                        actual_return=daily_return,
                        expected_return=expected_move if event.goldstein_scale < 0 else -expected_move,
                        z_score=0,  # Not applicable for muted response
                        event_id=event.id,
                        event_goldstein=event.goldstein_scale,
                        event_description=f"{event.actor1_code} → {event.actor2_code}",
                        anomaly_score=expected_move / max(actual_move, 0.0001),  # Ratio of expected/actual
                    )
                    anomalies.append(anomaly)

        logger.info(f"Found {len(anomalies)} muted responses")
        return anomalies

    def _get_related_symbols(self, event: Event) -> list[str]:
        """
        Get symbols that should be affected by an event.

        Uses the mappings from constants.py to determine which assets
        are sensitive to events in certain countries or of certain types.

        Args:
            event: Event object

        Returns:
            List of related ticker symbols
        """
        symbols = set()

        # Get symbols based on country
        for country in [event.actor1_country_code, event.actor2_country_code, event.action_geo_country_code]:
            if country and country in COUNTRY_ASSET_MAP:
                symbols.update(COUNTRY_ASSET_MAP[country])

        # Get symbols based on event type
        event_group = get_event_group(event.event_root_code)
        if event_group in EVENT_ASSET_MAP:
            symbols.update(EVENT_ASSET_MAP[event_group])

        return list(symbols)

    def save_anomaly(self, anomaly: Anomaly) -> None:
        """
        Save an anomaly to the database.

        Args:
            anomaly: Anomaly to save
        """
        with get_session() as session:
            save_analysis_result(session, {
                "event_id": anomaly.event_id or 0,  # Use 0 for unexplained moves
                "symbol": anomaly.symbol,
                "analysis_type": "anomaly_detection",
                "car": anomaly.actual_return,
                "expected_return": anomaly.expected_return,
                "actual_return": anomaly.actual_return,
                "abnormal_return": anomaly.actual_return - anomaly.expected_return,
                "is_anomaly": True,
                "anomaly_type": anomaly.anomaly_type,
                "anomaly_score": anomaly.anomaly_score,
                "is_significant": True,  # All detected anomalies are significant by definition
            })

    def run_full_detection(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
        save: bool = True,
    ) -> dict[str, list[Anomaly]]:
        """
        Run full anomaly detection for multiple symbols.

        Args:
            symbols: List of symbols to analyze
            start_date: Start date
            end_date: End date
            save: Whether to save results to database

        Returns:
            Dictionary mapping anomaly type to list of anomalies
        """
        all_unexplained = []
        all_muted = []

        # Detect unexplained moves for each symbol
        for symbol in symbols:
            try:
                unexplained = self.detect_unexplained_moves(symbol, start_date, end_date)
                all_unexplained.extend(unexplained)
                if save:
                    for a in unexplained:
                        self.save_anomaly(a)
            except Exception as e:
                logger.error(f"Error detecting unexplained moves for {symbol}: {e}")

        # Detect muted responses (checks all symbols via mappings)
        try:
            muted = self.detect_muted_responses(start_date, end_date)
            all_muted.extend(muted)
            if save:
                for a in muted:
                    self.save_anomaly(a)
        except Exception as e:
            logger.error(f"Error detecting muted responses: {e}")

        return {
            "unexplained_moves": all_unexplained,
            "muted_responses": all_muted,
        }


def explain_anomaly(anomaly: Anomaly) -> str:
    """
    Generate a human-readable explanation of an anomaly.

    Args:
        anomaly: Anomaly to explain

    Returns:
        Explanation string
    """
    if anomaly.anomaly_type == "unexplained_move":
        direction = "up" if anomaly.actual_return > 0 else "down"
        return f"""
UNEXPLAINED MOVE DETECTED
=========================
Date: {anomaly.date}
Symbol: {anomaly.symbol}

{anomaly.symbol} moved {direction} by {abs(anomaly.actual_return)*100:.2f}%
This is {abs(anomaly.z_score):.1f} standard deviations from normal.

No significant geopolitical event was found to explain this move.
Possible causes: unreported news, insider activity, technical factors.

Anomaly Score: {anomaly.anomaly_score:.2f}
"""
    else:  # muted_response
        return f"""
MUTED RESPONSE DETECTED
=======================
Date: {anomaly.date}
Symbol: {anomaly.symbol}

A significant event occurred: {anomaly.event_description}
Goldstein Score: {anomaly.event_goldstein} ({"conflict" if anomaly.event_goldstein < 0 else "cooperation"})

Expected market reaction: ~{abs(anomaly.expected_return)*100:.2f}%
Actual market reaction: {anomaly.actual_return*100:.2f}%

The market did not react as strongly as expected.
Possible causes: event was priced in, market disagrees with importance.

Anomaly Score: {anomaly.anomaly_score:.2f}
"""
