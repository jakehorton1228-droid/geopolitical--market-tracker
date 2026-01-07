"""
Event Study Analysis Module.

WHAT IS AN EVENT STUDY?
-----------------------
An event study measures if a specific event caused an abnormal market reaction.
It's a standard technique in finance research to answer questions like:
- "Did the stock price drop when the CEO resigned?"
- "Did oil prices spike when war broke out?"

THE KEY INSIGHT:
---------------
Markets always move. The question isn't "did it move?" but "did it move MORE
than we'd expect?" We compare actual returns to expected (normal) returns.

METHODOLOGY:
-----------
1. ESTIMATION WINDOW: A period before the event (e.g., 30 days) where we
   calculate "normal" behavior - average return and volatility.

2. EVENT WINDOW: A short period around the event (e.g., -1 to +5 days)
   where we measure the actual returns.

3. ABNORMAL RETURN: actual_return - expected_return
   If positive, the asset did better than expected.
   If negative, worse than expected.

4. CAR (Cumulative Abnormal Return): Sum of abnormal returns over the
   event window. This captures the total impact.

5. STATISTICAL SIGNIFICANCE: Is the CAR large enough that it's unlikely
   to be random? We use t-statistics and p-values to test this.

TIMELINE EXAMPLE:
----------------
    |-------- Estimation Window --------|-- Event Window --|
    Day -35                          Day -6  -1   0   +5
                                              │
                                         EVENT HAPPENS

USAGE:
------
    from src.analysis.event_study import EventStudy

    study = EventStudy()

    # Analyze one event
    result = study.analyze_event(
        event_id=12345,
        symbol="CL=F",
        event_date=date(2024, 2, 24)
    )

    print(f"CAR: {result['car']:.2%}")
    print(f"Significant: {result['is_significant']}")
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional
import logging

import numpy as np
import pandas as pd
from scipy import stats

from src.config.settings import (
    EVENT_STUDY_ESTIMATION_WINDOW,
    EVENT_STUDY_EVENT_WINDOW_BEFORE,
    EVENT_STUDY_EVENT_WINDOW_AFTER,
    SIGNIFICANCE_LEVEL,
)
from src.db.connection import get_session
from src.db.models import Event, MarketData, AnalysisResult
from src.db.queries import get_market_data, save_analysis_result

logger = logging.getLogger(__name__)


@dataclass
class EventStudyResult:
    """
    Container for event study results.

    Using @dataclass automatically generates __init__, __repr__, etc.
    It's a clean way to bundle related data together.
    """

    event_id: int
    symbol: str
    event_date: date

    # Returns
    car: float  # Cumulative Abnormal Return
    abnormal_returns: list[float]  # Daily abnormal returns in event window

    # Statistics
    t_statistic: float
    p_value: float
    is_significant: bool

    # Context
    expected_return: float  # Average daily return in estimation window
    std_dev: float  # Standard deviation in estimation window
    actual_return: float  # Total return in event window

    # Window info
    estimation_start: date
    estimation_end: date
    event_window_start: date
    event_window_end: date

    # Data quality
    estimation_days: int  # Actual trading days in estimation window
    event_days: int  # Actual trading days in event window


class EventStudy:
    """
    Performs event study analysis to measure market reactions to events.

    This class implements the standard event study methodology used in
    academic finance research.
    """

    def __init__(
        self,
        estimation_window: int = None,
        event_window_before: int = None,
        event_window_after: int = None,
        significance_level: float = None,
    ):
        """
        Initialize the event study analyzer.

        Args:
            estimation_window: Days before event to calculate normal returns
            event_window_before: Days before event to include in analysis
            event_window_after: Days after event to include in analysis
            significance_level: P-value threshold (default 0.05 = 95% confidence)
        """
        self.estimation_window = estimation_window or EVENT_STUDY_ESTIMATION_WINDOW
        self.event_window_before = event_window_before or EVENT_STUDY_EVENT_WINDOW_BEFORE
        self.event_window_after = event_window_after or EVENT_STUDY_EVENT_WINDOW_AFTER
        self.significance_level = significance_level or SIGNIFICANCE_LEVEL

    def _get_price_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Fetch price data from database and convert to DataFrame.

        Args:
            symbol: Ticker symbol
            start_date: Start of date range
            end_date: End of date range

        Returns:
            DataFrame with date, close, and log_return columns
        """
        with get_session() as session:
            data = get_market_data(session, symbol, start_date, end_date)

            if not data:
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame([
                {
                    "date": d.date,
                    "close": float(d.close),
                    "log_return": d.log_return,
                }
                for d in data
            ])

            # Sort by date
            df = df.sort_values("date").reset_index(drop=True)

            return df

    def _calculate_expected_return(self, returns: pd.Series) -> tuple[float, float]:
        """
        Calculate expected (normal) return from estimation window.

        This uses the simple "mean model" - we assume the expected return
        is just the historical average. More sophisticated models exist
        (market model, CAPM) but this is a good starting point.

        Args:
            returns: Series of log returns in estimation window

        Returns:
            Tuple of (mean_return, std_dev)
        """
        # Remove NaN values
        clean_returns = returns.dropna()

        if len(clean_returns) < 5:
            raise ValueError("Not enough data in estimation window")

        mean_return = clean_returns.mean()
        std_dev = clean_returns.std()

        return mean_return, std_dev

    def _calculate_abnormal_returns(
        self,
        actual_returns: pd.Series,
        expected_return: float,
    ) -> pd.Series:
        """
        Calculate abnormal returns (actual - expected).

        Abnormal return tells us how much the asset deviated from normal.
        Positive = better than expected, Negative = worse than expected.

        Args:
            actual_returns: Series of actual log returns
            expected_return: The expected (mean) daily return

        Returns:
            Series of abnormal returns
        """
        return actual_returns - expected_return

    def _calculate_car(self, abnormal_returns: pd.Series) -> float:
        """
        Calculate Cumulative Abnormal Return.

        CAR is simply the sum of all abnormal returns in the event window.
        It represents the total "abnormal" impact of the event.

        Example:
            Day 0: +1.2% abnormal
            Day 1: +0.8% abnormal
            Day 2: -0.3% abnormal
            CAR = 1.2 + 0.8 - 0.3 = 1.7%

        Args:
            abnormal_returns: Series of abnormal returns

        Returns:
            Cumulative abnormal return (sum)
        """
        return abnormal_returns.sum()

    def _calculate_significance(
        self,
        car: float,
        std_dev: float,
        n_days: int,
    ) -> tuple[float, float, bool]:
        """
        Test if the CAR is statistically significant.

        We use a t-test to determine if the CAR is significantly different
        from zero. The intuition:
        - If CAR is small relative to normal volatility, it might be random
        - If CAR is large relative to normal volatility, it's likely real

        THE MATH:
        ---------
        t-statistic = CAR / (std_dev * sqrt(n_days))

        This "standardizes" the CAR by the expected volatility over the window.
        A t-stat > 2 roughly means 95% confidence the effect is real.

        Args:
            car: Cumulative abnormal return
            std_dev: Standard deviation from estimation window
            n_days: Number of days in event window

        Returns:
            Tuple of (t_statistic, p_value, is_significant)
        """
        if std_dev == 0 or n_days == 0:
            return 0.0, 1.0, False

        # Standard error of CAR
        # Under the assumption of independent returns:
        # SE(CAR) = std_dev * sqrt(n_days)
        std_error = std_dev * np.sqrt(n_days)

        # T-statistic
        t_stat = car / std_error if std_error > 0 else 0

        # P-value (two-tailed test)
        # We use the t-distribution with degrees of freedom from estimation window
        # For simplicity, we use a large df (approximates normal distribution)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=100))

        # Is it significant?
        is_significant = p_value < self.significance_level

        return t_stat, p_value, is_significant

    def analyze_event(
        self,
        event_id: int,
        symbol: str,
        event_date: date,
    ) -> Optional[EventStudyResult]:
        """
        Perform event study analysis for a single event-symbol pair.

        This is the main method that ties everything together.

        Args:
            event_id: Database ID of the event
            symbol: Ticker symbol to analyze
            event_date: Date of the event

        Returns:
            EventStudyResult with all analysis results, or None if insufficient data
        """
        logger.info(f"Analyzing event {event_id} for {symbol} on {event_date}")

        # Calculate date ranges
        # Estimation window: ends before event window starts
        estimation_end = event_date - timedelta(days=self.event_window_before + 1)
        estimation_start = estimation_end - timedelta(days=self.estimation_window + 10)  # Extra buffer for weekends

        # Event window
        event_window_start = event_date - timedelta(days=self.event_window_before)
        event_window_end = event_date + timedelta(days=self.event_window_after)

        # Fetch all data at once (more efficient)
        all_data = self._get_price_data(symbol, estimation_start, event_window_end)

        if all_data.empty:
            logger.warning(f"No price data for {symbol}")
            return None

        # Split into estimation and event windows
        estimation_data = all_data[
            (all_data["date"] >= estimation_start) &
            (all_data["date"] <= estimation_end)
        ]

        event_data = all_data[
            (all_data["date"] >= event_window_start) &
            (all_data["date"] <= event_window_end)
        ]

        # Check we have enough data
        if len(estimation_data) < 10:
            logger.warning(f"Insufficient estimation data for {symbol}: {len(estimation_data)} days")
            return None

        if len(event_data) < 1:
            logger.warning(f"No event window data for {symbol}")
            return None

        # Step 1: Calculate expected return from estimation window
        try:
            expected_return, std_dev = self._calculate_expected_return(
                estimation_data["log_return"]
            )
        except ValueError as e:
            logger.warning(f"Could not calculate expected return: {e}")
            return None

        # Step 2: Calculate abnormal returns in event window
        event_returns = event_data["log_return"].dropna()
        abnormal_returns = self._calculate_abnormal_returns(event_returns, expected_return)

        # Step 3: Calculate CAR
        car = self._calculate_car(abnormal_returns)

        # Step 4: Test significance
        t_stat, p_value, is_significant = self._calculate_significance(
            car, std_dev, len(event_returns)
        )

        # Actual total return in event window
        actual_return = event_returns.sum()

        # Build result
        result = EventStudyResult(
            event_id=event_id,
            symbol=symbol,
            event_date=event_date,
            car=car,
            abnormal_returns=abnormal_returns.tolist(),
            t_statistic=t_stat,
            p_value=p_value,
            is_significant=is_significant,
            expected_return=expected_return,
            std_dev=std_dev,
            actual_return=actual_return,
            estimation_start=estimation_data["date"].min(),
            estimation_end=estimation_data["date"].max(),
            event_window_start=event_data["date"].min(),
            event_window_end=event_data["date"].max(),
            estimation_days=len(estimation_data),
            event_days=len(event_data),
        )

        logger.info(
            f"Event {event_id} {symbol}: CAR={car:.4f}, "
            f"t={t_stat:.2f}, p={p_value:.4f}, sig={is_significant}"
        )

        return result

    def save_result(self, result: EventStudyResult) -> None:
        """
        Save event study result to database.

        Args:
            result: EventStudyResult to save
        """
        with get_session() as session:
            save_analysis_result(session, {
                "event_id": result.event_id,
                "symbol": result.symbol,
                "analysis_type": "event_study",
                "car": result.car,
                "car_t_stat": result.t_statistic,
                "car_p_value": result.p_value,
                "estimation_window_start": result.estimation_start,
                "estimation_window_end": result.estimation_end,
                "event_window_start": result.event_window_start,
                "event_window_end": result.event_window_end,
                "expected_return": result.expected_return,
                "actual_return": result.actual_return,
                "abnormal_return": result.car,  # CAR is total abnormal return
                "is_significant": result.is_significant,
                "is_anomaly": False,  # Event study doesn't detect anomalies
            })

    def analyze_event_for_symbols(
        self,
        event_id: int,
        event_date: date,
        symbols: list[str],
        save: bool = True,
    ) -> list[EventStudyResult]:
        """
        Analyze an event's impact on multiple symbols.

        Args:
            event_id: Database ID of the event
            event_date: Date of the event
            symbols: List of symbols to analyze
            save: Whether to save results to database

        Returns:
            List of EventStudyResult objects
        """
        results = []

        for symbol in symbols:
            try:
                result = self.analyze_event(event_id, symbol, event_date)
                if result:
                    results.append(result)
                    if save:
                        self.save_result(result)
            except Exception as e:
                logger.error(f"Error analyzing {symbol} for event {event_id}: {e}")

        return results


def explain_result(result: EventStudyResult) -> str:
    """
    Generate a human-readable explanation of an event study result.

    Args:
        result: EventStudyResult to explain

    Returns:
        String with explanation
    """
    direction = "increased" if result.car > 0 else "decreased"
    car_pct = result.car * 100

    explanation = f"""
EVENT STUDY RESULT
==================
Symbol: {result.symbol}
Event Date: {result.event_date}

FINDINGS:
- The asset {direction} by {abs(car_pct):.2f}% more than expected
- This is {"" if result.is_significant else "NOT "}statistically significant
  (p-value: {result.p_value:.4f}, threshold: 0.05)

DETAILS:
- Expected daily return: {result.expected_return*100:.3f}%
- Actual return over window: {result.actual_return*100:.2f}%
- Abnormal return (CAR): {car_pct:.2f}%
- T-statistic: {result.t_statistic:.2f}

DATA QUALITY:
- Estimation window: {result.estimation_days} trading days
- Event window: {result.event_days} trading days
"""
    return explanation
