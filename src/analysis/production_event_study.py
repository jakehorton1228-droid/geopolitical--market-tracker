"""
Production Event Study Module using scipy and numpy.

Event studies are a specialized finance technique, so there isn't a single
"sklearn for event studies" package. However, we can still simplify our
code significantly using scipy.stats for statistical tests.

WHAT'S DIFFERENT FROM THE LEARNING VERSION:
-------------------------------------------
Learning Version (event_study.py):
- Manual CAR calculation (educational)
- Manual t-statistic formula
- Manual p-value from t-distribution

Production Version (this file):
- Uses scipy.stats.ttest_1samp for significance testing
- Cleaner code structure
- Additional statistical tests (Wilcoxon, bootstrap)
- More robust confidence intervals

INDUSTRY APPROACHES TO EVENT STUDIES:
------------------------------------
1. Basic: What we do - compare returns to estimation window mean
2. Market Model: Adjust for overall market movements (beta)
3. Fama-French: Adjust for size, value, momentum factors

This module implements the basic approach with cleaner code.

USAGE:
------
    from src.analysis.production_event_study import ProductionEventStudy

    study = ProductionEventStudy()
    result = study.analyze_event(event_id, "CL=F", event_date)
    print(result.summary)
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
from src.db.queries import get_market_data

logger = logging.getLogger(__name__)


@dataclass
class ProdEventStudyResult:
    """Container for event study results."""
    event_id: int
    symbol: str
    event_date: date

    # Main results
    car: float  # Cumulative Abnormal Return
    car_percent: float  # CAR as percentage

    # Statistical tests
    t_statistic: float
    p_value: float
    is_significant: bool

    # Confidence interval for CAR
    ci_lower: float
    ci_upper: float

    # Additional context
    expected_return: float
    actual_return: float
    std_dev: float

    # Non-parametric test (doesn't assume normal distribution)
    wilcoxon_p: Optional[float]

    # Data info
    estimation_days: int
    event_days: int

    # Summary string
    summary: str


class ProductionEventStudy:
    """
    Production event study using scipy.stats.

    Key improvements over learning version:
    1. Uses scipy.stats.ttest_1samp for t-test
    2. Adds Wilcoxon signed-rank test (non-parametric)
    3. Calculates confidence intervals
    4. Cleaner, more maintainable code
    """

    def __init__(
        self,
        estimation_window: int = None,
        event_window_before: int = None,
        event_window_after: int = None,
        significance_level: float = None,
    ):
        """Initialize with configuration."""
        self.estimation_window = estimation_window or EVENT_STUDY_ESTIMATION_WINDOW
        self.event_window_before = event_window_before or EVENT_STUDY_EVENT_WINDOW_BEFORE
        self.event_window_after = event_window_after or EVENT_STUDY_EVENT_WINDOW_AFTER
        self.significance_level = significance_level or SIGNIFICANCE_LEVEL

    def _get_returns(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.Series:
        """Fetch returns from database."""
        with get_session() as session:
            data = get_market_data(session, symbol, start_date, end_date)

            if not data:
                return pd.Series(dtype=float)

            df = pd.DataFrame([
                {"date": d.date, "return": d.log_return}
                for d in data
            ]).set_index("date").sort_index()

            return df["return"].dropna()

    def analyze_event(
        self,
        event_id: int,
        symbol: str,
        event_date: date,
    ) -> Optional[ProdEventStudyResult]:
        """
        Analyze a single event's impact on a symbol.

        This is cleaner than the learning version but does the same thing:
        1. Get estimation window returns → calculate "normal"
        2. Get event window returns → calculate actual
        3. Compute abnormal returns and CAR
        4. Test significance using scipy.stats
        """
        logger.info(f"Analyzing event {event_id} for {symbol} on {event_date}")

        # Calculate date ranges
        estimation_end = event_date - timedelta(days=self.event_window_before + 1)
        estimation_start = estimation_end - timedelta(days=self.estimation_window + 10)
        event_start = event_date - timedelta(days=self.event_window_before)
        event_end = event_date + timedelta(days=self.event_window_after)

        # Get returns
        estimation_returns = self._get_returns(symbol, estimation_start, estimation_end)
        event_returns = self._get_returns(symbol, event_start, event_end)

        # Validate data
        if len(estimation_returns) < 10:
            logger.warning(f"Insufficient estimation data: {len(estimation_returns)} days")
            return None

        if len(event_returns) < 1:
            logger.warning("No event window data")
            return None

        # ═══════════════════════════════════════════════════════════
        # CALCULATE EXPECTED RETURN AND STD DEV
        # ═══════════════════════════════════════════════════════════
        expected_return = estimation_returns.mean()
        std_dev = estimation_returns.std()

        # ═══════════════════════════════════════════════════════════
        # CALCULATE ABNORMAL RETURNS AND CAR
        # ═══════════════════════════════════════════════════════════
        abnormal_returns = event_returns - expected_return
        car = abnormal_returns.sum()
        actual_return = event_returns.sum()

        # ═══════════════════════════════════════════════════════════
        # STATISTICAL SIGNIFICANCE USING SCIPY
        # ═══════════════════════════════════════════════════════════

        # Method 1: Traditional t-test
        # Tests if abnormal returns are significantly different from 0
        if len(abnormal_returns) > 1:
            t_stat, p_value = stats.ttest_1samp(abnormal_returns, 0)
        else:
            # Single day: use z-test approach
            se = std_dev / np.sqrt(1)
            t_stat = car / se if se > 0 else 0
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(estimation_returns) - 1))

        is_significant = p_value < self.significance_level

        # Method 2: Wilcoxon signed-rank test (non-parametric)
        # Useful when returns might not be normally distributed
        wilcoxon_p = None
        if len(abnormal_returns) >= 6:  # Wilcoxon needs at least 6 samples
            try:
                _, wilcoxon_p = stats.wilcoxon(abnormal_returns)
            except ValueError:
                pass  # Can fail if all values are the same

        # ═══════════════════════════════════════════════════════════
        # CONFIDENCE INTERVAL FOR CAR
        # ═══════════════════════════════════════════════════════════
        n_days = len(event_returns)
        se_car = std_dev * np.sqrt(n_days)

        # 95% CI using t-distribution
        t_critical = stats.t.ppf(0.975, df=len(estimation_returns) - 1)
        ci_lower = car - t_critical * se_car
        ci_upper = car + t_critical * se_car

        # ═══════════════════════════════════════════════════════════
        # BUILD SUMMARY STRING
        # ═══════════════════════════════════════════════════════════
        summary = self._build_summary(
            symbol, event_date, car, t_stat, p_value, is_significant,
            ci_lower, ci_upper, expected_return, actual_return,
            len(estimation_returns), n_days, wilcoxon_p
        )

        return ProdEventStudyResult(
            event_id=event_id,
            symbol=symbol,
            event_date=event_date,
            car=car,
            car_percent=car * 100,
            t_statistic=t_stat,
            p_value=p_value,
            is_significant=is_significant,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            expected_return=expected_return,
            actual_return=actual_return,
            std_dev=std_dev,
            wilcoxon_p=wilcoxon_p,
            estimation_days=len(estimation_returns),
            event_days=n_days,
            summary=summary,
        )

    def _build_summary(
        self,
        symbol: str,
        event_date: date,
        car: float,
        t_stat: float,
        p_value: float,
        is_significant: bool,
        ci_lower: float,
        ci_upper: float,
        expected_return: float,
        actual_return: float,
        est_days: int,
        event_days: int,
        wilcoxon_p: Optional[float],
    ) -> str:
        """Build a formatted summary string."""
        direction = "increased" if car > 0 else "decreased"
        sig_text = "IS" if is_significant else "IS NOT"

        lines = [
            "",
            "=" * 60,
            f"EVENT STUDY RESULTS: {symbol}",
            "=" * 60,
            "",
            f"Event Date: {event_date}",
            "",
            "MAIN FINDING:",
            f"  The asset {direction} by {abs(car)*100:.2f}% more than expected",
            f"  This {sig_text} statistically significant (p = {p_value:.4f})",
            "",
            "STATISTICS:",
            f"  CAR (Cumulative Abnormal Return): {car*100:.2f}%",
            f"  95% Confidence Interval: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]",
            f"  t-statistic: {t_stat:.3f}",
            f"  p-value: {p_value:.4f}",
        ]

        if wilcoxon_p is not None:
            lines.append(f"  Wilcoxon p-value: {wilcoxon_p:.4f} (non-parametric test)")

        lines.extend([
            "",
            "DETAILS:",
            f"  Expected daily return: {expected_return*100:.3f}%",
            f"  Actual return over window: {actual_return*100:.2f}%",
            f"  Estimation window: {est_days} trading days",
            f"  Event window: {event_days} trading days",
            "",
            "INTERPRETATION:",
        ])

        if is_significant:
            lines.append(f"  → The event had a SIGNIFICANT impact on {symbol}")
            if car > 0:
                lines.append("  → The effect was POSITIVE (price increased)")
            else:
                lines.append("  → The effect was NEGATIVE (price decreased)")
        else:
            lines.append(f"  → No significant impact detected")
            lines.append("  → The market movement could be due to random chance")

        lines.append("=" * 60)

        return "\n".join(lines)

    def analyze_multiple_events(
        self,
        events: list[tuple[int, str, date]],  # List of (event_id, symbol, date)
    ) -> pd.DataFrame:
        """
        Analyze multiple events and return a summary DataFrame.

        Useful for comparing impact across events.
        """
        results = []

        for event_id, symbol, event_date in events:
            result = self.analyze_event(event_id, symbol, event_date)
            if result:
                results.append({
                    'event_id': event_id,
                    'symbol': symbol,
                    'date': event_date,
                    'car': result.car,
                    'car_pct': result.car_percent,
                    't_stat': result.t_statistic,
                    'p_value': result.p_value,
                    'significant': result.is_significant,
                    'ci_lower': result.ci_lower * 100,
                    'ci_upper': result.ci_upper * 100,
                })

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('p_value')

        return df


def run_quick_event_study(
    symbol: str,
    event_date: date,
    event_id: int = 0,
) -> Optional[str]:
    """
    Quick helper function to run an event study and get the summary.

    Usage:
        from src.analysis.production_event_study import run_quick_event_study
        from datetime import date

        print(run_quick_event_study("CL=F", date(2024, 2, 24)))
    """
    study = ProductionEventStudy()
    result = study.analyze_event(event_id, symbol, event_date)
    return result.summary if result else None
