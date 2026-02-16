"""
Correlation analysis between geopolitical event features and market returns.

Provides:
1. Point-in-time Pearson/Spearman correlation between event metrics and returns
2. Rolling window correlation to see how relationships change over time
3. Cross-symbol comparison to find the strongest event-market pairs
"""

from dataclasses import dataclass
from datetime import date
from typing import Literal, Optional
import logging

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.feature_engineering import FeatureEngineering
from src.config.constants import get_all_symbols, EVENT_GROUPS, CAMEO_CATEGORIES

logger = logging.getLogger(__name__)

# Event metrics available for correlation
EVENT_METRICS = [
    "goldstein_mean",
    "mentions_total",
    "avg_tone",
    "conflict_count",
    "cooperation_count",
]


@dataclass
class CorrelationResult:
    """Result of correlating one event metric with market returns."""
    symbol: str
    event_metric: str
    correlation: float
    p_value: float
    n_observations: int
    method: str


@dataclass
class RollingCorrelationResult:
    """Rolling correlation timeseries."""
    dates: list
    correlations: list[float]
    upper_ci: list[float]
    lower_ci: list[float]
    event_metric: str
    window_days: int


class CorrelationAnalyzer:
    """Analyzes correlation between geopolitical events and market returns."""

    def __init__(self):
        self.fe = FeatureEngineering()

    def _get_merged_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Fetch and merge market + event data into a single DataFrame."""
        market_df = self.fe.fetch_market_data(symbol, start_date, end_date)
        if market_df.empty:
            return pd.DataFrame()

        events_df = self.fe.fetch_events(start_date, end_date, include_cooperation=True)
        if events_df.empty:
            return pd.DataFrame()

        event_agg = self.fe.aggregate_events(
            events_df,
            goldstein_metrics=["mean"],
            mention_metrics=["sum"],
            include_cooperation=True,
        )

        merged = self.fe.merge_market_events(market_df, event_agg)
        return merged

    def compute_correlations(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        method: Literal["pearson", "spearman"] = "pearson",
    ) -> list[CorrelationResult]:
        """
        Compute correlation between each event metric and log_return.

        Returns one CorrelationResult per event metric.
        """
        merged = self._get_merged_data(symbol, start_date, end_date)
        if merged.empty or len(merged) < 10:
            return []

        results = []
        corr_func = stats.pearsonr if method == "pearson" else stats.spearmanr

        for metric in EVENT_METRICS:
            if metric not in merged.columns:
                continue

            x = merged[metric].values
            y = merged["log_return"].values

            # Skip if no variance
            if np.std(x) == 0 or np.std(y) == 0:
                continue

            corr, p_val = corr_func(x, y)

            results.append(CorrelationResult(
                symbol=symbol,
                event_metric=metric,
                correlation=float(corr),
                p_value=float(p_val),
                n_observations=len(merged),
                method=method,
            ))

        return results

    def compute_rolling_correlation(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        event_metric: str = "conflict_count",
        window_days: int = 30,
    ) -> Optional[RollingCorrelationResult]:
        """
        Compute rolling window correlation between an event metric and returns.

        Returns a timeseries of correlation values with 95% confidence intervals.
        """
        merged = self._get_merged_data(symbol, start_date, end_date)
        if merged.empty or event_metric not in merged.columns:
            return None

        if len(merged) < window_days + 10:
            return None

        rolling_corr = merged["log_return"].rolling(
            window=window_days, min_periods=max(10, window_days // 2)
        ).corr(merged[event_metric])

        # Drop NaN from the beginning
        valid = rolling_corr.dropna()
        if valid.empty:
            return None

        valid_dates = merged.loc[valid.index, "date"].tolist()
        corr_values = valid.tolist()

        # Fisher z-transform for confidence intervals
        z_critical = 1.96
        upper_ci = []
        lower_ci = []
        for r in corr_values:
            r_clipped = np.clip(r, -0.999, 0.999)
            z = np.arctanh(r_clipped)
            se = 1.0 / np.sqrt(max(window_days - 3, 1))
            upper_ci.append(float(np.tanh(z + z_critical * se)))
            lower_ci.append(float(np.tanh(z - z_critical * se)))

        return RollingCorrelationResult(
            dates=valid_dates,
            correlations=corr_values,
            upper_ci=upper_ci,
            lower_ci=lower_ci,
            event_metric=event_metric,
            window_days=window_days,
        )

    def top_correlated_pairs(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
        method: Literal["pearson", "spearman"] = "pearson",
        limit: int = 20,
    ) -> pd.DataFrame:
        """
        Find strongest correlations across all symbol-event metric pairs.

        Returns a DataFrame sorted by absolute correlation strength.
        """
        rows = []

        for symbol in symbols:
            try:
                results = self.compute_correlations(
                    symbol, start_date, end_date, method
                )
                for r in results:
                    rows.append({
                        "symbol": symbol,
                        "event_metric": r.event_metric,
                        "correlation": r.correlation,
                        "abs_correlation": abs(r.correlation),
                        "p_value": r.p_value,
                        "n_observations": r.n_observations,
                        "direction": "positive" if r.correlation > 0 else "negative",
                    })
            except Exception as e:
                logger.warning(f"Error computing correlation for {symbol}: {e}")

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df = df.sort_values("abs_correlation", ascending=False).head(limit)
        return df.reset_index(drop=True)

    def correlation_heatmap(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
        method: Literal["pearson", "spearman"] = "pearson",
    ) -> dict:
        """
        Compute correlation matrix: symbols x event_metrics.

        Returns dict with symbols, event_types, and matrix suitable for heatmap.
        """
        matrix = []

        for symbol in symbols:
            results = self.compute_correlations(
                symbol, start_date, end_date, method
            )
            row = {}
            for r in results:
                row[r.event_metric] = r.correlation
            matrix.append(row)

        # Build a consistent matrix
        all_metrics = EVENT_METRICS
        grid = []
        for row in matrix:
            grid.append([row.get(m, 0.0) for m in all_metrics])

        return {
            "symbols": symbols,
            "event_metrics": all_metrics,
            "matrix": grid,
        }
