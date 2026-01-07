"""
Production Regression Module using statsmodels.

This is the INDUSTRY-STANDARD way to do regression analysis, especially
when you need statistical inference (p-values, confidence intervals).

WHY STATSMODELS OVER SKLEARN FOR REGRESSION?
--------------------------------------------
sklearn's LinearRegression:
- Great for predictions
- Does NOT give p-values, t-stats, confidence intervals
- Used when you just want to predict

statsmodels OLS:
- Gives FULL statistical output (like academic papers)
- P-values, t-stats, confidence intervals, F-test
- Used when you want to understand relationships
- What economists, researchers, and analysts use

COMPARISON TO LEARNING VERSION:
-------------------------------
Learning Version (regression.py):
- 400+ lines of code
- Manual matrix math: β = (XᵀX)⁻¹Xᵀy
- Manual R², t-stats, p-values
- Educational

Production Version (this file):
- ~100 lines of code
- model.fit() does everything
- model.summary() gives beautiful output
- What you'd use in a research paper or at work

USAGE:
------
    from src.analysis.production_regression import ProductionRegression

    reg = ProductionRegression()
    result = reg.analyze("CL=F", start_date, end_date)
    print(result.summary)  # Full statistical report
"""

from dataclasses import dataclass
from datetime import date
from typing import Optional
import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from src.db.connection import get_session
from src.db.queries import get_events_by_date_range, get_market_data

logger = logging.getLogger(__name__)


@dataclass
class ProdRegressionResult:
    """Container for regression results."""
    symbol: str

    # Model fit
    r_squared: float
    adj_r_squared: float
    f_statistic: float
    f_pvalue: float

    # Coefficients with full stats
    coefficients: dict[str, float]
    std_errors: dict[str, float]
    t_values: dict[str, float]
    p_values: dict[str, float]
    conf_int_lower: dict[str, float]
    conf_int_upper: dict[str, float]

    # Data info
    n_observations: int
    n_features: int

    # The full summary string from statsmodels
    summary: str


class ProductionRegression:
    """
    Production regression analysis using statsmodels.

    statsmodels gives us everything we calculated manually:
    - Coefficients
    - Standard errors
    - T-statistics
    - P-values
    - Confidence intervals
    - R-squared
    - F-statistic

    All in one line: model = sm.OLS(y, X).fit()
    """

    def __init__(self):
        """Initialize the regression analyzer."""
        self.feature_names: list[str] = []

    def _prepare_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Prepare features and target for regression."""
        # Get market data
        with get_session() as session:
            market_data = get_market_data(session, symbol, start_date, end_date)
            if not market_data:
                return np.array([]), np.array([]), []

            market_df = pd.DataFrame([
                {"date": m.date, "log_return": m.log_return}
                for m in market_data
            ])

        # Get events
        with get_session() as session:
            events = get_events_by_date_range(session, start_date, end_date)
            if not events:
                return np.array([]), np.array([]), []

            events_df = pd.DataFrame([
                {
                    "date": e.event_date,
                    "goldstein_scale": e.goldstein_scale or 0,
                    "num_mentions": e.num_mentions or 0,
                    "avg_tone": e.avg_tone or 0,
                    "is_conflict": 1 if e.event_root_code in ["18", "19", "20"] else 0,
                }
                for e in events
            ])

        # Aggregate by date
        event_agg = events_df.groupby("date").agg({
            "goldstein_scale": "mean",
            "num_mentions": "sum",
            "avg_tone": "mean",
            "is_conflict": "sum",
        }).reset_index()

        event_agg.columns = ["date", "goldstein_mean", "mentions_total", "avg_tone", "conflict_count"]

        # Merge
        merged = pd.merge(market_df, event_agg, on="date", how="left").fillna(0)
        merged = merged.dropna(subset=["log_return"])

        if len(merged) < 10:
            return np.array([]), np.array([]), []

        # Target
        y = merged["log_return"].values

        # Features
        feature_cols = ["goldstein_mean", "mentions_total", "avg_tone", "conflict_count"]
        X = merged[feature_cols].values

        return X, y, feature_cols

    def analyze(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        features: list[str] = None,
    ) -> Optional[ProdRegressionResult]:
        """
        Run regression analysis using statsmodels.

        This is the key method. Notice how simple it is compared to
        our manual implementation!
        """
        logger.info(f"Running production regression for {symbol}")

        # Prepare data
        X, y, feature_names = self._prepare_data(symbol, start_date, end_date)

        if len(X) == 0:
            logger.warning(f"No data for {symbol}")
            return None

        self.feature_names = feature_names

        # ═══════════════════════════════════════════════════════════
        # THIS IS WHERE STATSMODELS SHINES
        # One line does everything we did manually!
        # ═══════════════════════════════════════════════════════════

        # Add constant (intercept) - statsmodels requires this explicitly
        X_with_const = sm.add_constant(X)

        # Fit the model - this does all the matrix math internally
        model = sm.OLS(y, X_with_const).fit()

        # ═══════════════════════════════════════════════════════════
        # EXTRACT ALL THE STATISTICS
        # ═══════════════════════════════════════════════════════════

        all_names = ['const'] + feature_names

        # Confidence intervals
        conf_int = model.conf_int()

        result = ProdRegressionResult(
            symbol=symbol,

            # Model fit
            r_squared=model.rsquared,
            adj_r_squared=model.rsquared_adj,
            f_statistic=model.fvalue,
            f_pvalue=model.f_pvalue,

            # Coefficients and stats
            coefficients=dict(zip(all_names, model.params)),
            std_errors=dict(zip(all_names, model.bse)),
            t_values=dict(zip(all_names, model.tvalues)),
            p_values=dict(zip(all_names, model.pvalues)),
            conf_int_lower=dict(zip(all_names, conf_int[:, 0])),
            conf_int_upper=dict(zip(all_names, conf_int[:, 1])),

            # Data info
            n_observations=int(model.nobs),
            n_features=len(feature_names),

            # Full summary (this is the beautiful output)
            summary=model.summary().as_text(),
        )

        logger.info(f"{symbol}: R² = {result.r_squared:.4f}")

        return result

    def analyze_with_summary(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> Optional[str]:
        """
        Run analysis and return just the summary string.

        This is what you'd put in a report or paper.
        """
        result = self.analyze(symbol, start_date, end_date)
        if result:
            return result.summary
        return None

    def compare_markets(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Compare regression results across multiple markets.

        Returns a DataFrame for easy comparison.
        """
        rows = []

        for symbol in symbols:
            result = self.analyze(symbol, start_date, end_date)
            if result:
                # Get significant predictors
                sig_predictors = [
                    name for name, p in result.p_values.items()
                    if p < 0.05 and name != 'const'
                ]

                rows.append({
                    'symbol': symbol,
                    'r_squared': result.r_squared,
                    'adj_r_squared': result.adj_r_squared,
                    'f_pvalue': result.f_pvalue,
                    'n_obs': result.n_observations,
                    'significant_predictors': ', '.join(sig_predictors) or 'None',
                    'goldstein_coef': result.coefficients.get('goldstein_mean', 0),
                    'goldstein_pval': result.p_values.get('goldstein_mean', 1),
                })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values('r_squared', ascending=False)

        return df


def print_interpretation(result: ProdRegressionResult) -> None:
    """
    Print a human-readable interpretation of results.

    This shows how to extract insights from the statsmodels output.
    """
    print(f"\n{'='*60}")
    print(f"REGRESSION RESULTS: {result.symbol}")
    print(f"{'='*60}")

    print(f"\nMODEL FIT:")
    print(f"  R-squared: {result.r_squared:.4f} ({result.r_squared*100:.1f}% variance explained)")
    print(f"  Adjusted R-squared: {result.adj_r_squared:.4f}")
    print(f"  F-statistic p-value: {result.f_pvalue:.4f}", end="")
    print(" *" if result.f_pvalue < 0.05 else "")

    print(f"\nCOEFFICIENTS:")
    print(f"{'Variable':<18} {'Coef':>10} {'p-value':>10} {'95% CI':>20} {'Sig':>5}")
    print("-" * 65)

    for name in result.coefficients.keys():
        coef = result.coefficients[name]
        pval = result.p_values[name]
        ci_low = result.conf_int_lower[name]
        ci_high = result.conf_int_upper[name]
        sig = "*" if pval < 0.05 else ""

        ci_str = f"[{ci_low:.4f}, {ci_high:.4f}]"
        print(f"{name:<18} {coef:>10.6f} {pval:>10.4f} {ci_str:>20} {sig:>5}")

    print("\nINTERPRETATION:")
    for name, pval in result.p_values.items():
        if name == 'const':
            continue
        if pval < 0.05:
            coef = result.coefficients[name]
            direction = "increases" if coef > 0 else "decreases"
            print(f"  - {name}: When it increases by 1, return {direction} by {abs(coef)*100:.4f}%")

    if result.r_squared < 0.05:
        print(f"\n  Note: Low R² ({result.r_squared:.1%}) is normal for daily returns.")
        print("  Markets are affected by many factors beyond geopolitical events.")
