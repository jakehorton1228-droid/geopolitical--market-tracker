"""
Regression Analysis Module.

WHY REGRESSION?
---------------
While event studies analyze individual events, regression helps us understand
GENERAL patterns:
- "Do conflict events move oil prices more than diplomatic events?"
- "Does media coverage (num_mentions) predict larger market moves?"
- "Which countries' events have the biggest market impact?"

TYPES OF REGRESSION:
-------------------
1. LINEAR REGRESSION (OLS - Ordinary Least Squares)
   - Simplest form: y = β₀ + β₁x + ε
   - Finds the line of best fit
   - Coefficients tell you the relationship strength

2. MULTIPLE REGRESSION
   - Multiple predictors: y = β₀ + β₁x₁ + β₂x₂ + ... + ε
   - Controls for multiple factors simultaneously

3. LOGISTIC REGRESSION
   - For binary outcomes (yes/no, up/down)
   - Predicts probability of an outcome

KEY OUTPUTS:
-----------
- Coefficients (β): How much does Y change when X increases by 1?
- R-squared: How much variance does the model explain? (0-1)
- P-values: Is each coefficient statistically significant?
- Standard errors: Uncertainty in coefficient estimates

USAGE:
------
    from src.analysis.regression import EventRegression

    reg = EventRegression()

    # Analyze what drives market reactions
    results = reg.analyze_event_impact("CL=F", start_date, end_date)

    # Interpret
    print(f"Goldstein coefficient: {results['coefficients']['goldstein_scale']}")
    print(f"R-squared: {results['r_squared']}")
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional
import logging

import numpy as np
import pandas as pd
from scipy import stats

from src.db.connection import get_session
from src.db.models import Event, MarketData
from src.db.queries import get_events_by_date_range, get_market_data
from src.config.constants import get_event_group

logger = logging.getLogger(__name__)


@dataclass
class RegressionResult:
    """Container for regression results."""

    # Model info
    dependent_var: str  # What we're predicting (e.g., "log_return")
    independent_vars: list[str]  # Predictors

    # Coefficients
    coefficients: dict[str, float]  # Variable name → coefficient
    std_errors: dict[str, float]  # Variable name → standard error
    t_statistics: dict[str, float]  # Variable name → t-stat
    p_values: dict[str, float]  # Variable name → p-value

    # Model fit
    r_squared: float  # Variance explained (0-1)
    adj_r_squared: float  # Adjusted for number of predictors
    f_statistic: float  # Overall model significance
    f_p_value: float

    # Data info
    n_observations: int
    n_predictors: int


class EventRegression:
    """
    Regression analysis for event-market relationships.

    This class helps answer questions like:
    - Do more severe events (higher |Goldstein|) cause bigger market moves?
    - Does media coverage predict market reaction?
    - Which event types have the strongest market impact?
    """

    def __init__(self):
        """Initialize the regression analyzer."""
        pass

    def _prepare_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Prepare merged event + market data for regression.

        This creates a dataset where each row is a day with:
        - Market return for that day
        - Aggregated event features for that day

        Args:
            symbol: Ticker symbol
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame ready for regression
        """
        # Get market data
        with get_session() as session:
            market_data = get_market_data(session, symbol, start_date, end_date)

            if not market_data:
                return pd.DataFrame()

            market_df = pd.DataFrame([
                {
                    "date": m.date,
                    "log_return": m.log_return,
                    "close": float(m.close),
                }
                for m in market_data
            ])

        # Get events
        with get_session() as session:
            events = get_events_by_date_range(session, start_date, end_date)

            if not events:
                return pd.DataFrame()

            events_df = pd.DataFrame([
                {
                    "date": e.event_date,
                    "goldstein_scale": e.goldstein_scale or 0,
                    "num_mentions": e.num_mentions or 0,
                    "num_articles": e.num_articles or 0,
                    "avg_tone": e.avg_tone or 0,
                    "event_root_code": e.event_root_code,
                    "is_conflict": e.event_root_code in ["18", "19", "20"],
                }
                for e in events
            ])

        # Aggregate events by date (multiple events can happen per day)
        event_agg = events_df.groupby("date").agg({
            "goldstein_scale": ["mean", "min", "max", "std"],  # Avg, worst, best, spread
            "num_mentions": "sum",  # Total media coverage
            "num_articles": "sum",
            "avg_tone": "mean",
            "is_conflict": "sum",  # Count of conflict events
        }).reset_index()

        # Flatten column names
        event_agg.columns = [
            "date",
            "goldstein_mean", "goldstein_min", "goldstein_max", "goldstein_std",
            "total_mentions", "total_articles", "avg_tone", "conflict_count",
        ]

        # Fill NaN in std (happens when only one event)
        event_agg["goldstein_std"] = event_agg["goldstein_std"].fillna(0)

        # Merge market and event data
        merged = pd.merge(market_df, event_agg, on="date", how="left")

        # Fill NaN for days with no events
        merged = merged.fillna({
            "goldstein_mean": 0,
            "goldstein_min": 0,
            "goldstein_max": 0,
            "goldstein_std": 0,
            "total_mentions": 0,
            "total_articles": 0,
            "avg_tone": 0,
            "conflict_count": 0,
        })

        # Remove rows with NaN returns
        merged = merged.dropna(subset=["log_return"])

        return merged

    def run_ols_regression(
        self,
        y: np.ndarray,
        X: np.ndarray,
        feature_names: list[str],
    ) -> RegressionResult:
        """
        Run Ordinary Least Squares regression.

        OLS EXPLAINED:
        -------------
        We want to find coefficients (β) that minimize the sum of squared errors:
        minimize Σ(y - Xβ)²

        The solution is: β = (X'X)⁻¹X'y

        Args:
            y: Dependent variable (what we're predicting)
            X: Independent variables (predictors), should include intercept column
            feature_names: Names of features (including "intercept")

        Returns:
            RegressionResult with all statistics
        """
        n = len(y)  # Number of observations
        k = X.shape[1]  # Number of predictors (including intercept)

        # OLS estimation: β = (X'X)⁻¹X'y
        # X' means X transpose
        XtX = X.T @ X  # X transpose times X
        XtX_inv = np.linalg.pinv(XtX)  # Pseudo-inverse (handles near-singular matrices)
        Xty = X.T @ y
        beta = XtX_inv @ Xty  # Coefficients

        # Predictions and residuals
        y_pred = X @ beta
        residuals = y - y_pred

        # Sum of squares
        ss_res = np.sum(residuals ** 2)  # Residual sum of squares
        ss_tot = np.sum((y - np.mean(y)) ** 2)  # Total sum of squares

        # R-squared: proportion of variance explained
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Adjusted R-squared: penalizes adding more predictors
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k) if n > k else 0

        # Standard error of residuals
        mse = ss_res / (n - k) if n > k else 0  # Mean squared error
        se_residuals = np.sqrt(mse)

        # Standard errors of coefficients
        # SE(β) = sqrt(MSE * diag((X'X)⁻¹))
        var_beta = mse * np.diag(XtX_inv)
        se_beta = np.sqrt(np.maximum(var_beta, 0))  # Avoid negative due to numerical issues

        # T-statistics: β / SE(β)
        t_stats = beta / np.where(se_beta > 0, se_beta, 1)

        # P-values (two-tailed test)
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k))

        # F-statistic for overall model significance
        if k > 1 and ss_res > 0:
            ss_reg = ss_tot - ss_res
            f_stat = (ss_reg / (k - 1)) / (ss_res / (n - k))
            f_p_value = 1 - stats.f.cdf(f_stat, k - 1, n - k)
        else:
            f_stat = 0
            f_p_value = 1

        # Build result
        return RegressionResult(
            dependent_var="log_return",
            independent_vars=feature_names,
            coefficients=dict(zip(feature_names, beta)),
            std_errors=dict(zip(feature_names, se_beta)),
            t_statistics=dict(zip(feature_names, t_stats)),
            p_values=dict(zip(feature_names, p_values)),
            r_squared=r_squared,
            adj_r_squared=adj_r_squared,
            f_statistic=f_stat,
            f_p_value=f_p_value,
            n_observations=n,
            n_predictors=k - 1,  # Excluding intercept
        )

    def analyze_event_impact(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        features: list[str] = None,
    ) -> Optional[RegressionResult]:
        """
        Analyze how event characteristics predict market returns.

        This runs a multiple regression:
        return = β₀ + β₁(goldstein) + β₂(mentions) + β₃(conflicts) + ε

        Args:
            symbol: Ticker symbol to analyze
            start_date: Start of analysis period
            end_date: End of analysis period
            features: Which features to include (default: all)

        Returns:
            RegressionResult with coefficients and statistics
        """
        logger.info(f"Running regression analysis for {symbol}")

        # Prepare data
        data = self._prepare_data(symbol, start_date, end_date)

        if data.empty or len(data) < 10:
            logger.warning(f"Insufficient data for regression: {len(data)} rows")
            return None

        # Default features
        if features is None:
            features = [
                "goldstein_mean",
                "total_mentions",
                "conflict_count",
            ]

        # Verify features exist
        available_features = [f for f in features if f in data.columns]
        if not available_features:
            logger.warning("No valid features for regression")
            return None

        # Prepare arrays
        y = data["log_return"].values
        X_data = data[available_features].values

        # Add intercept column (column of 1s)
        intercept = np.ones((len(y), 1))
        X = np.hstack([intercept, X_data])

        feature_names = ["intercept"] + available_features

        # Run regression
        result = self.run_ols_regression(y, X, feature_names)

        logger.info(f"Regression complete: R² = {result.r_squared:.4f}")

        return result


def explain_regression(result: RegressionResult) -> str:
    """
    Generate a human-readable explanation of regression results.

    Args:
        result: RegressionResult to explain

    Returns:
        Explanation string
    """
    lines = [
        "REGRESSION ANALYSIS RESULTS",
        "=" * 50,
        f"Dependent Variable: {result.dependent_var}",
        f"Observations: {result.n_observations}",
        f"Predictors: {result.n_predictors}",
        "",
        "MODEL FIT:",
        f"  R-squared: {result.r_squared:.4f} ({result.r_squared*100:.1f}% of variance explained)",
        f"  Adjusted R-squared: {result.adj_r_squared:.4f}",
        f"  F-statistic: {result.f_statistic:.2f} (p={result.f_p_value:.4f})",
        "",
        "COEFFICIENTS:",
        "-" * 50,
        f"{'Variable':<20} {'Coef':>10} {'Std Err':>10} {'t-stat':>8} {'p-value':>10}",
        "-" * 50,
    ]

    for var in result.independent_vars:
        coef = result.coefficients[var]
        se = result.std_errors[var]
        t = result.t_statistics[var]
        p = result.p_values[var]
        sig = "*" if p < 0.05 else ""

        lines.append(f"{var:<20} {coef:>10.6f} {se:>10.6f} {t:>8.2f} {p:>9.4f}{sig}")

    lines.extend([
        "-" * 50,
        "* = significant at p < 0.05",
        "",
        "INTERPRETATION:",
    ])

    # Add interpretations for significant coefficients
    for var in result.independent_vars:
        if var == "intercept":
            continue
        if result.p_values[var] < 0.05:
            coef = result.coefficients[var]
            direction = "increases" if coef > 0 else "decreases"
            lines.append(
                f"  - When {var} increases by 1, return {direction} by {abs(coef)*100:.4f}%"
            )

    if result.r_squared < 0.05:
        lines.append("\n  Note: Low R² suggests events explain little of daily return variance.")
        lines.append("  This is normal - markets are affected by many factors beyond events.")

    return "\n".join(lines)
