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
        """
        Prepare features and target for regression.

        Delegates to shared FeatureEngineering module for consistent
        feature preparation across all analysis modules.
        """
        from src.analysis.feature_engineering import FeatureEngineering

        fe = FeatureEngineering()
        return fe.prepare_regression_features(symbol, start_date, end_date)

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


# =============================================================================
# LOGISTIC REGRESSION (Level 2 Prediction)
# =============================================================================

@dataclass
class LogisticRegressionResult:
    """Container for logistic regression prediction results."""
    symbol: str
    prediction: str  # "UP" or "DOWN"
    probability_up: float
    coefficients: dict[str, float]
    feature_contributions: list[dict]
    accuracy: float
    n_training_samples: int


class LogisticRegressionAnalyzer:
    """
    Logistic regression for binary market direction prediction.

    Uses sklearn's LogisticRegression for interpretable UP/DOWN predictions.
    Each coefficient tells a clear story about which event features
    push the market up or down.
    """

    def __init__(self):
        self.feature_names: list[str] = []

    def train_and_predict(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        current_features: dict[str, float],
    ) -> Optional[LogisticRegressionResult]:
        """
        Train logistic regression on historical data, then predict
        for the given current event features.

        Args:
            symbol: Market symbol to predict
            start_date: Start of training period
            end_date: End of training period
            current_features: Dict of feature values for prediction
                e.g. {"goldstein_mean": -5.0, "mentions_total": 100, ...}
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler

        from src.analysis.feature_engineering import FeatureEngineering

        fe = FeatureEngineering()
        X, y, feature_names = fe.prepare_classification_features(
            symbol, start_date, end_date
        )

        if len(X) < 30:
            logger.warning(f"Insufficient data for {symbol}: {len(X)} samples")
            return None

        self.feature_names = feature_names

        # Scale features for better convergence
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train logistic regression
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_scaled, y)

        # Cross-validated accuracy
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")
        accuracy = float(cv_scores.mean())

        # Prepare current features for prediction
        feature_vector = np.array([
            current_features.get(name, 0.0) for name in feature_names
        ]).reshape(1, -1)
        feature_vector_scaled = scaler.transform(feature_vector)

        # Predict
        prob = model.predict_proba(feature_vector_scaled)[0]
        prob_up = float(prob[1]) if len(prob) > 1 else float(prob[0])
        prediction = "UP" if prob_up >= 0.5 else "DOWN"

        # Extract coefficients
        coefficients = dict(zip(feature_names, model.coef_[0]))

        # Compute per-feature contribution (coefficient * scaled feature value)
        scaled_values = feature_vector_scaled[0]
        contributions = []
        for i, name in enumerate(feature_names):
            contrib = float(model.coef_[0][i] * scaled_values[i])
            contributions.append({
                "feature": name,
                "coefficient": float(model.coef_[0][i]),
                "value": current_features.get(name, 0.0),
                "contribution": contrib,
            })

        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)

        return LogisticRegressionResult(
            symbol=symbol,
            prediction=prediction,
            probability_up=prob_up,
            coefficients=coefficients,
            feature_contributions=contributions,
            accuracy=accuracy,
            n_training_samples=len(X),
        )

    def get_model_summary(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> Optional[dict]:
        """
        Get model fit statistics without making a prediction.

        Returns coefficients, accuracy, and feature importance for display.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler

        from src.analysis.feature_engineering import FeatureEngineering

        fe = FeatureEngineering()
        X, y, feature_names = fe.prepare_classification_features(
            symbol, start_date, end_date
        )

        if len(X) < 30:
            return None

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_scaled, y)

        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")

        coefficients = dict(zip(feature_names, model.coef_[0].tolist()))
        abs_coefs = {k: abs(v) for k, v in coefficients.items()}

        return {
            "symbol": symbol,
            "coefficients": coefficients,
            "feature_importance": dict(
                sorted(abs_coefs.items(), key=lambda x: x[1], reverse=True)
            ),
            "intercept": float(model.intercept_[0]),
            "accuracy": float(cv_scores.mean()),
            "accuracy_std": float(cv_scores.std()),
            "n_training_samples": len(X),
            "up_ratio": float(y.mean()),
        }
