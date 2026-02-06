"""Mean-variance portfolio optimization using cvxpy."""

import numpy as np
import pandas as pd

from quantdash.core.models import PortfolioOptimizationResult


def optimize_mean_variance(
    returns_df: pd.DataFrame,
    risk_aversion: float = 1.0,
    risk_free_rate: float = 0.0,
) -> PortfolioOptimizationResult:
    """
    Optimize portfolio weights using mean-variance optimization.

    Args:
        returns_df: DataFrame of daily returns, columns are asset names
        risk_aversion: Risk aversion parameter (higher = more conservative)
        risk_free_rate: Annual risk-free rate for Sharpe calculation

    Returns:
        PortfolioOptimizationResult with optimal weights and metrics
    """
    import cvxpy as cp

    symbols = list(returns_df.columns)
    n = len(symbols)
    mu = returns_df.mean().values * 252  # Annualized
    cov = returns_df.cov().values * 252

    w = cp.Variable(n)
    ret = mu @ w
    risk = cp.quad_form(w, cov)

    objective = cp.Maximize(ret - risk_aversion * risk)
    constraints = [
        cp.sum(w) == 1,
        w >= 0,  # Long-only
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS)

    weights = np.clip(w.value, 0, 1)
    weights = weights / weights.sum()  # Normalize

    port_return = float(mu @ weights)
    port_vol = float(np.sqrt(weights @ cov @ weights))
    sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0.0

    # Risk contributions
    marginal_risk = cov @ weights
    risk_contribs = weights * marginal_risk
    total_risk = risk_contribs.sum()
    rc_pct = risk_contribs / total_risk if total_risk > 0 else risk_contribs

    return PortfolioOptimizationResult(
        method="mean_variance",
        weights={s: round(float(w), 6) for s, w in zip(symbols, weights)},
        risk_contributions={s: round(float(r), 6) for s, r in zip(symbols, rc_pct)},
        expected_return=round(port_return, 6),
        portfolio_volatility=round(port_vol, 6),
        sharpe_ratio=round(sharpe, 4),
    )
