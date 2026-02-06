"""Risk parity portfolio optimization using scipy."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quantdash.core.models import PortfolioOptimizationResult


def optimize_risk_parity(
    returns_df: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> PortfolioOptimizationResult:
    """
    Optimize portfolio for equal risk contribution (risk parity).

    Args:
        returns_df: DataFrame of daily returns, columns are asset names
        risk_free_rate: Annual risk-free rate for Sharpe calculation

    Returns:
        PortfolioOptimizationResult with risk-parity weights
    """
    symbols = list(returns_df.columns)
    n = len(symbols)
    mu = returns_df.mean().values * 252
    cov = returns_df.cov().values * 252

    def risk_parity_objective(w):
        port_vol = np.sqrt(w @ cov @ w)
        if port_vol == 0:
            return 0.0
        marginal = cov @ w
        rc = w * marginal / port_vol
        target = port_vol / n
        return np.sum((rc - target) ** 2)

    w0 = np.ones(n) / n
    bounds = [(0.01, 1.0)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    result = minimize(
        risk_parity_objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    weights = result.x / result.x.sum()

    port_return = float(mu @ weights)
    port_vol = float(np.sqrt(weights @ cov @ weights))
    sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0.0

    marginal = cov @ weights
    rc = weights * marginal
    total = rc.sum()
    rc_pct = rc / total if total > 0 else rc

    return PortfolioOptimizationResult(
        method="risk_parity",
        weights={s: round(float(w), 6) for s, w in zip(symbols, weights)},
        risk_contributions={s: round(float(r), 6) for s, r in zip(symbols, rc_pct)},
        expected_return=round(port_return, 6),
        portfolio_volatility=round(port_vol, 6),
        sharpe_ratio=round(sharpe, 4),
    )
