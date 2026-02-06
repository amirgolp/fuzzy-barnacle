"""Circuit breaker risk controls.

Monitors portfolio state and triggers protective actions:
- Daily loss > 3% → flatten all positions
- Drawdown > threshold → reduce max leverage to 1×
- Cross-asset correlation spike > 0.9 → reduce total exposure 50%
"""

from __future__ import annotations

import logging

from quantdash.ml.config import AssetRiskProfile, CircuitBreakerState

logger = logging.getLogger(__name__)


def check_daily_loss(
    state: CircuitBreakerState,
    max_daily_loss_pct: float = 3.0,
) -> bool:
    """Check if daily loss exceeds threshold.

    Returns True if circuit breaker triggered.
    """
    if abs(state.daily_pnl_pct) > max_daily_loss_pct and state.daily_pnl_pct < 0:
        if "daily_loss" not in state.triggered:
            state.triggered.append("daily_loss")
            logger.warning(
                "CIRCUIT BREAKER: Daily loss %.2f%% exceeds %.1f%% threshold. "
                "Flatten all positions.",
                state.daily_pnl_pct, max_daily_loss_pct,
            )
        return True
    return False


def check_drawdown(
    state: CircuitBreakerState,
    risk_profile: AssetRiskProfile,
) -> bool:
    """Check if drawdown exceeds asset-specific threshold.

    Returns True if circuit breaker triggered.
    """
    if state.drawdown_pct > risk_profile.max_drawdown_pct:
        if "drawdown" not in state.triggered:
            state.triggered.append("drawdown")
            logger.warning(
                "CIRCUIT BREAKER: Drawdown %.2f%% exceeds %.1f%% threshold. "
                "Reducing leverage to 1×.",
                state.drawdown_pct, risk_profile.max_drawdown_pct,
            )
        return True
    return False


def check_correlation_spike(
    correlation_matrix: dict[str, dict[str, float]] | None,
    threshold: float = 0.9,
) -> bool:
    """Check if cross-asset correlations are dangerously high.

    High correlation means diversification breaks down — reduce exposure.

    Args:
        correlation_matrix: Pairwise correlations {sym1: {sym2: corr}}.
        threshold: Correlation threshold.

    Returns True if any pair exceeds threshold.
    """
    if correlation_matrix is None:
        return False

    for sym1, correlations in correlation_matrix.items():
        for sym2, corr in correlations.items():
            if sym1 != sym2 and abs(corr) > threshold:
                logger.warning(
                    "CIRCUIT BREAKER: Correlation spike %.3f between %s and %s. "
                    "Reduce exposure 50%%.",
                    corr, sym1, sym2,
                )
                return True
    return False


def update_circuit_breaker_state(
    state: CircuitBreakerState,
    bar_pnl: float,
    current_equity: float,
) -> CircuitBreakerState:
    """Update circuit breaker state with latest bar data.

    Args:
        state: Current state.
        bar_pnl: PnL from the latest bar (as percentage).
        current_equity: Current portfolio equity.

    Returns:
        Updated state.
    """
    state.daily_pnl_pct += bar_pnl
    state.current_equity = current_equity

    if current_equity > state.peak_equity:
        state.peak_equity = current_equity

    if state.peak_equity > 0:
        state.drawdown_pct = (
            (state.peak_equity - current_equity) / state.peak_equity * 100
        )
    else:
        state.drawdown_pct = 0.0

    return state


def get_effective_leverage(
    state: CircuitBreakerState,
    risk_profile: AssetRiskProfile,
    max_daily_loss_pct: float = 3.0,
) -> float:
    """Get effective max leverage after circuit breaker checks.

    Returns 0 if positions should be flattened, reduced leverage if
    drawdown triggered, or normal max_leverage if no breakers triggered.
    """
    if check_daily_loss(state, max_daily_loss_pct):
        return 0.0  # flatten

    if check_drawdown(state, risk_profile):
        return 1.0  # reduce to no leverage

    return risk_profile.max_leverage
