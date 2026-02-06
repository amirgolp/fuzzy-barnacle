"""Strategy system for QuantDash."""

from .base import Strategy
from .registry import (
    STRATEGY_REGISTRY,
    get_strategy,
    list_strategies,
    register_strategy,
)

# Auto-register builtins on import
from .builtins import (  # noqa: F401
    adx_trend,
    atr_breakout,
    bollinger_reversion,
    breakout_trading,
    chandelier_exit,
    contango_fade,
    expected_move,
    fibonacci_pullback,
    gamma_scalp,
    gap_trading,
    heikin_ashi_momentum,
    ichimoku_cloud_strategy,
    iv_mean_reversion,
    macd_divergence,
    mean_reversion,
    ml_momentum,
    ml_momentum_v2,
    multi_agent_broker,
    multi_agent_broker_v2,
    order_flow_imbalance,
    pairs_trading,
    regime_detection,
    rsi_extremes,
    sma_crossover,
    squeeze_momentum,
    vwap_reversion,
)

__all__ = [
    "Strategy",
    "STRATEGY_REGISTRY",
    "register_strategy",
    "get_strategy",
    "list_strategies",
]
