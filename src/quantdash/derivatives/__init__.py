"""
Derivatives pricing and analysis module.

Provides:
- Black-Scholes option pricing
- Greeks calculations (Delta, Gamma, Theta, Vega, Rho)
- Implied volatility calculation
- Put-call parity checks
- Basic IV surface modeling
"""

from .options import (
    black_scholes_call,
    black_scholes_put,
    calculate_greeks,
    implied_volatility,
    OptionPricer,
    OptionGreeks,
)

from .futures import (
    FuturesPricer,
    calculate_basis,
    calculate_roll_yield,
    fair_value_future,
)

__all__ = [
    # Options
    "black_scholes_call",
    "black_scholes_put",
    "calculate_greeks",
    "implied_volatility",
    "OptionPricer",
    "OptionGreeks",
    # Futures
    "FuturesPricer",
    "calculate_basis",
    "calculate_roll_yield",
    "fair_value_future",
]
