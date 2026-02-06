"""
Options pricing module implementing Black-Scholes model and Greeks.

This module provides:
- Black-Scholes option pricing (calls and puts)
- All first-order Greeks (Delta, Gamma, Theta, Vega, Rho)
- Implied volatility calculation via Newton-Raphson
- Put-call parity validation
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional
from scipy.stats import norm
from scipy.optimize import brentq


@dataclass
class OptionGreeks:
    """Container for option Greeks."""
    delta: float      # Rate of change of option price w.r.t. spot price
    gamma: float      # Rate of change of delta w.r.t. spot price
    theta: float      # Time decay (per day)
    vega: float       # Sensitivity to volatility (per 1% move)
    rho: float        # Sensitivity to interest rate (per 1% move)

    # Second-order Greeks (optional)
    vanna: Optional[float] = None   # d(delta)/d(vol)
    charm: Optional[float] = None   # d(delta)/d(time)
    vomma: Optional[float] = None   # d(vega)/d(vol)


def _d1(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Calculate d1 in Black-Scholes formula."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))


def _d2(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Calculate d2 in Black-Scholes formula."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return _d1(S, K, T, r, q, sigma) - sigma * math.sqrt(T)


def black_scholes_call(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> float:
    """
    Calculate Black-Scholes price for a European call option.

    Args:
        S: Current spot price of underlying
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (annualized, decimal)
        sigma: Volatility (annualized, decimal)
        q: Dividend yield (annualized, decimal)

    Returns:
        Call option price
    """
    if T <= 0:
        return max(0, S - K)
    if sigma <= 0:
        # No volatility - option is worth intrinsic value discounted
        return max(0, S * math.exp(-q * T) - K * math.exp(-r * T))

    d1 = _d1(S, K, T, r, q, sigma)
    d2 = _d2(S, K, T, r, q, sigma)

    call_price = S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return call_price


def black_scholes_put(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
) -> float:
    """
    Calculate Black-Scholes price for a European put option.

    Args:
        S: Current spot price of underlying
        K: Strike price
        T: Time to expiration in years
        r: Risk-free interest rate (annualized, decimal)
        sigma: Volatility (annualized, decimal)
        q: Dividend yield (annualized, decimal)

    Returns:
        Put option price
    """
    if T <= 0:
        return max(0, K - S)
    if sigma <= 0:
        return max(0, K * math.exp(-r * T) - S * math.exp(-q * T))

    d1 = _d1(S, K, T, r, q, sigma)
    d2 = _d2(S, K, T, r, q, sigma)

    put_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)
    return put_price


def calculate_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"],
    q: float = 0.0,
    include_second_order: bool = False,
) -> OptionGreeks:
    """
    Calculate all Greeks for an option.

    Args:
        S: Current spot price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free rate (annualized, decimal)
        sigma: Volatility (annualized, decimal)
        option_type: "call" or "put"
        q: Dividend yield (annualized, decimal)
        include_second_order: Whether to calculate second-order Greeks

    Returns:
        OptionGreeks dataclass with all calculated Greeks
    """
    if T <= 0 or sigma <= 0:
        # At expiration or no vol, Greeks are edge cases
        intrinsic = max(0, S - K) if option_type == "call" else max(0, K - S)
        itm = (S > K) if option_type == "call" else (S < K)
        return OptionGreeks(
            delta=1.0 if (option_type == "call" and itm) else (-1.0 if (option_type == "put" and itm) else 0.0),
            gamma=0.0,
            theta=0.0,
            vega=0.0,
            rho=0.0,
        )

    d1 = _d1(S, K, T, r, q, sigma)
    d2 = _d2(S, K, T, r, q, sigma)
    sqrt_T = math.sqrt(T)

    # Standard normal PDF at d1
    pdf_d1 = norm.pdf(d1)

    # Delta
    if option_type == "call":
        delta = math.exp(-q * T) * norm.cdf(d1)
    else:
        delta = -math.exp(-q * T) * norm.cdf(-d1)

    # Gamma (same for call and put)
    gamma = math.exp(-q * T) * pdf_d1 / (S * sigma * sqrt_T)

    # Theta (per year, we'll convert to per day)
    term1 = -(S * sigma * math.exp(-q * T) * pdf_d1) / (2 * sqrt_T)
    if option_type == "call":
        term2 = -r * K * math.exp(-r * T) * norm.cdf(d2)
        term3 = q * S * math.exp(-q * T) * norm.cdf(d1)
    else:
        term2 = r * K * math.exp(-r * T) * norm.cdf(-d2)
        term3 = -q * S * math.exp(-q * T) * norm.cdf(-d1)
    theta_annual = term1 + term2 + term3
    theta_daily = theta_annual / 365  # Convert to daily decay

    # Vega (per 1% move in vol)
    vega = S * math.exp(-q * T) * pdf_d1 * sqrt_T / 100

    # Rho (per 1% move in rate)
    if option_type == "call":
        rho = K * T * math.exp(-r * T) * norm.cdf(d2) / 100
    else:
        rho = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100

    greeks = OptionGreeks(
        delta=delta,
        gamma=gamma,
        theta=theta_daily,
        vega=vega,
        rho=rho,
    )

    # Second-order Greeks
    if include_second_order:
        # Vanna: d(delta)/d(vol) = d(vega)/d(S)
        greeks.vanna = -math.exp(-q * T) * pdf_d1 * d2 / sigma

        # Charm: d(delta)/d(t) (per day)
        charm_annual = q * math.exp(-q * T) * norm.cdf(d1 if option_type == "call" else -d1)
        charm_annual -= math.exp(-q * T) * pdf_d1 * (2 * (r - q) * T - d2 * sigma * sqrt_T) / (2 * T * sigma * sqrt_T)
        greeks.charm = charm_annual / 365

        # Vomma: d(vega)/d(vol)
        greeks.vomma = vega * d1 * d2 / sigma

    return greeks


def implied_volatility(
    option_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: Literal["call", "put"],
    q: float = 0.0,
    precision: float = 1e-6,
    max_iterations: int = 100,
) -> Optional[float]:
    """
    Calculate implied volatility using Brent's method.

    Args:
        option_price: Market price of the option
        S: Current spot price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free rate
        option_type: "call" or "put"
        q: Dividend yield
        precision: Desired precision for IV
        max_iterations: Maximum iterations

    Returns:
        Implied volatility as decimal, or None if not found
    """
    if T <= 0:
        return None

    # Intrinsic value bounds
    if option_type == "call":
        intrinsic = max(0, S * math.exp(-q * T) - K * math.exp(-r * T))
    else:
        intrinsic = max(0, K * math.exp(-r * T) - S * math.exp(-q * T))

    if option_price < intrinsic:
        return None  # Price below intrinsic - no valid IV

    pricing_func = black_scholes_call if option_type == "call" else black_scholes_put

    def objective(sigma: float) -> float:
        return pricing_func(S, K, T, r, sigma, q) - option_price

    try:
        # Use Brent's method for robust root finding
        iv = brentq(objective, 0.001, 5.0, xtol=precision, maxiter=max_iterations)
        return iv
    except ValueError:
        # No root found in the interval
        return None


class OptionPricer:
    """
    High-level option pricing interface.

    Example:
        pricer = OptionPricer(spot=100, rate=0.05, dividend=0.02)
        price = pricer.price_call(strike=105, expiry_days=30, vol=0.25)
        greeks = pricer.greeks_call(strike=105, expiry_days=30, vol=0.25)
    """

    def __init__(
        self,
        spot: float,
        rate: float = 0.05,
        dividend: float = 0.0,
    ):
        """
        Initialize pricer with underlying parameters.

        Args:
            spot: Current spot price of underlying
            rate: Risk-free rate (annualized, decimal)
            dividend: Dividend yield (annualized, decimal)
        """
        self.spot = spot
        self.rate = rate
        self.dividend = dividend

    def _days_to_years(self, days: int) -> float:
        """Convert days to years (assuming 365 days/year)."""
        return days / 365.0

    def price_call(
        self,
        strike: float,
        expiry_days: int,
        vol: float,
    ) -> float:
        """Price a call option."""
        T = self._days_to_years(expiry_days)
        return black_scholes_call(self.spot, strike, T, self.rate, vol, self.dividend)

    def price_put(
        self,
        strike: float,
        expiry_days: int,
        vol: float,
    ) -> float:
        """Price a put option."""
        T = self._days_to_years(expiry_days)
        return black_scholes_put(self.spot, strike, T, self.rate, vol, self.dividend)

    def greeks_call(
        self,
        strike: float,
        expiry_days: int,
        vol: float,
        include_second_order: bool = False,
    ) -> OptionGreeks:
        """Calculate Greeks for a call option."""
        T = self._days_to_years(expiry_days)
        return calculate_greeks(
            self.spot, strike, T, self.rate, vol, "call", self.dividend, include_second_order
        )

    def greeks_put(
        self,
        strike: float,
        expiry_days: int,
        vol: float,
        include_second_order: bool = False,
    ) -> OptionGreeks:
        """Calculate Greeks for a put option."""
        T = self._days_to_years(expiry_days)
        return calculate_greeks(
            self.spot, strike, T, self.rate, vol, "put", self.dividend, include_second_order
        )

    def implied_vol_call(
        self,
        strike: float,
        expiry_days: int,
        market_price: float,
    ) -> Optional[float]:
        """Calculate implied volatility for a call option."""
        T = self._days_to_years(expiry_days)
        return implied_volatility(
            market_price, self.spot, strike, T, self.rate, "call", self.dividend
        )

    def implied_vol_put(
        self,
        strike: float,
        expiry_days: int,
        market_price: float,
    ) -> Optional[float]:
        """Calculate implied volatility for a put option."""
        T = self._days_to_years(expiry_days)
        return implied_volatility(
            market_price, self.spot, strike, T, self.rate, "put", self.dividend
        )

    def put_call_parity_check(
        self,
        call_price: float,
        put_price: float,
        strike: float,
        expiry_days: int,
        tolerance: float = 0.01,
    ) -> dict:
        """
        Check put-call parity: C - P = S*e^(-qT) - K*e^(-rT)

        Returns dict with expected difference, actual difference, and whether parity holds.
        """
        T = self._days_to_years(expiry_days)
        expected_diff = self.spot * math.exp(-self.dividend * T) - strike * math.exp(-self.rate * T)
        actual_diff = call_price - put_price
        deviation = abs(actual_diff - expected_diff)

        return {
            "expected_diff": expected_diff,
            "actual_diff": actual_diff,
            "deviation": deviation,
            "parity_holds": deviation <= tolerance * self.spot,
        }

    def price_chain(
        self,
        strikes: list[float],
        expiry_days: int,
        vol: float,
    ) -> list[dict]:
        """
        Price a full option chain for given strikes.

        Returns list of dicts with strike, call_price, put_price, and Greeks.
        """
        T = self._days_to_years(expiry_days)
        chain = []

        for K in strikes:
            call_price = black_scholes_call(self.spot, K, T, self.rate, vol, self.dividend)
            put_price = black_scholes_put(self.spot, K, T, self.rate, vol, self.dividend)
            call_greeks = calculate_greeks(self.spot, K, T, self.rate, vol, "call", self.dividend)
            put_greeks = calculate_greeks(self.spot, K, T, self.rate, vol, "put", self.dividend)

            chain.append({
                "strike": K,
                "call_price": round(call_price, 4),
                "put_price": round(put_price, 4),
                "call_delta": round(call_greeks.delta, 4),
                "put_delta": round(put_greeks.delta, 4),
                "gamma": round(call_greeks.gamma, 6),
                "call_theta": round(call_greeks.theta, 4),
                "put_theta": round(put_greeks.theta, 4),
                "vega": round(call_greeks.vega, 4),
            })

        return chain
