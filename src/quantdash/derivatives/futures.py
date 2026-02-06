"""
Futures pricing and analysis module.

Provides:
- Fair value calculation for futures
- Basis and roll yield analysis
- Contango/backwardation detection
- Cost of carry model
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class FuturesAnalysis:
    """Container for futures analysis results."""
    fair_value: float           # Theoretical fair value
    basis: float                # Spot - Futures price
    basis_pct: float            # Basis as percentage of spot
    annualized_basis: float     # Annualized basis (carry)
    market_structure: Literal["contango", "backwardation", "flat"]
    implied_rate: Optional[float]  # Implied interest rate from basis
    roll_yield: Optional[float]    # Estimated roll yield


def fair_value_future(
    spot: float,
    days_to_expiry: int,
    risk_free_rate: float,
    storage_cost: float = 0.0,
    convenience_yield: float = 0.0,
    dividend_yield: float = 0.0,
) -> float:
    """
    Calculate fair value of a futures contract using cost-of-carry model.

    F = S * e^((r + u - y - q) * T)

    Where:
        F = Futures price
        S = Spot price
        r = Risk-free rate
        u = Storage cost rate
        y = Convenience yield
        q = Dividend yield (for equity futures)
        T = Time to expiry in years

    Args:
        spot: Current spot price
        days_to_expiry: Days until futures expiry
        risk_free_rate: Annualized risk-free rate (decimal)
        storage_cost: Annualized storage cost rate (decimal, for commodities)
        convenience_yield: Annualized convenience yield (decimal, for commodities)
        dividend_yield: Annualized dividend yield (decimal, for equity index futures)

    Returns:
        Fair value of futures contract
    """
    T = days_to_expiry / 365.0
    carry_cost = risk_free_rate + storage_cost - convenience_yield - dividend_yield
    return spot * math.exp(carry_cost * T)


def calculate_basis(
    spot: float,
    futures_price: float,
    days_to_expiry: int,
) -> FuturesAnalysis:
    """
    Calculate basis and analyze futures market structure.

    Args:
        spot: Current spot price
        futures_price: Current futures price
        days_to_expiry: Days until futures expiry

    Returns:
        FuturesAnalysis with basis metrics and market structure
    """
    T = days_to_expiry / 365.0 if days_to_expiry > 0 else 0.001

    basis = spot - futures_price
    basis_pct = (basis / spot) * 100 if spot != 0 else 0

    # Annualized basis (simple annualization)
    annualized_basis = (basis / spot) * (365 / days_to_expiry) * 100 if days_to_expiry > 0 else 0

    # Market structure
    if abs(basis_pct) < 0.1:
        structure: Literal["contango", "backwardation", "flat"] = "flat"
    elif futures_price > spot:
        structure = "contango"
    else:
        structure = "backwardation"

    # Implied interest rate from basis
    # F = S * e^(r * T) => r = ln(F/S) / T
    implied_rate = None
    if futures_price > 0 and spot > 0 and T > 0:
        implied_rate = math.log(futures_price / spot) / T

    return FuturesAnalysis(
        fair_value=futures_price,  # Using market price as fair value
        basis=basis,
        basis_pct=basis_pct,
        annualized_basis=annualized_basis,
        market_structure=structure,
        implied_rate=implied_rate,
        roll_yield=None,
    )


def calculate_roll_yield(
    front_month_price: float,
    back_month_price: float,
    front_days_to_expiry: int,
    back_days_to_expiry: int,
) -> dict:
    """
    Calculate roll yield between two futures contracts.

    Roll yield represents the gain/loss from rolling a futures position
    from the front month to the back month contract.

    Args:
        front_month_price: Price of front month contract
        back_month_price: Price of back month contract
        front_days_to_expiry: Days to expiry for front month
        back_days_to_expiry: Days to expiry for back month

    Returns:
        Dict with roll yield metrics
    """
    if back_month_price <= 0 or front_month_price <= 0:
        return {
            "roll_yield": 0,
            "annualized_roll_yield": 0,
            "structure": "unknown",
        }

    days_between = back_days_to_expiry - front_days_to_expiry
    if days_between <= 0:
        days_between = 30  # Default to 30 days

    # Roll yield = (Front - Back) / Front
    roll_yield = (front_month_price - back_month_price) / front_month_price

    # Annualized roll yield
    annualized = roll_yield * (365 / days_between)

    # Structure
    if abs(roll_yield) < 0.001:
        structure = "flat"
    elif front_month_price > back_month_price:
        structure = "backwardation"  # Positive roll yield
    else:
        structure = "contango"  # Negative roll yield

    return {
        "roll_yield": roll_yield * 100,  # As percentage
        "annualized_roll_yield": annualized * 100,  # As percentage
        "structure": structure,
        "front_price": front_month_price,
        "back_price": back_month_price,
        "days_between": days_between,
    }


class FuturesPricer:
    """
    High-level futures pricing and analysis interface.

    Example:
        pricer = FuturesPricer(spot=100, risk_free_rate=0.05)
        fv = pricer.fair_value(expiry_days=30)
        analysis = pricer.analyze(futures_price=100.5, expiry_days=30)
    """

    # Common futures contract specifications
    CONTRACTS = {
        "/ES": {"multiplier": 50, "tick_size": 0.25, "name": "E-mini S&P 500", "type": "equity_index"},
        "/NQ": {"multiplier": 20, "tick_size": 0.25, "name": "E-mini Nasdaq 100", "type": "equity_index"},
        "/YM": {"multiplier": 5, "tick_size": 1.0, "name": "E-mini Dow", "type": "equity_index"},
        "/RTY": {"multiplier": 50, "tick_size": 0.10, "name": "E-mini Russell 2000", "type": "equity_index"},
        "/GC": {"multiplier": 100, "tick_size": 0.10, "name": "Gold", "type": "precious_metal"},
        "/SI": {"multiplier": 5000, "tick_size": 0.005, "name": "Silver", "type": "precious_metal"},
        "/CL": {"multiplier": 1000, "tick_size": 0.01, "name": "Crude Oil", "type": "energy"},
        "/NG": {"multiplier": 10000, "tick_size": 0.001, "name": "Natural Gas", "type": "energy"},
        "/ZC": {"multiplier": 50, "tick_size": 0.25, "name": "Corn", "type": "agricultural"},
        "/ZW": {"multiplier": 50, "tick_size": 0.25, "name": "Wheat", "type": "agricultural"},
        "/ZS": {"multiplier": 50, "tick_size": 0.25, "name": "Soybeans", "type": "agricultural"},
        "/6E": {"multiplier": 125000, "tick_size": 0.00005, "name": "Euro FX", "type": "currency"},
        "/6J": {"multiplier": 12500000, "tick_size": 0.0000005, "name": "Japanese Yen", "type": "currency"},
        "/ZB": {"multiplier": 1000, "tick_size": 1/32, "name": "30-Year T-Bond", "type": "interest_rate"},
        "/ZN": {"multiplier": 1000, "tick_size": 1/64, "name": "10-Year T-Note", "type": "interest_rate"},
    }

    def __init__(
        self,
        spot: float,
        risk_free_rate: float = 0.05,
        storage_cost: float = 0.0,
        convenience_yield: float = 0.0,
        dividend_yield: float = 0.0,
    ):
        """
        Initialize futures pricer.

        Args:
            spot: Current spot price of underlying
            risk_free_rate: Annualized risk-free rate
            storage_cost: Annualized storage cost (commodities)
            convenience_yield: Annualized convenience yield (commodities)
            dividend_yield: Annualized dividend yield (equity indices)
        """
        self.spot = spot
        self.risk_free_rate = risk_free_rate
        self.storage_cost = storage_cost
        self.convenience_yield = convenience_yield
        self.dividend_yield = dividend_yield

    def fair_value(self, expiry_days: int) -> float:
        """Calculate fair value for a futures contract."""
        return fair_value_future(
            self.spot,
            expiry_days,
            self.risk_free_rate,
            self.storage_cost,
            self.convenience_yield,
            self.dividend_yield,
        )

    def analyze(self, futures_price: float, expiry_days: int) -> FuturesAnalysis:
        """
        Analyze a futures contract vs spot.

        Args:
            futures_price: Current futures market price
            expiry_days: Days to contract expiry

        Returns:
            FuturesAnalysis with basis, structure, and implied rates
        """
        analysis = calculate_basis(self.spot, futures_price, expiry_days)

        # Override fair_value with our calculated value
        analysis.fair_value = self.fair_value(expiry_days)

        return analysis

    def term_structure(
        self,
        contracts: list[tuple[float, int]],
    ) -> list[dict]:
        """
        Analyze futures term structure from multiple contracts.

        Args:
            contracts: List of (price, days_to_expiry) tuples

        Returns:
            List of dicts with analysis for each contract
        """
        sorted_contracts = sorted(contracts, key=lambda x: x[1])

        results = []
        for i, (price, days) in enumerate(sorted_contracts):
            analysis = self.analyze(price, days)

            result = {
                "expiry_days": days,
                "price": price,
                "fair_value": round(analysis.fair_value, 4),
                "basis": round(analysis.basis, 4),
                "basis_pct": round(analysis.basis_pct, 4),
                "annualized_basis": round(analysis.annualized_basis, 4),
                "structure": analysis.market_structure,
            }

            # Calculate roll yield if not the last contract
            if i < len(sorted_contracts) - 1:
                next_price, next_days = sorted_contracts[i + 1]
                roll = calculate_roll_yield(price, next_price, days, next_days)
                result["roll_yield_to_next"] = round(roll["roll_yield"], 4)
                result["annualized_roll_yield"] = round(roll["annualized_roll_yield"], 4)

            results.append(result)

        return results

    def contract_value(
        self,
        symbol: str,
        price: float,
    ) -> Optional[dict]:
        """
        Calculate notional value and tick value for a futures contract.

        Args:
            symbol: Futures symbol (e.g., "/ES", "/GC")
            price: Current futures price

        Returns:
            Dict with notional value, tick value, and contract specs
        """
        if symbol not in self.CONTRACTS:
            return None

        spec = self.CONTRACTS[symbol]
        notional = price * spec["multiplier"]
        tick_value = spec["tick_size"] * spec["multiplier"]

        return {
            "symbol": symbol,
            "name": spec["name"],
            "type": spec["type"],
            "price": price,
            "multiplier": spec["multiplier"],
            "tick_size": spec["tick_size"],
            "notional_value": notional,
            "tick_value": tick_value,
        }

    @classmethod
    def get_contract_info(cls, symbol: str) -> Optional[dict]:
        """Get static contract specifications."""
        return cls.CONTRACTS.get(symbol)

    @classmethod
    def list_contracts(cls) -> list[str]:
        """List all available contract symbols."""
        return list(cls.CONTRACTS.keys())
