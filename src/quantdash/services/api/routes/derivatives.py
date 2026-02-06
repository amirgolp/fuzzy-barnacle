"""
API routes for derivatives pricing and analysis.

Provides endpoints for:
- Black-Scholes option pricing
- Greeks calculation
- Implied volatility calculation
- Futures fair value and basis analysis
"""

from typing import Literal, Optional
from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from quantdash.derivatives.options import (
    black_scholes_call,
    black_scholes_put,
    calculate_greeks,
    implied_volatility,
    OptionPricer,
)
from quantdash.derivatives.futures import (
    FuturesPricer,
    fair_value_future,
    calculate_basis,
    calculate_roll_yield,
)

router = APIRouter(prefix="/derivatives", tags=["derivatives"])


# ─── Request/Response Models ──────────────────────────────────────────────────


class OptionPriceRequest(BaseModel):
    """Request model for option pricing."""
    spot: float = Field(..., description="Current spot price of underlying")
    strike: float = Field(..., description="Strike price")
    expiry_days: int = Field(..., ge=0, description="Days to expiration")
    volatility: float = Field(..., gt=0, le=5, description="Annualized volatility (decimal, e.g. 0.25 for 25%)")
    rate: float = Field(0.05, description="Risk-free rate (decimal)")
    dividend: float = Field(0.0, ge=0, description="Dividend yield (decimal)")
    option_type: Literal["call", "put"] = Field("call", description="Option type")


class OptionPriceResponse(BaseModel):
    """Response model for option pricing."""
    price: float
    intrinsic: float
    time_value: float
    option_type: str
    spot: float
    strike: float
    expiry_days: int
    volatility: float


class GreeksRequest(BaseModel):
    """Request model for Greeks calculation."""
    spot: float
    strike: float
    expiry_days: int
    volatility: float
    rate: float = 0.05
    dividend: float = 0.0
    option_type: Literal["call", "put"] = "call"
    include_second_order: bool = False


class GreeksResponse(BaseModel):
    """Response model for Greeks."""
    delta: float
    gamma: float
    theta: float  # Per day
    vega: float   # Per 1% vol move
    rho: float    # Per 1% rate move
    vanna: Optional[float] = None
    charm: Optional[float] = None
    vomma: Optional[float] = None


class IVRequest(BaseModel):
    """Request model for implied volatility calculation."""
    market_price: float = Field(..., gt=0, description="Market price of option")
    spot: float
    strike: float
    expiry_days: int
    rate: float = 0.05
    dividend: float = 0.0
    option_type: Literal["call", "put"] = "call"


class ChainRequest(BaseModel):
    """Request model for option chain pricing."""
    spot: float
    expiry_days: int
    volatility: float
    rate: float = 0.05
    dividend: float = 0.0
    strikes: list[float] = Field(default_factory=list, description="List of strikes (auto-generated if empty)")
    strike_step: float = Field(5.0, description="Strike step for auto-generation")
    num_strikes: int = Field(11, ge=3, le=51, description="Number of strikes to generate")


class FuturesFairValueRequest(BaseModel):
    """Request model for futures fair value."""
    spot: float
    expiry_days: int
    rate: float = 0.05
    storage_cost: float = 0.0
    convenience_yield: float = 0.0
    dividend_yield: float = 0.0


class FuturesBasisRequest(BaseModel):
    """Request model for futures basis analysis."""
    spot: float
    futures_price: float
    expiry_days: int


class RollYieldRequest(BaseModel):
    """Request model for roll yield calculation."""
    front_price: float
    back_price: float
    front_days: int
    back_days: int


class TermStructureRequest(BaseModel):
    """Request model for term structure analysis."""
    spot: float
    contracts: list[tuple[float, int]] = Field(..., description="List of (price, days_to_expiry) tuples")
    rate: float = 0.05
    storage_cost: float = 0.0
    convenience_yield: float = 0.0
    dividend_yield: float = 0.0


# ─── Options Endpoints ────────────────────────────────────────────────────────


@router.post("/options/price", response_model=OptionPriceResponse)
async def price_option(req: OptionPriceRequest):
    """
    Calculate Black-Scholes option price.

    Returns the theoretical price along with intrinsic and time value breakdown.
    """
    T = req.expiry_days / 365.0

    if req.option_type == "call":
        price = black_scholes_call(req.spot, req.strike, T, req.rate, req.volatility, req.dividend)
        intrinsic = max(0, req.spot - req.strike)
    else:
        price = black_scholes_put(req.spot, req.strike, T, req.rate, req.volatility, req.dividend)
        intrinsic = max(0, req.strike - req.spot)

    time_value = price - intrinsic

    return OptionPriceResponse(
        price=round(price, 4),
        intrinsic=round(intrinsic, 4),
        time_value=round(time_value, 4),
        option_type=req.option_type,
        spot=req.spot,
        strike=req.strike,
        expiry_days=req.expiry_days,
        volatility=req.volatility,
    )


@router.post("/options/greeks", response_model=GreeksResponse)
async def get_greeks(req: GreeksRequest):
    """
    Calculate option Greeks.

    Returns Delta, Gamma, Theta, Vega, Rho, and optionally second-order Greeks.
    """
    T = req.expiry_days / 365.0
    greeks = calculate_greeks(
        req.spot, req.strike, T, req.rate, req.volatility,
        req.option_type, req.dividend, req.include_second_order
    )

    return GreeksResponse(
        delta=round(greeks.delta, 4),
        gamma=round(greeks.gamma, 6),
        theta=round(greeks.theta, 4),
        vega=round(greeks.vega, 4),
        rho=round(greeks.rho, 4),
        vanna=round(greeks.vanna, 6) if greeks.vanna else None,
        charm=round(greeks.charm, 6) if greeks.charm else None,
        vomma=round(greeks.vomma, 4) if greeks.vomma else None,
    )


@router.post("/options/iv")
async def calculate_iv(req: IVRequest):
    """
    Calculate implied volatility from market price.

    Uses Newton-Raphson method for root finding.
    """
    T = req.expiry_days / 365.0
    iv = implied_volatility(
        req.market_price, req.spot, req.strike, T, req.rate,
        req.option_type, req.dividend
    )

    if iv is None:
        return {
            "iv": None,
            "error": "Could not calculate IV - price may be below intrinsic or parameters invalid"
        }

    return {
        "iv": round(iv, 4),
        "iv_pct": round(iv * 100, 2),
        "market_price": req.market_price,
        "option_type": req.option_type,
    }


@router.post("/options/chain")
async def price_chain(req: ChainRequest):
    """
    Price a full option chain.

    Returns prices and Greeks for multiple strikes.
    """
    pricer = OptionPricer(req.spot, req.rate, req.dividend)

    # Generate strikes if not provided
    strikes = req.strikes
    if not strikes:
        center = round(req.spot / req.strike_step) * req.strike_step
        half = req.num_strikes // 2
        strikes = [center + i * req.strike_step for i in range(-half, half + 1)]

    chain = pricer.price_chain(strikes, req.expiry_days, req.volatility)

    return {
        "spot": req.spot,
        "expiry_days": req.expiry_days,
        "volatility": req.volatility,
        "rate": req.rate,
        "dividend": req.dividend,
        "chain": chain,
    }


@router.get("/options/quick-price")
async def quick_price(
    spot: float = Query(..., description="Spot price"),
    strike: float = Query(..., description="Strike price"),
    days: int = Query(..., ge=0, description="Days to expiry"),
    vol: float = Query(..., gt=0, description="Volatility (decimal)"),
    option_type: Literal["call", "put"] = Query("call"),
    rate: float = Query(0.05),
):
    """Quick option pricing via GET for simple calculations."""
    T = days / 365.0
    if option_type == "call":
        price = black_scholes_call(spot, strike, T, rate, vol)
    else:
        price = black_scholes_put(spot, strike, T, rate, vol)

    return {"price": round(price, 4), "option_type": option_type}


# ─── Futures Endpoints ────────────────────────────────────────────────────────


@router.post("/futures/fair-value")
async def futures_fair_value(req: FuturesFairValueRequest):
    """
    Calculate theoretical fair value for a futures contract.

    Uses cost-of-carry model: F = S * e^((r + u - y - q) * T)
    """
    fv = fair_value_future(
        req.spot,
        req.expiry_days,
        req.rate,
        req.storage_cost,
        req.convenience_yield,
        req.dividend_yield,
    )

    return {
        "fair_value": round(fv, 4),
        "spot": req.spot,
        "expiry_days": req.expiry_days,
        "carry_cost": round(req.rate + req.storage_cost - req.convenience_yield - req.dividend_yield, 4),
    }


@router.post("/futures/basis")
async def analyze_basis(req: FuturesBasisRequest):
    """
    Analyze futures basis (spot - futures).

    Returns basis metrics and market structure (contango/backwardation).
    """
    analysis = calculate_basis(req.spot, req.futures_price, req.expiry_days)

    return {
        "spot": req.spot,
        "futures_price": req.futures_price,
        "basis": round(analysis.basis, 4),
        "basis_pct": round(analysis.basis_pct, 4),
        "annualized_basis": round(analysis.annualized_basis, 4),
        "market_structure": analysis.market_structure,
        "implied_rate": round(analysis.implied_rate, 4) if analysis.implied_rate else None,
    }


@router.post("/futures/roll-yield")
async def get_roll_yield(req: RollYieldRequest):
    """
    Calculate roll yield between front and back month contracts.

    Positive roll yield = backwardation (favorable for long positions)
    Negative roll yield = contango (unfavorable for long positions)
    """
    result = calculate_roll_yield(
        req.front_price, req.back_price, req.front_days, req.back_days
    )
    return result


@router.post("/futures/term-structure")
async def analyze_term_structure(req: TermStructureRequest):
    """
    Analyze futures term structure across multiple contracts.

    Returns fair value, basis, and roll yield for each contract.
    """
    pricer = FuturesPricer(
        req.spot,
        req.rate,
        req.storage_cost,
        req.convenience_yield,
        req.dividend_yield,
    )

    results = pricer.term_structure(req.contracts)

    # Determine overall structure
    if len(results) >= 2:
        if all(r.get("structure") == "contango" for r in results):
            overall = "contango"
        elif all(r.get("structure") == "backwardation" for r in results):
            overall = "backwardation"
        else:
            overall = "mixed"
    else:
        overall = results[0]["structure"] if results else "unknown"

    return {
        "spot": req.spot,
        "overall_structure": overall,
        "contracts": results,
    }


@router.get("/futures/contracts")
async def list_futures_contracts():
    """List available futures contract specifications."""
    return {
        "contracts": [
            {"symbol": sym, **spec}
            for sym, spec in FuturesPricer.CONTRACTS.items()
        ]
    }


@router.get("/futures/contract/{symbol}")
async def get_contract_info(symbol: str):
    """Get specifications for a specific futures contract."""
    info = FuturesPricer.get_contract_info(symbol.upper())
    if not info:
        return {"error": f"Contract {symbol} not found", "available": FuturesPricer.list_contracts()}
    return {"symbol": symbol.upper(), **info}


@router.get("/futures/notional")
async def calculate_notional(
    symbol: str = Query(..., description="Futures symbol (e.g., /ES, /GC)"),
    price: float = Query(..., description="Current futures price"),
):
    """Calculate notional value and tick value for a futures position."""
    pricer = FuturesPricer(spot=price)
    result = pricer.contract_value(symbol.upper(), price)

    if not result:
        return {"error": f"Contract {symbol} not found", "available": FuturesPricer.list_contracts()}

    return result
