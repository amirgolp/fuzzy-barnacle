"""Features routes for indicators and patterns."""

from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from quantdash.data import YFinanceProvider, get_symbol_data
from quantdash.features import (
    INDICATORS_CONFIG,
    PATTERNS_CONFIG,
    compute_indicator,
    detect_all_patterns,
)
from quantdash.data.screener_cache import get_screener_cache

router = APIRouter()
provider = YFinanceProvider()


class IndicatorInfo(BaseModel):
    """Indicator metadata."""
    name: str
    category: str
    type: str
    description: str
    params: dict[str, Any]
    color: str


class IndicatorRequest(BaseModel):
    """Request for computing an indicator."""
    symbol: str
    timeframe: str = "1d"
    start: datetime
    end: datetime
    indicator: str
    params: Optional[dict[str, Any]] = None


class PatternInfo(BaseModel):
    """Detected pattern information."""
    pattern_type: str
    start_index: int
    end_index: int
    confidence: float
    direction: str


@router.get("/indicators")
async def list_indicators() -> list[IndicatorInfo]:
    """
    List all available technical indicators.
    """
    return [
        IndicatorInfo(
            name=name,
            category=config["category"],
            type=config["type"],
            description=config["desc"],
            params=config["params"],
            color=config.get("color", "#2962FF"),
        )
        for name, config in INDICATORS_CONFIG.items()
    ]


@router.get("/indicators/{category}")
async def list_indicators_by_category(category: str) -> list[IndicatorInfo]:
    """
    List indicators filtered by category.
    
    Categories: Trend, Momentum, Volatility, Volume
    """
    valid_categories = ["Trend", "Momentum", "Volatility", "Volume"]
    if category not in valid_categories:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid category. Must be one of: {valid_categories}"
        )
    
    return [
        IndicatorInfo(
            name=name,
            category=config["category"],
            type=config["type"],
            description=config["desc"],
            params=config["params"],
            color=config.get("color", "#2962FF"),
        )
        for name, config in INDICATORS_CONFIG.items()
        if config["category"] == category
    ]


@router.post("/compute")
async def compute_indicator_endpoint(request: IndicatorRequest) -> dict:
    """
    Compute a technical indicator for a symbol.
    
    Returns the indicator values as a time series.
    """
    if request.indicator not in INDICATORS_CONFIG:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown indicator: {request.indicator}"
        )
    
    try:
        # Fetch data
        df = provider.download_bars(
            [request.symbol], 
            request.timeframe, 
            request.start, 
            request.end
        )
        df = get_symbol_data(df, request.symbol)
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No data found")
        
        # Compute indicator
        result = compute_indicator(df, request.indicator, request.params)
        
        # Format response
        if hasattr(result, "to_dict"):
            # Single series
            values = {
                k.isoformat() if hasattr(k, "isoformat") else str(k): v 
                for k, v in result.dropna().to_dict().items()
            }
            return {
                "indicator": request.indicator,
                "symbol": request.symbol,
                "values": values,
            }
        elif isinstance(result, dict):
            # Multi-output indicator
            outputs = {}
            for key, series in result.items():
                if hasattr(series, "to_dict"):
                    outputs[key] = {
                        k.isoformat() if hasattr(k, "isoformat") else str(k): v 
                        for k, v in series.dropna().to_dict().items()
                    }
            return {
                "indicator": request.indicator,
                "symbol": request.symbol,
                "outputs": outputs,
            }
        else:
            return {
                "indicator": request.indicator,
                "symbol": request.symbol,
                "result": result,
            }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns")
async def list_patterns() -> list[dict]:
    """
    List all available chart patterns.
    """
    return [
        {
            "name": name,
            "direction": config["direction"],
            "category": config["category"],
            "description": config["desc"],
        }
        for name, config in PATTERNS_CONFIG.items()
    ]


@router.get("/patterns/detect")
async def detect_patterns(
    symbol: str = Query(..., description="Symbol to analyze"),
    timeframe: str = Query("1d", description="Timeframe"),
    start: datetime = Query(..., description="Start datetime"),
    end: datetime = Query(..., description="End datetime"),
) -> list[PatternInfo]:
    """
    Detect chart patterns for a symbol.
    """
    try:
        df = provider.download_bars([symbol], timeframe, start, end)
        df = get_symbol_data(df, symbol)
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No data found")
        
        patterns = detect_all_patterns(df)
        
        return [
            PatternInfo(
                pattern_type=p.pattern_type,
                start_index=p.start_index,
                end_index=p.end_index,
                confidence=p.confidence,
                direction=p.direction.value,
            )
            for p in patterns
        ]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/screener/analyze")
async def screen_symbol_endpoint(
    symbol: str = Query(..., description="Symbol to screen"),
    timeframe: str = Query("1d", description="Timeframe"),
    start: Optional[datetime] = Query(None, description="Start datetime (defaults to 1 year ago)"),
    end: Optional[datetime] = Query(None, description="End datetime (defaults to now)"),
) -> dict:
    """
    Run technical screener on a symbol.
    """
    try:
        from datetime import timedelta
        
        # Default dates if not provided
        if not end:
            end = datetime.now()
        if not start:
            start = end - timedelta(days=365)
            
        df = provider.download_bars([symbol], timeframe, start, end)
        df = get_symbol_data(df, symbol)
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No data found")
            
        cache = get_screener_cache()
        cached = cache.get_result("symbol", symbol=symbol, timeframe=timeframe)
        if cached:
            cached["from_cache"] = True
            return cached

        from quantdash.features import screen_symbol
        from quantdash.features.screener_models import ScreenerResult
        
        result = screen_symbol(df, symbol=symbol)
        data = result.model_dump()
        
        cache.store_result(data, "symbol", symbol=symbol, timeframe=timeframe)
        data["from_cache"] = False
        return data

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/watchlist")
async def get_watchlist_technicals(
    symbols: str = Query("AAPL,MSFT,GOOGL,AMZN,TSLA,NVDA,META,JPM,V,SPY", description="Comma-separated symbols"),
    timeframe: str = Query("1d", description="Timeframe"),
):
    """Compute technicals summary for multiple symbols (watchlist)."""
    try:
        from datetime import timedelta

        from quantdash.features.technicals import compute_technicals

        end = datetime.now()
        start = end - timedelta(days=400)
        symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]

        results = []
        for sym in symbol_list:
            try:
                df = provider.download_bars([sym], timeframe, start, end)
                df = get_symbol_data(df, sym)
                if df.empty:
                    continue
                tech = compute_technicals(df, sym)
                results.append(tech.model_dump())
            except Exception:
                continue

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class PortfolioScreenRequest(BaseModel):
    """Request for portfolio screening."""
    symbols: list[str]
    timeframe: str = "1d"


@router.post("/screener/portfolio")
async def screen_portfolio_endpoint(request: PortfolioScreenRequest) -> dict:
    """
    Screen multiple symbols and rank by signal strength.
    """
    try:
        from datetime import timedelta

        from quantdash.features import screen_portfolio

        end = datetime.now()
        start = end - timedelta(days=365)

        # Check cache (all symbols together as one key for simplicity, or we could cache individually)
        # For portfolio, the request usually changes, but if it matches exactly, we can return.
        # However, portfolio screening is heavy so caching per request makes sense.
        cache = get_screener_cache()
        # Sort symbols for stable key
        symbols_key = ",".join(sorted(request.symbols))
        cached = cache.get_result("portfolio", symbols=symbols_key, timeframe=request.timeframe)
        if cached:
            cached["from_cache"] = True
            return cached

        portfolio_data: dict = {}
        for symbol in request.symbols:
            try:
                df = provider.download_bars([symbol], request.timeframe, start, end)
                df = get_symbol_data(df, symbol)
                if not df.empty:
                    portfolio_data[symbol] = df
            except Exception:
                continue

        if not portfolio_data:
            raise HTTPException(status_code=404, detail="No data found for any symbol")

        result = screen_portfolio(portfolio_data)
        data = result.model_dump()
        
        cache.store_result(data, "portfolio", symbols=symbols_key, timeframe=request.timeframe)
        data["from_cache"] = False
        return data

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/technicals")
async def get_technicals(
    symbol: str = Query(..., description="Symbol to analyze"),
    timeframe: str = Query("1d", description="Timeframe"),
):
    """TradingView-style technical analysis with Oscillators, Moving Averages, and Summary gauges."""
    try:
        from datetime import timedelta

        from quantdash.features.technicals import compute_technicals

        end = datetime.now()
        start = end - timedelta(days=400)

        df = provider.download_bars([symbol], timeframe, start, end)
        df = get_symbol_data(df, symbol)

        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")

        result = compute_technicals(df, symbol)
        return result.model_dump()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
