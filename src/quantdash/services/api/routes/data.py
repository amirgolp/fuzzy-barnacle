"""Data routes for OHLCV data."""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from quantdash.data import cached_provider, get_symbol_data
from quantdash.config import get_logger

router = APIRouter()
logger = get_logger("api.data")

# Use cached provider for smart data fetching
provider = cached_provider


class BarsRequest(BaseModel):
    """Request model for bars endpoint."""
    symbols: list[str]
    timeframe: str = "1d"
    start: datetime
    end: datetime
    adjusted: bool = True


class BarsResponse(BaseModel):
    """Response model for bars endpoint."""
    symbol: str
    timeframe: str
    count: int
    data: list[dict]


@router.get("/bars")
async def get_bars(
    symbol: str = Query(..., description="Symbol to fetch"),
    timeframe: str = Query("1d", description="Timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1wk, 1mo)"),
    start: datetime = Query(..., description="Start datetime"),
    end: datetime = Query(..., description="End datetime"),
    adjusted: bool = Query(True, description="Use adjusted prices"),
) -> BarsResponse:
    """
    Get OHLCV bars for a symbol.
    
    Returns candlestick data with open, high, low, close, volume.
    """
    try:
        df = provider.download_bars([symbol], timeframe, start, end, adjusted)
        df = get_symbol_data(df, symbol)
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        # Convert to list of dicts
        data = []
        for idx, row in df.iterrows():
            data.append({
                "time": idx.isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]) if "volume" in row else 0,
            })
        
        return BarsResponse(
            symbol=symbol,
            timeframe=timeframe,
            count=len(data),
            data=data,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bars")
async def post_bars(request: BarsRequest) -> list[BarsResponse]:
    """
    Get OHLCV bars for multiple symbols.
    """
    results = []
    
    for symbol in request.symbols:
        try:
            df = provider.download_bars(
                [symbol], 
                request.timeframe, 
                request.start, 
                request.end, 
                request.adjusted
            )
            df = get_symbol_data(df, symbol)
            
            data = []
            for idx, row in df.iterrows():
                data.append({
                    "time": idx.isoformat(),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]) if "volume" in row else 0,
                })
            
            results.append(BarsResponse(
                symbol=symbol,
                timeframe=request.timeframe,
                count=len(data),
                data=data,
            ))
        except Exception:
            # Skip failed symbols
            continue
    
    return results
