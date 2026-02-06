"""Symbols routes for symbol universe."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from quantdash.config import (
    AssetType,
    get_all_symbols,
    get_symbol_info,
    get_symbols_by_type,
    search_symbols,
)
from quantdash.data import YFinanceProvider

router = APIRouter()
provider = YFinanceProvider()


class SymbolInfo(BaseModel):
    """Symbol metadata."""
    symbol: str
    name: str
    asset_type: str
    exchange: str
    currency: str
    description: Optional[str] = None


class FundamentalsResponse(BaseModel):
    """Fundamental data for a symbol."""
    symbol: str
    pe_ratio: Optional[float] = None
    eps: Optional[float] = None
    pb_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    roe: Optional[float] = None
    debt_to_equity: Optional[float] = None
    free_cash_flow: Optional[float] = None
    peg_ratio: Optional[float] = None
    enterprise_value: Optional[float] = None
    beta: Optional[float] = None
    market_cap: Optional[float] = None


@router.get("")
async def list_all_symbols() -> list[SymbolInfo]:
    """
    Get all symbols in the universe.
    """
    return [
        SymbolInfo(
            symbol=s.symbol,
            name=s.name,
            asset_type=s.asset_type.value,
            exchange=s.exchange,
            currency=s.currency,
            description=s.description,
        )
        for s in get_all_symbols()
    ]


@router.get("/search")
async def search_symbols_endpoint(
    q: str = Query(..., description="Search query"),
    asset_type: Optional[str] = Query(None, description="Filter by asset type"),
) -> list[SymbolInfo]:
    """
    Search symbols by name or ticker.
    """
    # Convert asset type string to enum if provided
    type_filter = None
    if asset_type:
        try:
            type_filter = AssetType(asset_type.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid asset type. Must be one of: {[t.value for t in AssetType]}"
            )
    
    results = search_symbols(q, type_filter)
    
    return [
        SymbolInfo(
            symbol=s.symbol,
            name=s.name,
            asset_type=s.asset_type.value,
            exchange=s.exchange,
            currency=s.currency,
            description=s.description,
        )
        for s in results
    ]


@router.get("/type/{asset_type}")
async def get_symbols_by_type_endpoint(asset_type: str) -> list[SymbolInfo]:
    """
    Get symbols filtered by asset type.
    
    Types: stock, etf, futures, options, forex, crypto, bond
    """
    try:
        type_enum = AssetType(asset_type.lower())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid asset type. Must be one of: {[t.value for t in AssetType]}"
        )
    
    symbols = get_symbols_by_type(type_enum)
    
    return [
        SymbolInfo(
            symbol=s.symbol,
            name=s.name,
            asset_type=s.asset_type.value,
            exchange=s.exchange,
            currency=s.currency,
            description=s.description,
        )
        for s in symbols
    ]


@router.get("/{symbol}")
async def get_symbol(symbol: str) -> SymbolInfo:
    """
    Get metadata for a specific symbol.
    """
    info = get_symbol_info(symbol)
    
    if not info:
        raise HTTPException(status_code=404, detail=f"Symbol not found: {symbol}")
    
    return SymbolInfo(
        symbol=info.symbol,
        name=info.name,
        asset_type=info.asset_type.value,
        exchange=info.exchange,
        currency=info.currency,
        description=info.description,
    )


@router.get("/{symbol}/fundamentals")
async def get_fundamentals(symbol: str) -> FundamentalsResponse:
    """
    Get fundamental data for a symbol.
    
    Includes P/E, EPS, P/B, dividend yield, ROE, debt-to-equity,
    free cash flow, PEG, enterprise value, beta, and market cap.
    """
    data = provider.get_fundamentals(symbol)
    
    if not data:
        raise HTTPException(
            status_code=404, 
            detail=f"No fundamental data available for {symbol}"
        )
    
    return FundamentalsResponse(**data)
