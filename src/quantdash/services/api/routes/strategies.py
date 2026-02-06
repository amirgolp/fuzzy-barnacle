"""Strategy, backtest, and portfolio optimization API routes."""

from datetime import datetime
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from quantdash.core.models import BacktestResult, PortfolioOptimizationResult
from quantdash.data import YFinanceProvider
from quantdash.engine.backtest.vectorbt_engine import VectorbtEngine
from quantdash.portfolio import optimize_mean_variance, optimize_risk_parity
from quantdash.strategies.registry import get_strategy, list_strategies

router = APIRouter()
provider = YFinanceProvider()
engine = VectorbtEngine()


# --- Request models ---


class BacktestRequest(BaseModel):
    symbol: str
    strategy_id: str
    params: dict[str, Any] = Field(default_factory=dict)
    timeframe: str = "1d"
    start: Optional[str] = None
    end: Optional[str] = None
    initial_cash: float = 100_000.0
    fee_bps: int = 10
    slippage_bps: int = 5


class OptimizeRequest(BaseModel):
    symbols: list[str]
    method: str = "mean_variance"  # or "risk_parity"
    timeframe: str = "1d"
    start: Optional[str] = None
    end: Optional[str] = None
    risk_aversion: float = 1.0


# --- Strategy routes ---


@router.get("/")
async def get_strategies():
    """List all registered strategies."""
    return list_strategies()


@router.get("/{strategy_id}")
async def get_strategy_info(strategy_id: str):
    """Get strategy details."""
    try:
        s = get_strategy(strategy_id)
        info = s.get_info()
        info["id"] = strategy_id
        return info
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


# --- Backtest route ---


@router.post("/backtest", response_model=BacktestResult)
async def run_backtest(req: BacktestRequest):
    """Run a backtest for a strategy on a symbol."""
    try:
        # Parse dates
        end_dt = datetime.fromisoformat(req.end) if req.end else datetime.now()
        start_dt = datetime.fromisoformat(req.start) if req.start else datetime(end_dt.year - 2, end_dt.month, end_dt.day)

        # Download data
        df = provider.download_bars([req.symbol], req.timeframe, start_dt, end_dt)
        from quantdash.data import get_symbol_data
        df = get_symbol_data(df, req.symbol)

        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {req.symbol}")

        df.attrs["symbol"] = req.symbol

        result = engine.run(
            df=df,
            strategy_id=req.strategy_id,
            params=req.params if req.params else None,
            initial_cash=req.initial_cash,
            fee_bps=req.fee_bps,
            slippage_bps=req.slippage_bps,
        )
        result.symbol = req.symbol
        return result

    except HTTPException:
        raise
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Strategy overlay route ---


class OverlayRequest(BaseModel):
    symbol: str
    strategy_id: str
    params: dict[str, Any] = Field(default_factory=dict)
    timeframe: str = "1d"
    start: Optional[str] = None
    end: Optional[str] = None


@router.post("/overlay")
async def get_strategy_overlay(req: OverlayRequest):
    """Run a strategy and return overlay lines + buy/sell markers for chart rendering."""
    try:
        end_dt = datetime.fromisoformat(req.end) if req.end else datetime.now()
        start_dt = datetime.fromisoformat(req.start) if req.start else datetime(end_dt.year - 2, end_dt.month, end_dt.day)

        from quantdash.data import get_symbol_data
        df = provider.download_bars([req.symbol], req.timeframe, start_dt, end_dt)
        df = get_symbol_data(df, req.symbol)

        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {req.symbol}")

        strategy = get_strategy(req.strategy_id, req.params if req.params else None)
        signals_df = strategy.generate_signals(df)

        # Extract overlay lines (strategy-specific columns)
        skip_cols = set(df.columns) | {"signal"}
        lines: dict[str, list] = {}
        for col in signals_df.columns:
            if col in skip_cols:
                continue
            series = signals_df[col]
            if hasattr(series, "dtype") and series.dtype in ("float64", "float32", "int64", "int32"):
                data = []
                for idx, val in series.items():
                    if pd.notna(val) and val != 0:
                        ts = int(idx.timestamp()) if hasattr(idx, "timestamp") else 0
                        data.append({"time": ts, "value": float(val)})
                if data:
                    lines[col] = data

        # Extract buy/sell markers with TP/SL and freshness
        markers = []
        signal_col = signals_df["signal"]
        close_col = df["close"]
        has_sl = "sl_level" in signals_df.columns
        has_tp = "tp_level" in signals_df.columns
        last_idx = len(signals_df) - 1

        for i, (idx, sig) in enumerate(signal_col.items()):
            ts = int(idx.timestamp()) if hasattr(idx, "timestamp") else 0
            if sig == 1:
                marker: dict = {"time": ts, "position": "belowBar", "color": "#26a69a", "shape": "arrowUp", "text": "Buy"}
            elif sig == -1:
                marker = {"time": ts, "position": "aboveBar", "color": "#ef5350", "shape": "arrowDown", "text": "Sell"}
            else:
                continue
            entry = float(close_col.iloc[i]) if i < len(close_col) else None
            if entry is not None:
                marker["entry_price"] = round(entry, 4)
            if has_sl and pd.notna(signals_df["sl_level"].iloc[i]):
                marker["sl"] = round(float(signals_df["sl_level"].iloc[i]), 4)
            if has_tp and pd.notna(signals_df["tp_level"].iloc[i]):
                marker["tp"] = round(float(signals_df["tp_level"].iloc[i]), 4)
            marker["bars_ago"] = last_idx - i
            markers.append(marker)

        # Assign colors to lines
        line_colors = {
            "ce_long_stop": "#26a69a", "ce_short_stop": "#ef5350",
            "sqz_momentum": "#2962FF", "sqz_on": "#FF6D00", "sqz_off": "#26a69a",
        }

        line_configs = []
        for name, data in lines.items():
            if name.endswith("_direction"):
                continue  # Skip integer direction columns
            color = line_colors.get(name, "#2962FF")
            line_configs.append({"name": name, "color": color, "data": data})

        return {"strategy_id": req.strategy_id, "lines": line_configs, "markers": markers}

    except HTTPException:
        raise
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Portfolio optimization route ---


@router.post("/optimize", response_model=PortfolioOptimizationResult)
async def optimize_portfolio(req: OptimizeRequest):
    """Run portfolio optimization."""
    try:
        end_dt = datetime.fromisoformat(req.end) if req.end else datetime.now()
        start_dt = datetime.fromisoformat(req.start) if req.start else datetime(end_dt.year - 2, end_dt.month, end_dt.day)

        # Download all symbols
        from quantdash.data import get_symbol_data
        returns_dict = {}
        for sym in req.symbols:
            df = provider.download_bars([sym], req.timeframe, start_dt, end_dt)
            df = get_symbol_data(df, sym)
            if not df.empty:
                returns_dict[sym] = df["close"].pct_change().dropna()

        if len(returns_dict) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 symbols with data")

        returns_df = pd.DataFrame(returns_dict).dropna()

        if req.method == "risk_parity":
            return optimize_risk_parity(returns_df)
        else:
            return optimize_mean_variance(returns_df, risk_aversion=req.risk_aversion)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
