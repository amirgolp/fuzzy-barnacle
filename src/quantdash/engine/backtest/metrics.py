"""Extract backtest metrics from vectorbt Portfolio."""

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from quantdash.core.models import BacktestResult, TradeRecord


def sanitize_float(value: float, default: float = 0.0) -> float:
    """Convert NaN, inf, or invalid float values to a safe default."""
    if pd.isna(value) or np.isinf(value):
        return default
    return float(value)


def extract_metrics(
    pf,  # vbt.Portfolio
    strategy_id: str = "",
    symbol: str = "",
) -> BacktestResult:
    """Extract BacktestResult from a vectorbt Portfolio object."""
    stats = pf.stats()

    # Equity curve
    equity = pf.value()
    equity_curve = []
    for ts, val in equity.items():
        equity_curve.append({
            "time": ts.isoformat() if isinstance(ts, (datetime, pd.Timestamp)) else str(ts),
            "value": sanitize_float(val),
        })

    # Drawdown curve
    dd = pf.drawdown()
    drawdown_curve = []
    for ts, val in dd.items():
        drawdown_curve.append({
            "time": ts.isoformat() if isinstance(ts, (datetime, pd.Timestamp)) else str(ts),
            "value": sanitize_float(val),
        })

    # Trade records
    trades = _extract_trades(pf, symbol)

    total_return = sanitize_float(stats.get("Total Return [%]", 0)) / 100
    max_dd = abs(sanitize_float(stats.get("Max Drawdown [%]", 0)) / 100)
    sharpe = sanitize_float(stats.get("Sharpe Ratio", 0))
    sortino = sanitize_float(stats.get("Sortino Ratio", 0))
    calmar = sanitize_float(stats.get("Calmar Ratio", 0))
    total_trades = int(sanitize_float(stats.get("Total Trades", 0)))
    win_rate = sanitize_float(stats.get("Win Rate [%]", 0)) / 100 if total_trades > 0 else 0.0

    # CAGR approximation
    n_days = (equity.index[-1] - equity.index[0]).days if len(equity) > 1 else 1
    n_years = max(n_days / 365.25, 1 / 365.25)
    try:
        cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1 if equity.iloc[0] > 0 else 0.0
        cagr = sanitize_float(cagr)
    except (ZeroDivisionError, ValueError, OverflowError):
        cagr = 0.0

    # Volatility
    returns = equity.pct_change().dropna()
    try:
        volatility = sanitize_float(returns.std() * np.sqrt(252)) if len(returns) > 1 else 0.0
    except (ValueError, OverflowError):
        volatility = 0.0

    # Exposure
    exposure = sanitize_float(stats.get("Exposure Time [%]", 0)) / 100

    start_date = equity.index[0].to_pydatetime() if len(equity) > 0 else None
    end_date = equity.index[-1].to_pydatetime() if len(equity) > 0 else None

    return BacktestResult(
        cagr=round(cagr, 6),
        total_return=round(total_return, 6),
        volatility=round(volatility, 6),
        sharpe_ratio=round(sharpe, 4),
        sortino_ratio=round(sortino, 4),
        calmar_ratio=round(calmar, 4),
        max_drawdown=round(max_dd, 6),
        win_rate=round(win_rate, 4),
        total_trades=total_trades,
        exposure=round(exposure, 4),
        equity_curve=equity_curve,
        drawdown_curve=drawdown_curve,
        trades=trades,
        strategy_id=strategy_id,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
    )


def _extract_trades(pf, symbol: str) -> list[TradeRecord]:
    """Extract trade records from portfolio."""
    trades = []
    try:
        trade_records = pf.trades.records_readable
        for _, row in trade_records.iterrows():
            trades.append(TradeRecord(
                entry_date=pd.Timestamp(row.get("Entry Timestamp", row.get("Entry Index"))),
                exit_date=pd.Timestamp(row.get("Exit Timestamp", row.get("Exit Index"))) if pd.notna(row.get("Exit Timestamp", row.get("Exit Index"))) else None,
                symbol=symbol,
                side="long",
                entry_price=sanitize_float(row.get("Avg Entry Price", 0)),
                exit_price=sanitize_float(row.get("Avg Exit Price", 0)) if pd.notna(row.get("Avg Exit Price")) else None,
                quantity=sanitize_float(row.get("Size", 0)),
                pnl=sanitize_float(row.get("PnL", 0)),
                return_pct=sanitize_float(row.get("Return", 0)),
            ))
    except Exception:
        pass
    return trades
