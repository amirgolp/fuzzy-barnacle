"""Vectorbt-based backtest engine."""

import pandas as pd
import vectorbt as vbt

from quantdash.core.models import BacktestResult
from quantdash.strategies.registry import get_strategy

from .base import BaseBacktestEngine
from .metrics import extract_metrics


class VectorbtEngine(BaseBacktestEngine):
    """Backtest engine using vectorbt."""

    def run(
        self,
        df: pd.DataFrame,
        strategy_id: str,
        params: dict | None = None,
        initial_cash: float = 100_000.0,
        fee_bps: int = 10,
        slippage_bps: int = 5,
    ) -> BacktestResult:
        strategy = get_strategy(strategy_id, params)
        signals_df = strategy.generate_signals(df)

        # Apply TP/SL/trailing stop if configured
        if strategy.has_tp_sl:
            signals_df = strategy.apply_tp_sl(df, signals_df)

        signal = signals_df["signal"]
        entries = signal == 1
        exits = signal == -1

        fees = fee_bps / 10_000
        slippage = slippage_bps / 10_000

        pf = vbt.Portfolio.from_signals(
            close=df["close"],
            entries=entries,
            exits=exits,
            init_cash=initial_cash,
            fees=fees,
            slippage=slippage,
            freq="1D",
        )

        symbol = df.attrs.get("symbol", "UNKNOWN") if hasattr(df, "attrs") else "UNKNOWN"
        return extract_metrics(pf, strategy_id=strategy_id, symbol=symbol)
