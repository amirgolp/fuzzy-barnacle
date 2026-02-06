"""Base backtest engine interface."""

from abc import ABC, abstractmethod

import pandas as pd

from quantdash.core.models import BacktestResult


class BaseBacktestEngine(ABC):
    """Abstract base for backtest engines."""

    @abstractmethod
    def run(
        self,
        df: pd.DataFrame,
        strategy_id: str,
        params: dict | None = None,
        initial_cash: float = 100_000.0,
        fee_bps: int = 10,
        slippage_bps: int = 5,
    ) -> BacktestResult:
        """
        Run a backtest on OHLCV data using a registered strategy.

        Args:
            df: OHLCV DataFrame with datetime index
            strategy_id: Registered strategy identifier
            params: Strategy parameter overrides
            initial_cash: Starting capital
            fee_bps: Trading fee in basis points
            slippage_bps: Slippage in basis points

        Returns:
            BacktestResult with metrics, equity curve, and trades
        """
        ...
