"""Base strategy interface."""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class Strategy(ABC):
    """
    Abstract base for all trading strategies.

    Subclasses must implement generate_signals() which returns a DataFrame
    with a 'signal' column: 1 = buy, -1 = sell, 0 = hold.

    All strategies support optional take_profit_pct, stop_loss_pct, and
    trailing_stop_pct parameters. When set, apply_tp_sl() modifies exit
    signals accordingly.
    """

    name: str = "base"
    description: str = ""
    default_params: dict[str, Any] = {}

    def __init__(self, params: dict[str, Any] | None = None):
        merged = {**self.default_params, **(params or {})}
        # Extract TP/SL/holding params from the merged dict
        self.take_profit_pct: float | None = merged.pop("take_profit_pct", None)
        self.stop_loss_pct: float | None = merged.pop("stop_loss_pct", None)
        self.trailing_stop_pct: float | None = merged.pop("trailing_stop_pct", None)
        # Max holding period in bars (e.g., 5 for intraweek on daily, 20 for intramonth)
        self.max_holding_bars: int | None = merged.pop("max_holding_bars", None)
        self.params = merged

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from OHLCV data.

        Args:
            df: OHLCV DataFrame with lowercase columns

        Returns:
            DataFrame with at least a 'signal' column (1=buy, -1=sell, 0=hold)
        """
        ...

    def apply_tp_sl(self, df: pd.DataFrame, signals_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply take profit, stop loss, trailing stop, and max holding period to signals.

        Walks through bars sequentially. When in a position:
        - Stop loss: exit if price drops stop_loss_pct% below entry
        - Take profit: exit if price rises take_profit_pct% above entry
        - Trailing stop: exit if price drops trailing_stop_pct% from peak since entry
        - Max holding: force exit after max_holding_bars bars in position

        Returns modified signals DataFrame with additional tp_level/sl_level columns.
        """
        tp = self.take_profit_pct
        sl = self.stop_loss_pct
        ts = self.trailing_stop_pct
        max_bars = self.max_holding_bars

        if tp is None and sl is None and ts is None and max_bars is None:
            return signals_df

        close = df["close"].values
        signal = signals_df["signal"].values.copy()
        n = len(close)

        in_position = False
        entry_price = 0.0
        peak_price = 0.0
        bars_held = 0

        for i in range(n):
            if signal[i] == 1 and not in_position:
                in_position = True
                entry_price = close[i]
                peak_price = close[i]
                bars_held = 0
            elif in_position:
                bars_held += 1
                peak_price = max(peak_price, close[i])

                # Max holding period - force exit
                if max_bars is not None and bars_held >= max_bars:
                    signal[i] = -1
                    in_position = False
                    continue

                # Stop loss
                if sl is not None and close[i] <= entry_price * (1 - sl / 100):
                    signal[i] = -1
                    in_position = False
                    continue

                # Take profit
                if tp is not None and close[i] >= entry_price * (1 + tp / 100):
                    signal[i] = -1
                    in_position = False
                    continue

                # Trailing stop
                if ts is not None and close[i] <= peak_price * (1 - ts / 100):
                    signal[i] = -1
                    in_position = False
                    continue

                # Strategy-generated sell
                if signal[i] == -1:
                    in_position = False

        result = signals_df.copy()
        result["signal"] = signal
        return result

    @property
    def has_tp_sl(self) -> bool:
        """Whether any TP/SL/trailing stop/max holding is configured."""
        return any([
            self.take_profit_pct,
            self.stop_loss_pct,
            self.trailing_stop_pct,
            self.max_holding_bars,
        ])

    def get_info(self) -> dict[str, Any]:
        """Return strategy metadata."""
        return {
            "name": self.name,
            "description": self.description,
            "params": self.params,
            "default_params": self.default_params,
            "take_profit_pct": self.take_profit_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "trailing_stop_pct": self.trailing_stop_pct,
            "max_holding_bars": self.max_holding_bars,
        }
