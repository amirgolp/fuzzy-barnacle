"""Chandelier Exit strategy (ported from TradingView Pine Script v6)."""

import numpy as np
import pandas as pd

from quantdash.strategies.base import Strategy
from quantdash.strategies.registry import register_strategy


class ChandelierExit(Strategy):
    """
    Chandelier Exit strategy by Alex Orekhov.

    Uses ATR-based trailing stops to generate buy/sell signals.
    Buy when price crosses above the short stop (direction flips to 1).
    Sell when price crosses below the long stop (direction flips to -1).
    """

    name = "chandelier_exit"
    description = "Chandelier Exit (ATR trailing stop) strategy"
    default_params = {"atr_period": 22, "atr_multiplier": 3.0, "use_close": True}

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        period = self.params["atr_period"]
        mult = self.params["atr_multiplier"]
        use_close = self.params["use_close"]

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        n = len(close)

        # ATR calculation
        tr = np.zeros(n)
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )
        tr[0] = high[0] - low[0]

        atr = np.zeros(n)
        atr[:period] = np.nan
        if n >= period:
            atr[period - 1] = np.mean(tr[:period])
            for i in range(period, n):
                atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

        atr_scaled = mult * atr

        # Highest/lowest over period
        long_stop = np.full(n, np.nan)
        short_stop = np.full(n, np.nan)
        direction = np.zeros(n, dtype=int)

        for i in range(period - 1, n):
            start = max(0, i - period + 1)
            if use_close:
                highest = np.max(close[start:i + 1])
                lowest = np.min(close[start:i + 1])
            else:
                highest = np.max(high[start:i + 1])
                lowest = np.min(low[start:i + 1])

            raw_long = highest - atr_scaled[i]
            raw_short = lowest + atr_scaled[i]

            # Ratchet long stop up (never lower if previous close was above prev long stop)
            if i > period - 1 and not np.isnan(long_stop[i - 1]):
                if close[i - 1] > long_stop[i - 1]:
                    long_stop[i] = max(raw_long, long_stop[i - 1])
                else:
                    long_stop[i] = raw_long
            else:
                long_stop[i] = raw_long

            # Ratchet short stop down
            if i > period - 1 and not np.isnan(short_stop[i - 1]):
                if close[i - 1] < short_stop[i - 1]:
                    short_stop[i] = min(raw_short, short_stop[i - 1])
                else:
                    short_stop[i] = raw_short
            else:
                short_stop[i] = raw_short

            # Direction
            if i == period - 1:
                direction[i] = 1
            else:
                prev_short = short_stop[i - 1] if not np.isnan(short_stop[i - 1]) else raw_short
                prev_long = long_stop[i - 1] if not np.isnan(long_stop[i - 1]) else raw_long
                if close[i] > prev_short:
                    direction[i] = 1
                elif close[i] < prev_long:
                    direction[i] = -1
                else:
                    direction[i] = direction[i - 1]

        # Signals: direction change
        signal = np.zeros(n, dtype=int)
        for i in range(1, n):
            if direction[i] == 1 and direction[i - 1] == -1:
                signal[i] = 1   # Buy
            elif direction[i] == -1 and direction[i - 1] == 1:
                signal[i] = -1  # Sell

        result = df.copy()
        result["signal"] = signal
        result["ce_long_stop"] = np.where(direction == 1, long_stop, np.nan)
        result["ce_short_stop"] = np.where(direction == -1, short_stop, np.nan)
        result["ce_direction"] = direction

        return result


register_strategy("chandelier_exit", ChandelierExit)
