"""MACD Divergence Strategy.

Detects bullish and bearish divergences between price and MACD.
Best for: Reversal trading, identifying trend exhaustion.
"""

import numpy as np
import pandas as pd

from quantdash.features.indicators import atr, macd
from quantdash.strategies.base import Strategy
from quantdash.strategies.registry import register_strategy


class MACDDivergenceStrategy(Strategy):
    """
    MACD divergence strategy for reversals.

    Regular divergence: Price makes new high/low but MACD doesn't (reversal signal)
    Hidden divergence: Price pullback but MACD continues trend (continuation signal)
    """

    name = "macd_divergence"
    description = "MACD divergence for reversals and continuations"
    default_params = {
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "divergence_lookback": 14,     # Bars to look back for divergence
        "min_divergence_bars": 5,      # Min distance between peaks
        "atr_period": 14,
        "sl_atr_multiplier": 2.0,
        "tp_atr_multiplier": 3.0,
        "trail_atr_multiplier": 1.5,
        "use_hidden_divergence": False, # Hidden divergence (continuation)
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate MACD divergence signals."""
        fast = self.params["macd_fast"]
        slow = self.params["macd_slow"]
        signal_period = self.params["macd_signal"]
        lookback = self.params["divergence_lookback"]
        min_bars = self.params["min_divergence_bars"]
        atr_period = self.params["atr_period"]

        result = df.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Calculate MACD
        macd_line, signal_line, histogram = macd(df, fast_period=fast, slow_period=slow, signal_period=signal_period)
        result["macd"] = macd_line
        result["macd_signal"] = signal_line
        result["macd_histogram"] = histogram

        # Calculate ATR
        atr_values = atr(df, period=atr_period)
        result["atr"] = atr_values

        # Find price peaks and troughs
        price_highs = high.rolling(min_bars, center=True).max() == high
        price_lows = low.rolling(min_bars, center=True).min() == low

        # Find MACD peaks and troughs
        macd_highs = macd_line.rolling(min_bars, center=True).max() == macd_line
        macd_lows = macd_line.rolling(min_bars, center=True).min() == macd_line

        # Detect regular bullish divergence
        # Price makes lower low, but MACD makes higher low
        bullish_div = self._detect_bullish_divergence(
            df, macd_line, price_lows, macd_lows, lookback
        )

        # Detect regular bearish divergence
        # Price makes higher high, but MACD makes lower high
        bearish_div = self._detect_bearish_divergence(
            df, macd_line, price_highs, macd_highs, lookback
        )

        result["bullish_divergence"] = bullish_div
        result["bearish_divergence"] = bearish_div

        # Generate signals
        result["signal"] = 0

        # Enter on divergence + MACD histogram confirmation
        long_signal = bullish_div & (histogram > 0)
        short_signal = bearish_div & (histogram < 0)

        result.loc[long_signal, "signal"] = 1
        result.loc[short_signal, "signal"] = -1

        # Exit on opposite MACD signal cross
        exit_long = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
        exit_short = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))

        result = self._apply_exits(result, exit_long, exit_short)

        # Calculate SL/TP levels
        sl_mult = self.params["sl_atr_multiplier"]
        tp_mult = self.params["tp_atr_multiplier"]

        result["sl_distance"] = atr_values * sl_mult
        result["tp_distance"] = atr_values * tp_mult

        result.loc[result["signal"] == 1, "sl_level"] = close - result["sl_distance"]
        result.loc[result["signal"] == 1, "tp_level"] = close + result["tp_distance"]

        result.loc[result["signal"] == -1, "sl_level"] = close + result["sl_distance"]
        result.loc[result["signal"] == -1, "tp_level"] = close - result["tp_distance"]

        # Apply trailing stop
        result = self._apply_trailing_stop(result, atr_values)

        # Clean up
        result["signal"] = result["signal"].fillna(0)
        for col in result.select_dtypes(include=[np.number]).columns:
            if col != "signal":
                result[col] = result[col].ffill().fillna(0)

        return result

    def _detect_bullish_divergence(self, df, macd_line, price_lows, macd_lows, lookback):
        """Detect bullish divergence: price lower low + MACD higher low."""
        divergence = pd.Series(False, index=df.index)
        low = df["low"].values
        macd_vals = macd_line.values

        for i in range(lookback, len(df)):
            if not price_lows.iloc[i]:
                continue

            # Find previous price low within lookback
            for j in range(i - lookback, i):
                if price_lows.iloc[j]:
                    # Check if price makes lower low but MACD makes higher low
                    if (low[i] < low[j]) and (macd_vals[i] > macd_vals[j]):
                        divergence.iloc[i] = True
                        break

        return divergence

    def _detect_bearish_divergence(self, df, macd_line, price_highs, macd_highs, lookback):
        """Detect bearish divergence: price higher high + MACD lower high."""
        divergence = pd.Series(False, index=df.index)
        high = df["high"].values
        macd_vals = macd_line.values

        for i in range(lookback, len(df)):
            if not price_highs.iloc[i]:
                continue

            # Find previous price high within lookback
            for j in range(i - lookback, i):
                if price_highs.iloc[j]:
                    # Check if price makes higher high but MACD makes lower high
                    if (high[i] > high[j]) and (macd_vals[i] < macd_vals[j]):
                        divergence.iloc[i] = True
                        break

        return divergence

    def _apply_exits(self, df: pd.DataFrame, exit_long: pd.Series, exit_short: pd.Series) -> pd.DataFrame:
        """Apply exit signals."""
        signals = df["signal"].values.copy()
        in_position = 0

        for i in range(len(signals)):
            if signals[i] == 1:
                in_position = 1
            elif signals[i] == -1:
                in_position = -1
            elif in_position == 1 and exit_long.iloc[i]:
                signals[i] = -1
                in_position = 0
            elif in_position == -1 and exit_short.iloc[i]:
                signals[i] = 1
                in_position = 0

        df = df.copy()
        df["signal"] = signals
        return df

    def _apply_trailing_stop(self, df: pd.DataFrame, atr_values: pd.Series) -> pd.DataFrame:
        """Apply trailing stop logic."""
        trail_mult = self.params["trail_atr_multiplier"]

        signals = df["signal"].values.copy()
        close = df["close"].values
        atr_vals = atr_values.values

        in_position = 0
        peak_price = 0.0

        for i in range(len(signals)):
            if signals[i] == 1 and in_position == 0:
                in_position = 1
                peak_price = close[i]
            elif signals[i] == -1 and in_position == 0:
                in_position = -1
                peak_price = close[i]
            elif in_position != 0:
                if in_position == 1:
                    if close[i] > peak_price:
                        peak_price = close[i]
                    trail_stop = peak_price - (atr_vals[i] * trail_mult)
                    if close[i] < trail_stop:
                        signals[i] = -1
                        in_position = 0
                elif in_position == -1:
                    if close[i] < peak_price:
                        peak_price = close[i]
                    trail_stop = peak_price + (atr_vals[i] * trail_mult)
                    if close[i] > trail_stop:
                        signals[i] = 1
                        in_position = 0

        df = df.copy()
        df["signal"] = signals
        return df


register_strategy("macd_divergence", MACDDivergenceStrategy)
