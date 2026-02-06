"""Bollinger Band Mean Reversion Strategy.

Enters when price touches outer Bollinger Bands with RSI confirmation.
Best for: Ranging markets, oversold/overbought conditions.
"""

import numpy as np
import pandas as pd

from quantdash.features.indicators import atr, rsi
from quantdash.strategies.base import Strategy
from quantdash.strategies.registry import register_strategy


class BollingerReversionStrategy(Strategy):
    """
    Bollinger Band mean reversion strategy with RSI confirmation.

    Enters long when price touches lower band with RSI oversold,
    enters short when price touches upper band with RSI overbought.
    """

    name = "bollinger_reversion"
    description = "Mean reversion at Bollinger Band extremes with RSI"
    default_params = {
        "bb_period": 20,
        "bb_std": 2.0,                 # Standard deviations
        "rsi_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "atr_period": 14,
        "sl_atr_multiplier": 2.0,
        "tp_target_bb": True,          # TP at middle band
        "trail_atr_multiplier": 1.0,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate Bollinger Band reversion signals."""
        bb_period = self.params["bb_period"]
        bb_std = self.params["bb_std"]
        rsi_period = self.params["rsi_period"]
        rsi_oversold = self.params["rsi_oversold"]
        rsi_overbought = self.params["rsi_overbought"]
        atr_period = self.params["atr_period"]

        result = df.copy()
        close = df["close"]

        # Calculate Bollinger Bands
        bb_middle = close.rolling(bb_period).mean()
        bb_std_dev = close.rolling(bb_period).std()
        bb_upper = bb_middle + (bb_std * bb_std_dev)
        bb_lower = bb_middle - (bb_std * bb_std_dev)

        result["bb_middle"] = bb_middle
        result["bb_upper"] = bb_upper
        result["bb_lower"] = bb_lower

        # Calculate RSI
        rsi_values = rsi(df, period=rsi_period)
        result["rsi"] = rsi_values

        # Calculate ATR
        atr_values = atr(df, period=atr_period)
        result["atr"] = atr_values

        # Calculate %B (position within bands)
        percent_b = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)
        result["percent_b"] = percent_b

        # Generate signals
        result["signal"] = 0

        # Long: price at/below lower band + RSI oversold
        long_signal = (
            (close <= bb_lower) &
            (rsi_values < rsi_oversold)
        )

        # Short: price at/above upper band + RSI overbought
        short_signal = (
            (close >= bb_upper) &
            (rsi_values > rsi_overbought)
        )

        # Exit when price reaches middle band
        exit_long = close >= bb_middle
        exit_short = close <= bb_middle

        result.loc[long_signal, "signal"] = 1
        result.loc[short_signal, "signal"] = -1

        # Track position and apply mean reversion exits
        result = self._apply_mean_reversion_exits(result, exit_long, exit_short)

        # Calculate SL/TP levels
        sl_mult = self.params["sl_atr_multiplier"]

        if self.params["tp_target_bb"]:
            # TP at middle band (mean reversion target)
            result.loc[result["signal"] == 1, "sl_level"] = close - (atr_values * sl_mult)
            result.loc[result["signal"] == 1, "tp_level"] = bb_middle

            result.loc[result["signal"] == -1, "sl_level"] = close + (atr_values * sl_mult)
            result.loc[result["signal"] == -1, "tp_level"] = bb_middle
        else:
            # Fixed ATR-based TP
            result.loc[result["signal"] == 1, "sl_level"] = close - (atr_values * sl_mult)
            result.loc[result["signal"] == 1, "tp_level"] = close + (atr_values * sl_mult * 2)

            result.loc[result["signal"] == -1, "sl_level"] = close + (atr_values * sl_mult)
            result.loc[result["signal"] == -1, "tp_level"] = close - (atr_values * sl_mult * 2)

        # Apply trailing stop
        result = self._apply_trailing_stop(result, atr_values)

        # Clean up
        result["signal"] = result["signal"].fillna(0)
        for col in result.select_dtypes(include=[np.number]).columns:
            if col != "signal":
                result[col] = result[col].ffill().fillna(0)

        return result

    def _apply_mean_reversion_exits(self, df: pd.DataFrame, exit_long: pd.Series, exit_short: pd.Series) -> pd.DataFrame:
        """Apply exit signals when price reaches middle band."""
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


register_strategy("bollinger_reversion", BollingerReversionStrategy)
