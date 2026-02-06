"""ATR Breakout Strategy.

Enters when price moves beyond a threshold based on ATR (volatility expansion).
Best for: Momentum trading, volatility breakouts.
"""

import numpy as np
import pandas as pd

from quantdash.features.indicators import atr
from quantdash.strategies.base import Strategy
from quantdash.strategies.registry import register_strategy


class ATRBreakoutStrategy(Strategy):
    """
    ATR-based breakout strategy for volatility expansion.

    Enters when price moves more than X * ATR from recent reference point,
    indicating a strong directional move with volatility expansion.
    """

    name = "atr_breakout"
    description = "Volatility expansion breakout using ATR"
    default_params = {
        "atr_period": 14,
        "breakout_multiplier": 2.0,    # Price move > 2 * ATR = breakout
        "reference_period": 5,         # Look back for reference price
        "volume_confirmation": True,    # Require volume spike
        "volume_multiplier": 1.5,      # Volume > 1.5x average
        "volume_period": 20,
        "sl_atr_multiplier": 2.0,
        "tp_atr_multiplier": 4.0,
        "trail_atr_multiplier": 1.5,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate ATR breakout signals."""
        atr_period = self.params["atr_period"]
        breakout_mult = self.params["breakout_multiplier"]
        ref_period = self.params["reference_period"]
        vol_confirm = self.params["volume_confirmation"]
        vol_mult = self.params["volume_multiplier"]
        vol_period = self.params["volume_period"]

        result = df.copy()
        close = df["close"]
        volume = df["volume"] if "volume" in df.columns else pd.Series(1, index=df.index)

        # Calculate ATR
        atr_values = atr(df, period=atr_period)
        result["atr"] = atr_values

        # Calculate reference price (recent high/low)
        ref_high = df["high"].rolling(ref_period).max().shift(1)
        ref_low = df["low"].rolling(ref_period).min().shift(1)
        ref_close = close.rolling(ref_period).mean().shift(1)

        result["ref_high"] = ref_high
        result["ref_low"] = ref_low
        result["ref_close"] = ref_close

        # Calculate breakout threshold
        breakout_threshold = atr_values * breakout_mult
        result["breakout_threshold"] = breakout_threshold

        # Detect breakouts
        # Bullish breakout: price breaks above ref_high by > threshold
        bullish_breakout = (close - ref_high) > breakout_threshold

        # Bearish breakout: price breaks below ref_low by > threshold
        bearish_breakout = (ref_low - close) > breakout_threshold

        # Volume confirmation
        if vol_confirm:
            avg_volume = volume.rolling(vol_period).mean()
            high_volume = volume > (avg_volume * vol_mult)
            result["volume_ratio"] = volume / (avg_volume + 1e-10)

            bullish_breakout = bullish_breakout & high_volume
            bearish_breakout = bearish_breakout & high_volume

        # ATR percentile (filter for normal volatility)
        atr_percentile = atr_values.rolling(100).rank(pct=True)
        normal_volatility = (atr_percentile > 0.2) & (atr_percentile < 0.8)
        result["atr_percentile"] = atr_percentile

        # Generate signals
        result["signal"] = 0

        # Only trade in normal volatility regime
        long_signal = bullish_breakout & normal_volatility
        short_signal = bearish_breakout & normal_volatility

        result.loc[long_signal, "signal"] = 1
        result.loc[short_signal, "signal"] = -1

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


register_strategy("atr_breakout", ATRBreakoutStrategy)
