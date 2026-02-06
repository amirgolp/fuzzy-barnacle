"""Breakout Trading Strategy.

Identifies consolidation ranges and enters on breakouts with volume confirmation.
Best for: High volatility instruments, trending markets.
"""

import numpy as np
import pandas as pd

from quantdash.features.indicators import atr
from quantdash.strategies.base import Strategy
from quantdash.strategies.registry import register_strategy


class BreakoutTradingStrategy(Strategy):
    """
    Breakout trading strategy with volume confirmation.

    Enters when price breaks above resistance or below support
    after a consolidation period, confirmed by volume expansion.
    """

    name = "breakout_trading"
    description = "Support/resistance breakout with volume confirmation"
    default_params = {
        "lookback_period": 20,         # Period to identify S/R levels
        "consolidation_threshold": 0.5, # Max ATR ratio for consolidation
        "volume_multiplier": 1.5,      # Volume must be X times average
        "breakout_threshold": 0.02,    # Min % move for breakout
        "atr_period": 14,
        "sl_atr_multiplier": 2.0,
        "tp_atr_multiplier": 3.0,
        "trail_atr_multiplier": 1.5,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate breakout signals."""
        lookback = self.params["lookback_period"]
        consol_threshold = self.params["consolidation_threshold"]
        vol_mult = self.params["volume_multiplier"]
        breakout_pct = self.params["breakout_threshold"]
        atr_period = self.params["atr_period"]

        result = df.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"] if "volume" in df.columns else pd.Series(1, index=df.index)

        # Calculate ATR for volatility measurement
        atr_values = atr(df, period=atr_period)
        result["atr"] = atr_values

        # Calculate rolling support and resistance
        resistance = high.rolling(lookback).max()
        support = low.rolling(lookback).min()
        result["resistance"] = resistance
        result["support"] = support

        # Calculate range (as % of price)
        price_range = (resistance - support) / close
        result["range_pct"] = price_range

        # Identify consolidation: range is narrow relative to ATR
        avg_atr = atr_values.rolling(lookback).mean()
        range_atr_ratio = (resistance - support) / (avg_atr + 1e-10)
        is_consolidating = range_atr_ratio < consol_threshold
        result["consolidating"] = is_consolidating.astype(int)

        # Volume analysis
        avg_volume = volume.rolling(lookback).mean()
        volume_ratio = volume / (avg_volume + 1e-10)
        high_volume = volume_ratio > vol_mult
        result["volume_ratio"] = volume_ratio

        # Detect breakouts
        # Bullish breakout: price breaks above resistance after consolidation
        prev_consol = is_consolidating.shift(1).fillna(False)
        breakout_up = (
            (close > resistance.shift(1)) &
            ((close - resistance.shift(1)) / close > breakout_pct) &
            prev_consol &
            high_volume
        )

        # Bearish breakout: price breaks below support after consolidation
        breakout_down = (
            (close < support.shift(1)) &
            ((support.shift(1) - close) / close > breakout_pct) &
            prev_consol &
            high_volume
        )

        # Generate signals
        result["signal"] = 0
        result.loc[breakout_up, "signal"] = 1
        result.loc[breakout_down, "signal"] = -1

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


register_strategy("breakout_trading", BreakoutTradingStrategy)
