"""ADX Trend Strength Strategy.

Uses ADX (Average Directional Index) to identify strong trends
and +DI/-DI crossovers for direction.
Best for: Trending markets, avoiding whipsaws.
"""

import numpy as np
import pandas as pd

from quantdash.features.indicators import atr
from quantdash.strategies.base import Strategy
from quantdash.strategies.registry import register_strategy


class ADXTrendStrategy(Strategy):
    """
    ADX + Directional Movement trend following strategy.

    Enters when ADX indicates strong trend (>25) and directional
    indicators (+DI/-DI) signal trend direction.
    """

    name = "adx_trend"
    description = "ADX trend strength with directional movement"
    default_params = {
        "adx_period": 14,
        "adx_threshold": 25,           # ADX > 25 = strong trend
        "di_period": 14,
        "atr_period": 14,
        "sl_atr_multiplier": 2.0,
        "tp_atr_multiplier": 4.0,
        "trail_atr_multiplier": 1.5,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate ADX-based signals."""
        adx_period = self.params["adx_period"]
        adx_threshold = self.params["adx_threshold"]
        di_period = self.params["di_period"]
        atr_period = self.params["atr_period"]

        result = df.copy()
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # Calculate ATR
        atr_values = atr(df, period=atr_period)
        result["atr"] = atr_values

        # Calculate Directional Movement
        plus_dm = (high - high.shift(1)).clip(lower=0)
        minus_dm = (low.shift(1) - low).clip(lower=0)

        # Zero out if opposite move is larger
        plus_dm = pd.Series(np.where(plus_dm > minus_dm, plus_dm, 0), index=df.index)
        minus_dm = pd.Series(np.where(minus_dm > plus_dm, minus_dm, 0), index=df.index)

        # Smooth directional movements
        plus_dm_smooth = plus_dm.rolling(di_period).sum()
        minus_dm_smooth = minus_dm.rolling(di_period).sum()
        atr_smooth = atr_values.rolling(di_period).sum()

        # Calculate Directional Indicators
        plus_di = 100 * (plus_dm_smooth / (atr_smooth + 1e-10))
        minus_di = 100 * (minus_dm_smooth / (atr_smooth + 1e-10))

        result["plus_di"] = plus_di
        result["minus_di"] = minus_di

        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(adx_period).mean()
        result["adx"] = adx

        # Generate signals
        result["signal"] = 0

        # Long: +DI crosses above -DI with strong ADX
        di_cross_up = (plus_di > minus_di) & (plus_di.shift(1) <= minus_di.shift(1))
        long_signal = di_cross_up & (adx > adx_threshold)

        # Short: -DI crosses above +DI with strong ADX
        di_cross_down = (minus_di > plus_di) & (minus_di.shift(1) <= plus_di.shift(1))
        short_signal = di_cross_down & (adx > adx_threshold)

        # Exit when DI reverses
        exit_long = (minus_di > plus_di) & (minus_di.shift(1) <= plus_di.shift(1))
        exit_short = (plus_di > minus_di) & (plus_di.shift(1) <= minus_di.shift(1))

        result.loc[long_signal, "signal"] = 1
        result.loc[short_signal, "signal"] = -1

        # Track position and apply exits
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

    def _apply_exits(self, df: pd.DataFrame, exit_long: pd.Series, exit_short: pd.Series) -> pd.DataFrame:
        """Apply exit signals based on DI crossovers."""
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


register_strategy("adx_trend", ADXTrendStrategy)
