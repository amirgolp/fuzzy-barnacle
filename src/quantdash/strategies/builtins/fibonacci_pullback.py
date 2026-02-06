"""Fibonacci Pullback Strategy.

Enters on pullbacks to Fibonacci retracement levels within established trends.
Best for: Trending markets, swing trading.
"""

import numpy as np
import pandas as pd

from quantdash.features.indicators import atr
from quantdash.strategies.base import Strategy
from quantdash.strategies.registry import register_strategy


class FibonacciPullbackStrategy(Strategy):
    """
    Trend + Pullback strategy using Fibonacci retracements.

    Identifies trends, waits for pullbacks to key Fibonacci levels
    (38.2%, 50%, 61.8%), then enters in the trend direction.
    """

    name = "fibonacci_pullback"
    description = "Enter on Fibonacci pullbacks within trends"
    default_params = {
        "trend_period": 50,            # SMA period for trend
        "swing_lookback": 20,          # Period to identify swing high/low
        "fib_tolerance": 0.02,         # % tolerance for Fib level hits
        "min_pullback_bars": 3,        # Min bars in pullback
        "atr_period": 14,
        "sl_atr_multiplier": 2.0,
        "tp_risk_ratio": 2.0,          # TP is 2x SL distance
        "trail_atr_multiplier": 1.5,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate Fibonacci pullback signals."""
        trend_period = self.params["trend_period"]
        swing_lookback = self.params["swing_lookback"]
        fib_tol = self.params["fib_tolerance"]
        min_pullback = self.params["min_pullback_bars"]
        atr_period = self.params["atr_period"]

        result = df.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Calculate trend
        sma = close.rolling(trend_period).mean()
        result["sma"] = sma
        trend = np.where(close > sma, 1, -1)
        result["trend"] = trend

        # Calculate ATR
        atr_values = atr(df, period=atr_period)
        result["atr"] = atr_values

        # Identify swing highs and lows
        swing_high = high.rolling(swing_lookback, center=True).max()
        swing_low = low.rolling(swing_lookback, center=True).min()

        # Calculate Fibonacci levels
        fib_range = swing_high - swing_low
        fib_382 = swing_low + 0.382 * fib_range
        fib_500 = swing_low + 0.500 * fib_range
        fib_618 = swing_low + 0.618 * fib_range

        result["swing_high"] = swing_high
        result["swing_low"] = swing_low
        result["fib_382"] = fib_382
        result["fib_500"] = fib_500
        result["fib_618"] = fib_618

        # Detect pullbacks
        # Count bars since trend started
        trend_change = (trend != np.roll(trend, 1)).astype(int)
        bars_in_trend = pd.Series(0, index=df.index)
        counter = 0
        for i in range(len(df)):
            if trend_change[i]:
                counter = 0
            else:
                counter += 1
            bars_in_trend.iloc[i] = counter

        result["bars_in_trend"] = bars_in_trend

        # Price touching Fibonacci levels
        price_range = close * fib_tol
        at_fib_382 = (close >= fib_382 - price_range) & (close <= fib_382 + price_range)
        at_fib_500 = (close >= fib_500 - price_range) & (close <= fib_500 + price_range)
        at_fib_618 = (close >= fib_618 - price_range) & (close <= fib_618 + price_range)

        at_any_fib = at_fib_382 | at_fib_500 | at_fib_618

        # Generate signals
        result["signal"] = 0

        # Long: uptrend + pullback to Fib + enough bars in trend
        long_signal = (
            (trend == 1) &
            at_any_fib &
            (bars_in_trend >= min_pullback) &
            (close < close.shift(1))  # Pullback (price declining)
        )

        # Short: downtrend + pullback to Fib + enough bars in trend
        short_signal = (
            (trend == -1) &
            at_any_fib &
            (bars_in_trend >= min_pullback) &
            (close > close.shift(1))  # Pullback (price rising)
        )

        result.loc[long_signal, "signal"] = 1
        result.loc[short_signal, "signal"] = -1

        # Calculate SL/TP levels
        sl_mult = self.params["sl_atr_multiplier"]
        tp_ratio = self.params["tp_risk_ratio"]

        sl_distance = atr_values * sl_mult
        tp_distance = sl_distance * tp_ratio

        result["sl_distance"] = sl_distance
        result["tp_distance"] = tp_distance

        result.loc[result["signal"] == 1, "sl_level"] = close - sl_distance
        result.loc[result["signal"] == 1, "tp_level"] = close + tp_distance

        result.loc[result["signal"] == -1, "sl_level"] = close + sl_distance
        result.loc[result["signal"] == -1, "tp_level"] = close - tp_distance

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


register_strategy("fibonacci_pullback", FibonacciPullbackStrategy)
