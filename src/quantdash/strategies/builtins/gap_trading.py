"""Gap Trading Strategy.

Trades overnight gaps with statistical fill probability.
Best for: CFDs, indices, gap fill opportunities.
"""

import numpy as np
import pandas as pd

from quantdash.features.indicators import atr
from quantdash.strategies.base import Strategy
from quantdash.strategies.registry import register_strategy


class GapTradingStrategy(Strategy):
    """
    Gap trading strategy for CFDs and overnight gaps.

    Identifies gaps between previous close and current open,
    trades gap fill with statistical edge.
    """

    name = "gap_trading"
    description = "Overnight gap trading with fill probability"
    default_params = {
        "min_gap_pct": 0.5,            # Min gap size (% of price)
        "max_gap_pct": 3.0,            # Max gap size (too large = risk)
        "gap_fill_target": 0.5,        # Fill 50% of gap
        "trend_filter": True,          # Only trade gaps in direction of trend
        "trend_period": 50,
        "atr_period": 14,
        "sl_atr_multiplier": 2.0,
        "trail_atr_multiplier": 1.0,
        "max_hold_bars": 5,            # Exit if gap doesn't fill quickly
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate gap trading signals."""
        min_gap = self.params["min_gap_pct"] / 100
        max_gap = self.params["max_gap_pct"] / 100
        gap_fill_target = self.params["gap_fill_target"]
        use_trend_filter = self.params["trend_filter"]
        trend_period = self.params["trend_period"]
        atr_period = self.params["atr_period"]

        result = df.copy()
        open_price = df["open"] if "open" in df.columns else df["close"]
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Calculate gaps
        prev_close = close.shift(1)
        gap = (open_price - prev_close) / prev_close
        gap_size = abs(gap)
        result["gap"] = gap
        result["gap_size"] = gap_size

        # Gap type
        gap_up = gap > 0
        gap_down = gap < 0

        # Calculate ATR
        atr_values = atr(df, period=atr_period)
        result["atr"] = atr_values

        # Trend filter
        if use_trend_filter:
            sma = close.rolling(trend_period).mean()
            uptrend = close > sma
            downtrend = close < sma
            result["sma"] = sma
        else:
            uptrend = pd.Series(True, index=df.index)
            downtrend = pd.Series(True, index=df.index)

        # Calculate gap fill levels
        gap_fill_level = prev_close + (gap * gap_fill_target)
        result["gap_fill_level"] = gap_fill_level

        # Generate signals
        result["signal"] = 0

        # Fade gap up (short): gap up + in downtrend + gap size valid
        fade_gap_up = (
            gap_up &
            (gap_size >= min_gap) &
            (gap_size <= max_gap) &
            downtrend
        )

        # Fade gap down (long): gap down + in uptrend + gap size valid
        fade_gap_down = (
            gap_down &
            (gap_size >= min_gap) &
            (gap_size <= max_gap) &
            uptrend
        )

        result.loc[fade_gap_up, "signal"] = -1  # Short to fade gap up
        result.loc[fade_gap_down, "signal"] = 1  # Long to fade gap down

        # Calculate SL/TP levels
        sl_mult = self.params["sl_atr_multiplier"]

        # For gap trades, SL is beyond the gap
        # TP is the gap fill level
        result.loc[result["signal"] == 1, "sl_level"] = open_price - (atr_values * sl_mult)
        result.loc[result["signal"] == 1, "tp_level"] = gap_fill_level

        result.loc[result["signal"] == -1, "sl_level"] = open_price + (atr_values * sl_mult)
        result.loc[result["signal"] == -1, "tp_level"] = gap_fill_level

        # Apply time-based exit (if gap doesn't fill quickly, exit)
        result = self._apply_time_exit(result, gap_fill_level)

        # Apply trailing stop
        result = self._apply_trailing_stop(result, atr_values)

        # Clean up
        result["signal"] = result["signal"].fillna(0)
        for col in result.select_dtypes(include=[np.number]).columns:
            if col != "signal":
                result[col] = result[col].ffill().fillna(0)

        return result

    def _apply_time_exit(self, df: pd.DataFrame, gap_fill_level: pd.Series) -> pd.DataFrame:
        """Exit if gap doesn't fill within max_hold_bars."""
        max_hold = self.params["max_hold_bars"]
        signals = df["signal"].values.copy()
        close = df["close"].values
        gap_fill = gap_fill_level.values

        in_position = 0
        entry_bar = 0

        for i in range(len(signals)):
            if signals[i] == 1:
                in_position = 1
                entry_bar = i
            elif signals[i] == -1:
                in_position = -1
                entry_bar = i
            elif in_position != 0:
                bars_held = i - entry_bar

                # Check if gap filled
                if in_position == 1:
                    # Long position: check if price reached gap fill level
                    if close[i] >= gap_fill[entry_bar]:
                        signals[i] = -1  # Exit
                        in_position = 0
                elif in_position == -1:
                    # Short position: check if price reached gap fill level
                    if close[i] <= gap_fill[entry_bar]:
                        signals[i] = 1  # Exit
                        in_position = 0

                # Time-based exit
                if bars_held >= max_hold and in_position != 0:
                    signals[i] = -in_position  # Exit
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


register_strategy("gap_trading", GapTradingStrategy)
