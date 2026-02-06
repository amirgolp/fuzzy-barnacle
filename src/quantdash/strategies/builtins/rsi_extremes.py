"""RSI Extremes Strategy.

Enters on RSI extreme levels with multi-timeframe confirmation.
Best for: Mean reversion, oversold/overbought conditions.
"""

import numpy as np
import pandas as pd

from quantdash.features.indicators import atr, rsi
from quantdash.strategies.base import Strategy
from quantdash.strategies.registry import register_strategy


class RSIExtremesStrategy(Strategy):
    """
    RSI extremes strategy with multi-timeframe confirmation.

    Enters long when RSI is oversold, short when overbought.
    Optional multi-timeframe confirmation for higher quality signals.
    """

    name = "rsi_extremes"
    description = "RSI oversold/overbought with multi-timeframe confirmation"
    default_params = {
        "rsi_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "rsi_exit_long": 70,           # Exit long when RSI reaches this
        "rsi_exit_short": 30,          # Exit short when RSI reaches this
        "use_mtf": True,               # Multi-timeframe confirmation
        "mtf_multiplier": 3,           # 3x period for MTF RSI
        "atr_period": 14,
        "sl_atr_multiplier": 2.0,
        "tp_atr_multiplier": 3.0,
        "trail_atr_multiplier": 1.0,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate RSI extreme signals."""
        rsi_period = self.params["rsi_period"]
        rsi_oversold = self.params["rsi_oversold"]
        rsi_overbought = self.params["rsi_overbought"]
        rsi_exit_long = self.params["rsi_exit_long"]
        rsi_exit_short = self.params["rsi_exit_short"]
        use_mtf = self.params["use_mtf"]
        mtf_mult = self.params["mtf_multiplier"]
        atr_period = self.params["atr_period"]

        result = df.copy()
        close = df["close"]

        # Calculate RSI
        rsi_values = rsi(df, period=rsi_period)
        result["rsi"] = rsi_values

        # Calculate multi-timeframe RSI if enabled
        if use_mtf:
            rsi_mtf = rsi(df, period=rsi_period * mtf_mult)
            result["rsi_mtf"] = rsi_mtf
        else:
            rsi_mtf = pd.Series(50, index=df.index)  # Neutral

        # Calculate ATR
        atr_values = atr(df, period=atr_period)
        result["atr"] = atr_values

        # Generate signals
        result["signal"] = 0

        # Long: RSI oversold on both timeframes
        if use_mtf:
            long_signal = (
                (rsi_values < rsi_oversold) &
                (rsi_mtf < rsi_oversold)
            )
            short_signal = (
                (rsi_values > rsi_overbought) &
                (rsi_mtf > rsi_overbought)
            )
        else:
            long_signal = rsi_values < rsi_oversold
            short_signal = rsi_values > rsi_overbought

        # Exit signals
        exit_long = rsi_values > rsi_exit_long
        exit_short = rsi_values < rsi_exit_short

        result.loc[long_signal, "signal"] = 1
        result.loc[short_signal, "signal"] = -1

        # Apply RSI-based exits
        result = self._apply_rsi_exits(result, exit_long, exit_short)

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

    def _apply_rsi_exits(self, df: pd.DataFrame, exit_long: pd.Series, exit_short: pd.Series) -> pd.DataFrame:
        """Apply exit signals based on RSI levels."""
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


register_strategy("rsi_extremes", RSIExtremesStrategy)
