"""ML-Inspired Momentum Strategy.

Based on MetaTrader5 CNN-LSTM strategy that predicts EURUSD H1 movements.
Uses 120-period momentum analysis with ATR-based risk management.

This strategy approximates the neural network predictions using:
- Long-term momentum trends (120 periods)
- Short-term price velocity
- Volatility regime detection
- Dynamic ATR-based stops
"""

import numpy as np
import pandas as pd

from quantdash.features.indicators import atr
from quantdash.strategies.base import Strategy
from quantdash.strategies.registry import register_strategy


class MLMomentumStrategy(Strategy):
    """
    ML-inspired momentum strategy with ATR-based risk management.

    Approximates neural network predictions using technical analysis:
    - 120-period price momentum
    - Normalized price position within recent range
    - Volatility-adjusted entry signals
    - Dynamic SL/TP based on ATR
    """

    name = "ml_momentum"
    description = "ML-inspired momentum with ATR risk management (from MT5 CNN-LSTM strategy)"
    default_params = {
        "lookback_period": 120,        # Same as ML model input size
        "momentum_threshold": 0.15,    # Minimum momentum score to trade (optimized)
        "atr_period": 14,
        "sl_atr_multiplier": 2.5,      # Stop loss = ATR * multiplier
        "tp_atr_multiplier": 3.0,      # Take profit = ATR * multiplier
        "trail_atr_multiplier": 1.0,   # Trailing stop = ATR * multiplier
        "risk_percent": 3.0,           # Position sizing: 3% risk per trade
        "volatility_filter": True,     # Only trade in normal volatility regime
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on ML-inspired momentum analysis."""
        lookback = self.params["lookback_period"]
        mom_threshold = self.params["momentum_threshold"]
        atr_period = self.params["atr_period"]
        vol_filter = self.params["volatility_filter"]

        result = df.copy()
        close = df["close"]

        # Calculate ATR for risk management
        atr_values = atr(df, period=atr_period)
        result["atr"] = atr_values

        # 1. Calculate normalized price position (similar to MinMaxScaler in ML model)
        rolling_min = close.rolling(window=lookback, min_periods=lookback).min()
        rolling_max = close.rolling(window=lookback, min_periods=lookback).max()

        # Avoid division by zero
        price_range = rolling_max - rolling_min
        price_range = price_range.replace(0, np.nan)

        normalized_price = (close - rolling_min) / price_range
        result["normalized_price"] = normalized_price

        # 2. Calculate momentum score (approximates neural network output)
        # This combines multiple timeframe momentum signals

        # Long-term momentum (120 periods)
        long_momentum = close.pct_change(lookback)

        # Medium-term momentum (30 periods)
        med_momentum = close.pct_change(30)

        # Short-term velocity (rate of change)
        short_velocity = close.pct_change(5)

        # Combine into composite momentum score (0-1 range)
        # Positive momentum -> higher score, negative -> lower score
        momentum_score = (
            0.5 +  # Base at 0.5 (neutral)
            0.3 * np.tanh(long_momentum * 100) +  # Scale and bound to [-0.3, 0.3]
            0.15 * np.tanh(med_momentum * 100) +
            0.05 * np.tanh(short_velocity * 100)
        )
        result["momentum_score"] = momentum_score

        # 3. Volatility regime detection
        # Calculate rolling volatility (20-period standard deviation of returns)
        returns = close.pct_change()
        rolling_vol = returns.rolling(window=20).std()
        vol_percentile = rolling_vol.rolling(window=100).rank(pct=True)
        result["volatility_regime"] = vol_percentile

        # 4. Generate signals
        result["signal"] = 0

        # Buy signal: momentum score > threshold (predicting price increase)
        buy_condition = momentum_score > (0.5 + mom_threshold)

        # Sell signal: momentum score < threshold (predicting price decrease)
        sell_condition = momentum_score < (0.5 - mom_threshold)

        # Apply volatility filter if enabled (avoid extreme volatility periods)
        if vol_filter:
            # Only trade when volatility is between 20th and 80th percentile
            normal_vol = (vol_percentile > 0.2) & (vol_percentile < 0.8)
            buy_condition = buy_condition & normal_vol
            sell_condition = sell_condition & normal_vol

        # Set signals
        result.loc[buy_condition, "signal"] = 1
        result.loc[sell_condition, "signal"] = -1

        # 5. Calculate dynamic SL/TP levels based on ATR
        sl_multiplier = self.params["sl_atr_multiplier"]
        tp_multiplier = self.params["tp_atr_multiplier"]

        result["sl_distance"] = atr_values * sl_multiplier
        result["tp_distance"] = atr_values * tp_multiplier

        # For buy signals: SL below entry, TP above entry
        result.loc[result["signal"] == 1, "sl_level"] = close - result["sl_distance"]
        result.loc[result["signal"] == 1, "tp_level"] = close + result["tp_distance"]

        # For sell signals: SL above entry, TP below entry
        result.loc[result["signal"] == -1, "sl_level"] = close + result["sl_distance"]
        result.loc[result["signal"] == -1, "tp_level"] = close - result["tp_distance"]

        # Apply trailing stop simulation
        result = self._apply_trailing_stop(result, atr_values)

        # Clean up NaN values (replace with 0 for signals, forward-fill for other columns)
        result["signal"] = result["signal"].fillna(0)

        # Fill NaN values in all numeric columns
        for col in result.select_dtypes(include=[np.number]).columns:
            if col != "signal":
                result[col] = result[col].ffill().fillna(0)

        return result

    def _apply_trailing_stop(self, df: pd.DataFrame, atr_values: pd.Series) -> pd.DataFrame:
        """
        Apply trailing stop logic using ATR.

        When in position, track highest high (for longs) or lowest low (for shorts).
        Exit if price retraces by trail_atr_multiplier * ATR from the peak/trough.
        """
        trail_mult = self.params["trail_atr_multiplier"]

        signals = df["signal"].values.copy()
        close = df["close"].values
        atr_vals = atr_values.values

        in_position = 0  # 1 for long, -1 for short
        peak_price = 0.0

        for i in range(len(signals)):
            # Entry
            if signals[i] == 1 and in_position == 0:
                in_position = 1
                peak_price = close[i]
            elif signals[i] == -1 and in_position == 0:
                in_position = -1
                peak_price = close[i]

            # While in position, check trailing stop
            elif in_position != 0:
                if in_position == 1:  # Long position
                    # Update peak
                    if close[i] > peak_price:
                        peak_price = close[i]

                    # Check trailing stop (price drops from peak by ATR * multiplier)
                    trail_stop = peak_price - (atr_vals[i] * trail_mult)
                    if close[i] < trail_stop:
                        signals[i] = -1  # Exit signal
                        in_position = 0

                elif in_position == -1:  # Short position
                    # Update trough
                    if close[i] < peak_price:
                        peak_price = close[i]

                    # Check trailing stop (price rises from trough by ATR * multiplier)
                    trail_stop = peak_price + (atr_vals[i] * trail_mult)
                    if close[i] > trail_stop:
                        signals[i] = -1  # Exit signal (close short)
                        in_position = 0

                # Strategy-generated exit
                if signals[i] == -1:
                    in_position = 0

        df = df.copy()
        df["signal"] = signals
        return df


register_strategy("ml_momentum", MLMomentumStrategy)
