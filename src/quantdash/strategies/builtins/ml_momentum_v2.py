"""ML Momentum Strategy V2 - Improved.

Improvements over V1:
- Adaptive momentum threshold based on market volatility
- Trend strength confirmation
- Better entry/exit timing
- Multi-timeframe analysis
- Reduced false signals
"""

import numpy as np
import pandas as pd

from quantdash.features.indicators import atr
from quantdash.strategies.base import Strategy
from quantdash.strategies.registry import register_strategy


class MLMomentumV2Strategy(Strategy):
    """
    Improved ML-inspired momentum strategy with adaptive thresholds.

    Key improvements:
    - Adaptive momentum threshold based on volatility regime
    - Trend strength filter (ADX-like logic)
    - Multi-timeframe momentum confirmation
    - Better signal quality over quantity
    """

    name = "ml_momentum_v2"
    description = "Improved ML momentum with adaptive thresholds and trend confirmation"
    default_params = {
        "lookback_period": 120,
        "fast_period": 20,            # Fast momentum period
        "slow_period": 50,            # Slow momentum period
        "momentum_threshold": 0.08,   # Lower threshold for more signals
        "trend_strength_min": 0.15,   # Lower minimum (was 0.3 - too high!)
        "atr_period": 14,
        "sl_atr_multiplier": 2.0,     # Tighter stops
        "tp_atr_multiplier": 4.0,     # Wider targets (2:1 R:R)
        "trail_atr_multiplier": 1.5,
        "volatility_filter": False,   # Disable for now - was too restrictive
        "adaptive_threshold": False,  # Disable for now - use fixed threshold
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals with improved logic."""
        lookback = self.params["lookback_period"]
        fast_period = self.params["fast_period"]
        slow_period = self.params["slow_period"]
        base_threshold = self.params["momentum_threshold"]
        trend_strength_min = self.params["trend_strength_min"]
        atr_period = self.params["atr_period"]
        adaptive = self.params["adaptive_threshold"]

        result = df.copy()
        close = df["close"]

        # Calculate ATR for risk management
        atr_values = atr(df, period=atr_period)
        result["atr"] = atr_values

        # 1. Multi-timeframe momentum
        # Fast momentum (captures recent trends)
        fast_momentum = close.pct_change(fast_period)

        # Slow momentum (confirms longer-term direction)
        slow_momentum = close.pct_change(slow_period)

        # Very long momentum (overall trend)
        long_momentum = close.pct_change(lookback)

        # 2. Trend strength (ADX-like)
        # Calculate directional movement
        high = df["high"]
        low = df["low"]

        plus_dm = (high - high.shift(1)).clip(lower=0)
        minus_dm = (low.shift(1) - low).clip(lower=0)

        # Smooth directional movements
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr_values)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr_values)

        # Trend strength (0-1)
        dx = abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        trend_strength = dx.rolling(14).mean()
        result["trend_strength"] = trend_strength

        # 3. Adaptive momentum score with trend confirmation
        # Combine momentum with trend strength weighting
        momentum_score = (
            0.5 +  # Neutral baseline
            0.4 * np.tanh(fast_momentum * 100) +      # Fast momentum (40% weight)
            0.3 * np.tanh(slow_momentum * 100) +      # Slow momentum (30% weight)
            0.2 * np.tanh(long_momentum * 100) +      # Long momentum (20% weight)
            0.1 * (plus_di - minus_di) / 100          # Directional bias (10% weight)
        )
        result["momentum_score"] = momentum_score

        # 4. Adaptive threshold based on volatility
        if adaptive:
            # In high volatility, use higher threshold (be more selective)
            # In low volatility, use lower threshold (capture smaller moves)
            vol_percentile = atr_values.rolling(100).rank(pct=True)
            adaptive_threshold = base_threshold * (1 + 0.5 * vol_percentile)
        else:
            adaptive_threshold = base_threshold

        result["threshold"] = adaptive_threshold

        # 5. Generate signals with trend strength filter
        result["signal"] = 0

        # Buy conditions: strong upward momentum + strong trend
        buy_condition = (
            (momentum_score > (0.5 + adaptive_threshold)) &
            (trend_strength > trend_strength_min) &
            (fast_momentum > 0) &  # Fast momentum must be positive
            (slow_momentum > 0)    # Slow momentum confirms
        )

        # Sell conditions: strong downward momentum + strong trend
        sell_condition = (
            (momentum_score < (0.5 - adaptive_threshold)) &
            (trend_strength > trend_strength_min) &
            (fast_momentum < 0) &  # Fast momentum must be negative
            (slow_momentum < 0)    # Slow momentum confirms
        )

        # Apply volatility filter if enabled
        if self.params["volatility_filter"]:
            vol_percentile = atr_values.rolling(100).rank(pct=True)
            normal_vol = (vol_percentile > 0.15) & (vol_percentile < 0.85)
            buy_condition = buy_condition & normal_vol
            sell_condition = sell_condition & normal_vol

        result.loc[buy_condition, "signal"] = 1
        result.loc[sell_condition, "signal"] = -1

        # 6. Calculate dynamic SL/TP levels
        sl_multiplier = self.params["sl_atr_multiplier"]
        tp_multiplier = self.params["tp_atr_multiplier"]

        result["sl_distance"] = atr_values * sl_multiplier
        result["tp_distance"] = atr_values * tp_multiplier

        result.loc[result["signal"] == 1, "sl_level"] = close - result["sl_distance"]
        result.loc[result["signal"] == 1, "tp_level"] = close + result["tp_distance"]

        result.loc[result["signal"] == -1, "sl_level"] = close + result["sl_distance"]
        result.loc[result["signal"] == -1, "tp_level"] = close - result["tp_distance"]

        # Apply trailing stop
        result = self._apply_trailing_stop(result, atr_values)

        # Clean up NaN values
        result["signal"] = result["signal"].fillna(0)
        for col in result.select_dtypes(include=[np.number]).columns:
            if col != "signal":
                result[col] = result[col].ffill().fillna(0)

        return result

    def _apply_trailing_stop(self, df: pd.DataFrame, atr_values: pd.Series) -> pd.DataFrame:
        """Apply trailing stop logic using ATR."""
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
                        signals[i] = -1
                        in_position = 0

                if signals[i] == -1:
                    in_position = 0

        df = df.copy()
        df["signal"] = signals
        return df


register_strategy("ml_momentum_v2", MLMomentumV2Strategy)
