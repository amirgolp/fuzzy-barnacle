"""
Contango/Backwardation Fade Strategy

For commodities and VIX products, the term structure (contango/backwardation)
creates systematic alpha. This strategy:

1. In steep contango (futures > spot): Short bias (roll yield drag)
2. In steep backwardation (futures < spot): Long bias (positive roll yield)

For spot trading without futures data, we approximate term structure using:
- VIX contango: Compare VIX to its moving average
- Commodity ETFs: Use the slope of recent price action

This is particularly useful for:
- USO, UNG (energy)
- VXX, UVXY (volatility)
- GLD, SLV (precious metals)

Intraweek/Intramonth friendly with max_holding_bars parameter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantdash.strategies.base import Strategy
from quantdash.strategies.registry import register_strategy


class ContangoFadeStrategy(Strategy):
    """
    Contango fade / Backwardation ride strategy.

    Logic:
    - Estimate term structure from price action (rising = contango, falling = backwardation)
    - In contango: look for short opportunities on rallies
    - In backwardation: look for long opportunities on dips
    - Combine with mean reversion signals for entry timing

    Params:
        structure_period: Period to determine term structure (default: 20)
        contango_threshold: Slope threshold for contango (default: 0.002)
        backwardation_threshold: Slope threshold for backwardation (default: -0.002)
        rsi_period: RSI period for timing (default: 14)
        entry_rsi_high: RSI level to short in contango (default: 60)
        entry_rsi_low: RSI level to buy in backwardation (default: 40)
    """

    name = "contango_fade"
    description = "Trade based on implied term structure (contango/backwardation)"
    default_params = {
        "structure_period": 20,
        "contango_threshold": 0.002,
        "backwardation_threshold": -0.002,
        "rsi_period": 14,
        "entry_rsi_high": 60,
        "entry_rsi_low": 40,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        struct_period = p["structure_period"]
        contango_thresh = p["contango_threshold"]
        backward_thresh = p["backwardation_threshold"]
        rsi_period = p["rsi_period"]
        rsi_high = p["entry_rsi_high"]
        rsi_low = p["entry_rsi_low"]

        close = df["close"]

        # Estimate term structure from price slope
        # Positive slope = prices rising = contango environment
        # Negative slope = prices falling = backwardation environment
        def rolling_slope(series, window):
            """Calculate rolling slope using linear regression."""
            slopes = []
            for i in range(len(series)):
                if i < window - 1:
                    slopes.append(np.nan)
                else:
                    y = series.iloc[i - window + 1:i + 1].values
                    x = np.arange(window)
                    if len(y) == window and not np.any(np.isnan(y)):
                        slope = np.polyfit(x, y, 1)[0] / series.iloc[i]  # Normalized slope
                    else:
                        slope = np.nan
                    slopes.append(slope)
            return pd.Series(slopes, index=series.index)

        price_slope = rolling_slope(close, struct_period)

        # Classify term structure
        in_contango = price_slope > contango_thresh
        in_backwardation = price_slope < backward_thresh
        structure_neutral = (~in_contango) & (~in_backwardation)

        # Calculate RSI for entry timing
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        # Calculate momentum for confirmation
        momentum = close.pct_change(5)

        # Generate signals
        signal = pd.Series(0, index=df.index)

        # In contango: short on RSI bounces (fade the rally)
        short_in_contango = in_contango & (rsi > rsi_high) & (momentum > 0)

        # In backwardation: long on RSI dips (buy the dip)
        long_in_backwardation = in_backwardation & (rsi < rsi_low) & (momentum < 0)

        # Also add trend-following component
        # Strong contango with momentum = accelerating selling (short)
        strong_contango_trend = in_contango & (price_slope > contango_thresh * 2) & (momentum < -0.02)
        # Strong backwardation with momentum = accelerating buying (long)
        strong_backward_trend = in_backwardation & (price_slope < backward_thresh * 2) & (momentum > 0.02)

        signal = signal.where(~long_in_backwardation, 1)
        signal = signal.where(~strong_backward_trend, 1)
        signal = signal.where(~short_in_contango, -1)
        signal = signal.where(~strong_contango_trend, -1)

        result = pd.DataFrame(index=df.index)
        result["signal"] = signal.fillna(0).astype(int)
        result["price_slope"] = price_slope
        result["in_contango"] = in_contango.astype(int)
        result["in_backwardation"] = in_backwardation.astype(int)
        result["rsi"] = rsi

        # Calculate ATR for SL/TP
        high = df["high"]
        low = df["low"]
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()

        result["sl_level"] = close - (atr * 1.5)
        result["tp_level"] = close + (atr * 2)

        return result


# Register the strategy
register_strategy("contango_fade", ContangoFadeStrategy)
