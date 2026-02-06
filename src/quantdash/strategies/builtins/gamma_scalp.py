"""
Gamma Scalping Strategy for Spot Trading

Inspired by options gamma scalping where market makers hedge delta exposure.
This strategy identifies "gamma-like" setups in spot markets:

1. When price is near significant levels (like strike prices in options)
   and volatility is compressed, expect explosive moves
2. Trade the "gamma effect" - accelerating price moves away from key levels

Key concept: Options have highest gamma at-the-money near expiration.
For spot trading, we identify similar setups:
- Price at key support/resistance (like ATM strikes)
- Low realized volatility (like high gamma, low theta)
- Breakout potential

Intraweek/Intramonth friendly with max_holding_bars parameter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantdash.strategies.base import Strategy
from quantdash.strategies.registry import register_strategy


class GammaScalpStrategy(Strategy):
    """
    Gamma-inspired spot trading strategy.

    Logic:
    - Identify "pinning levels" (round numbers, recent pivots)
    - When price is near a pinning level and volatility is low, expect a breakout
    - Trade the breakout direction with tight stops
    - Use volatility expansion to determine exit

    Params:
        pivot_lookback: Period to identify pivot levels (default: 20)
        proximity_pct: How close price must be to level (default: 1.0)
        vol_lookback: Volatility measurement period (default: 10)
        vol_percentile: Low vol threshold percentile (default: 25)
        breakout_atr_mult: ATR multiplier for breakout confirmation (default: 1.5)
        round_number_weight: Weight given to round numbers (default: True)
    """

    name = "gamma_scalp"
    description = "Gamma-inspired breakout strategy for pinned prices"
    default_params = {
        "pivot_lookback": 20,
        "proximity_pct": 1.0,
        "vol_lookback": 10,
        "vol_percentile": 25,
        "breakout_atr_mult": 1.5,
        "round_number_weight": True,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        pivot_lookback = p["pivot_lookback"]
        proximity = p["proximity_pct"] / 100
        vol_lookback = p["vol_lookback"]
        vol_pct = p["vol_percentile"]
        atr_mult = p["breakout_atr_mult"]
        use_round = p["round_number_weight"]

        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Calculate ATR
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()

        # Calculate realized volatility
        log_returns = np.log(close / close.shift(1))
        realized_vol = log_returns.rolling(window=vol_lookback).std()
        vol_rank = realized_vol.rolling(window=252).apply(
            lambda x: (x.iloc[-1] <= np.percentile(x.dropna(), vol_pct)) if len(x.dropna()) > 0 else False,
            raw=False
        )
        low_vol = vol_rank == 1

        # Identify pivot levels
        pivot_high = high.rolling(window=pivot_lookback).max()
        pivot_low = low.rolling(window=pivot_lookback).min()
        pivot_mid = (pivot_high + pivot_low) / 2

        # Identify round number levels (optional)
        if use_round:
            # Find nearest round number (based on price magnitude)
            magnitude = 10 ** (np.floor(np.log10(close.dropna().iloc[-1] if len(close.dropna()) > 0 else 100)))
            round_level = (close / magnitude).round() * magnitude
        else:
            round_level = pd.Series(np.nan, index=df.index)

        # Check proximity to key levels
        near_pivot_high = (close - pivot_high.shift(1)).abs() / close < proximity
        near_pivot_low = (close - pivot_low.shift(1)).abs() / close < proximity
        near_round = (close - round_level).abs() / close < proximity if use_round else pd.Series(False, index=df.index)

        # Pinned condition: near any key level AND low volatility
        pinned = (near_pivot_high | near_pivot_low | near_round) & low_vol

        # Breakout detection
        breakout_up = close > (high.shift(1) + atr * atr_mult)
        breakout_down = close < (low.shift(1) - atr * atr_mult)

        # Generate signals
        signal = pd.Series(0, index=df.index)

        # Enter on breakout from pinned state
        # We need to track if we were pinned recently
        was_pinned = pinned.rolling(window=5).max() == 1

        buy_signal = was_pinned & breakout_up
        sell_signal = was_pinned & breakout_down

        signal = signal.where(~buy_signal, 1)
        signal = signal.where(~sell_signal, -1)

        # Also add pure low-vol breakout signals (gamma expansion plays)
        pure_breakout_up = low_vol.shift(1) & breakout_up & (~was_pinned)
        pure_breakout_down = low_vol.shift(1) & breakout_down & (~was_pinned)

        signal = signal.where(~pure_breakout_up, 1)
        signal = signal.where(~pure_breakout_down, -1)

        result = pd.DataFrame(index=df.index)
        result["signal"] = signal.fillna(0).astype(int)
        result["pinned"] = pinned.astype(int)
        result["low_vol"] = low_vol.astype(int)
        result["realized_vol"] = realized_vol
        result["atr"] = atr

        # Dynamic SL/TP based on ATR
        result["sl_level"] = close - (atr * 2)
        result["tp_level"] = close + (atr * 3)  # 1.5:1 reward/risk

        return result


# Register the strategy
register_strategy("gamma_scalp", GammaScalpStrategy)
