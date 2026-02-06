"""Heikin-Ashi Momentum Strategy.

Uses Heikin-Ashi candles to filter noise and identify clean trends.
Combines HA candle patterns (no lower wick = strong bull, no upper
wick = strong bear) with momentum confirmation via smoothed HA close.

Best for: Trend-following on noisy instruments, reducing whipsaws
compared to standard candlestick strategies.
"""

import numpy as np
import pandas as pd

from quantdash.features.indicators import atr, ema
from quantdash.strategies.base import Strategy
from quantdash.strategies.registry import register_strategy


class HeikinAshiMomentumStrategy(Strategy):
    """
    Heikin-Ashi smoothed momentum strategy.

    Heikin-Ashi candles are computed from standard OHLC data:
      HA_Close = (O + H + L + C) / 4
      HA_Open  = (prev_HA_Open + prev_HA_Close) / 2
      HA_High  = max(H, HA_Open, HA_Close)
      HA_Low   = min(L, HA_Open, HA_Close)

    Signal logic:
      - Strong bullish: HA_Close > HA_Open AND no lower wick (HA_Low == HA_Open)
      - Strong bearish: HA_Close < HA_Open AND no upper wick (HA_High == HA_Open)
      - Entry on N consecutive strong candles in same direction
      - Exit on reversal candle or EMA crossover

    Includes trend filter via EMA and ATR-based risk management.
    """

    name = "heikin_ashi_momentum"
    description = "Heikin-Ashi smoothed trend-following with momentum"
    default_params = {
        "consecutive_candles": 3,     # N strong candles to confirm entry
        "ema_fast": 12,               # Fast EMA for trend filter
        "ema_slow": 26,               # Slow EMA for trend filter
        "require_ema_align": True,    # Require EMA alignment with HA direction
        "wick_tolerance": 0.001,      # Tolerance for "no wick" detection (0.1%)
        "atr_period": 14,
        "sl_atr_multiplier": 2.0,
        "tp_atr_multiplier": 4.0,
        "trail_atr_multiplier": 1.5,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate Heikin-Ashi momentum signals."""
        consec = self.params["consecutive_candles"]
        ema_fast_p = self.params["ema_fast"]
        ema_slow_p = self.params["ema_slow"]
        require_ema = self.params["require_ema_align"]
        wick_tol = self.params["wick_tolerance"]
        atr_period = self.params["atr_period"]

        result = df.copy()
        n = len(df)
        o = df["open"].values if "open" in df.columns else df["close"].values
        h = df["high"].values
        l = df["low"].values
        c = df["close"].values

        # ── Compute Heikin-Ashi candles ──
        ha_close = np.zeros(n)
        ha_open = np.zeros(n)
        ha_high = np.zeros(n)
        ha_low = np.zeros(n)

        ha_close[0] = (o[0] + h[0] + l[0] + c[0]) / 4
        ha_open[0] = (o[0] + c[0]) / 2
        ha_high[0] = h[0]
        ha_low[0] = l[0]

        for i in range(1, n):
            ha_close[i] = (o[i] + h[i] + l[i] + c[i]) / 4
            ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2
            ha_high[i] = max(h[i], ha_open[i], ha_close[i])
            ha_low[i] = min(l[i], ha_open[i], ha_close[i])

        result["ha_open"] = ha_open
        result["ha_high"] = ha_high
        result["ha_low"] = ha_low
        result["ha_close"] = ha_close

        # ── Candle classification ──
        ha_body = ha_close - ha_open
        ha_range = ha_high - ha_low
        ha_range_safe = np.where(ha_range > 0, ha_range, 1.0)

        # Bullish: close > open, strong = no/tiny lower wick
        is_bull = ha_close > ha_open
        lower_wick_pct = np.where(is_bull, (ha_open - ha_low) / ha_range_safe, 0)
        strong_bull = is_bull & (lower_wick_pct <= wick_tol)

        # Bearish: close < open, strong = no/tiny upper wick
        is_bear = ha_close < ha_open
        upper_wick_pct = np.where(is_bear, (ha_high - ha_open) / ha_range_safe, 0)
        strong_bear = is_bear & (upper_wick_pct <= wick_tol)

        result["strong_bull"] = strong_bull.astype(int)
        result["strong_bear"] = strong_bear.astype(int)

        # ── Consecutive strong candle count ──
        bull_streak = np.zeros(n, dtype=int)
        bear_streak = np.zeros(n, dtype=int)

        for i in range(n):
            if strong_bull[i]:
                bull_streak[i] = (bull_streak[i - 1] if i > 0 else 0) + 1
            if strong_bear[i]:
                bear_streak[i] = (bear_streak[i - 1] if i > 0 else 0) + 1

        result["bull_streak"] = bull_streak
        result["bear_streak"] = bear_streak

        # ── EMA trend filter ──
        ema_f = ema(df, period=ema_fast_p).values
        ema_s = ema(df, period=ema_slow_p).values
        result["ema_fast"] = ema_f
        result["ema_slow"] = ema_s

        ema_bull = ema_f > ema_s
        ema_bear = ema_f < ema_s

        # ── ATR ──
        atr_values = atr(df, period=atr_period)
        result["atr"] = atr_values
        atr_vals = atr_values.values

        # ── Signal generation with position tracking ──
        signals = np.zeros(n, dtype=int)
        in_position = 0
        entry_price = 0.0
        peak_price = 0.0

        sl_mult = self.params["sl_atr_multiplier"]
        trail_mult = self.params["trail_atr_multiplier"]

        for i in range(consec, n):
            if in_position == 0:
                # Long entry
                can_long = bull_streak[i] >= consec
                if require_ema:
                    can_long = can_long and ema_bull[i]

                # Short entry
                can_short = bear_streak[i] >= consec
                if require_ema:
                    can_short = can_short and ema_bear[i]

                if can_long:
                    signals[i] = 1
                    in_position = 1
                    entry_price = c[i]
                    peak_price = c[i]
                elif can_short:
                    signals[i] = -1
                    in_position = -1
                    entry_price = c[i]
                    peak_price = c[i]

            elif in_position == 1:
                peak_price = max(peak_price, c[i])

                # Exit: reversal candle (bearish HA) or EMA cross down
                reversal = is_bear[i]
                ema_exit = require_ema and ema_bear[i]
                sl_hit = not np.isnan(atr_vals[i]) and c[i] < entry_price - atr_vals[i] * sl_mult
                trail_hit = not np.isnan(atr_vals[i]) and c[i] < peak_price - atr_vals[i] * trail_mult

                if reversal or ema_exit or sl_hit or trail_hit:
                    signals[i] = -1
                    in_position = 0

            elif in_position == -1:
                peak_price = min(peak_price, c[i])

                reversal = is_bull[i]
                ema_exit = require_ema and ema_bull[i]
                sl_hit = not np.isnan(atr_vals[i]) and c[i] > entry_price + atr_vals[i] * sl_mult
                trail_hit = not np.isnan(atr_vals[i]) and c[i] > peak_price + atr_vals[i] * trail_mult

                if reversal or ema_exit or sl_hit or trail_hit:
                    signals[i] = 1
                    in_position = 0

        result["signal"] = signals

        # SL/TP overlay
        tp_mult = self.params["tp_atr_multiplier"]
        close_s = df["close"]
        result.loc[result["signal"] == 1, "sl_level"] = close_s - atr_values * sl_mult
        result.loc[result["signal"] == 1, "tp_level"] = close_s + atr_values * tp_mult
        result.loc[result["signal"] == -1, "sl_level"] = close_s + atr_values * sl_mult
        result.loc[result["signal"] == -1, "tp_level"] = close_s - atr_values * tp_mult

        # Clean up
        result["signal"] = result["signal"].fillna(0)
        for col in result.select_dtypes(include=[np.number]).columns:
            if col != "signal":
                result[col] = result[col].ffill().fillna(0)

        return result


register_strategy("heikin_ashi_momentum", HeikinAshiMomentumStrategy)
