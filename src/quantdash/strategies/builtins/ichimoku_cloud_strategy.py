"""Ichimoku Cloud Strategy.

Full Ichimoku Kinko Hyo trading system using all five components:
Tenkan-sen, Kijun-sen, Senkou Span A/B (cloud), and Chikou Span.

Generates signals from TK cross, price-cloud relationship, and
Chikou Span confirmation for high-probability trend entries.

Best for: Trending markets, daily/4H timeframes, forex and indices.
"""

import numpy as np
import pandas as pd

from quantdash.features.indicators import ichimoku_cloud, atr
from quantdash.strategies.base import Strategy
from quantdash.strategies.registry import register_strategy


class IchimokuCloudStrategy(Strategy):
    """
    Ichimoku Cloud trend-following strategy.

    Entry conditions (long):
    1. Tenkan-sen crosses above Kijun-sen (TK cross bullish)
    2. Price is above the cloud (Senkou Span A & B)
    3. Chikou Span is above price from 26 periods ago (confirmation)

    Entry conditions (short): inverse of above.

    Exits on opposing TK cross or price entering the cloud.
    Cloud acts as dynamic support/resistance.
    """

    name = "ichimoku_cloud"
    description = "Ichimoku Cloud trend system with TK cross and cloud filter"
    default_params = {
        "tenkan_period": 9,
        "kijun_period": 26,
        "senkou_b_period": 52,
        "require_chikou": True,      # Require Chikou Span confirmation
        "require_cloud_clear": True,  # Price must be clearly above/below cloud
        "cloud_buffer_pct": 0.001,   # Min distance from cloud edge (0.1%)
        "atr_period": 14,
        "sl_atr_multiplier": 2.5,
        "tp_atr_multiplier": 4.0,
        "trail_atr_multiplier": 2.0,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate Ichimoku-based signals."""
        tenkan_p = self.params["tenkan_period"]
        kijun_p = self.params["kijun_period"]
        senkou_b_p = self.params["senkou_b_period"]
        require_chikou = self.params["require_chikou"]
        require_cloud = self.params["require_cloud_clear"]
        cloud_buffer = self.params["cloud_buffer_pct"]
        atr_period = self.params["atr_period"]

        result = df.copy()
        close = df["close"]
        n = len(df)

        # Compute Ichimoku components
        ichi = ichimoku_cloud(df, tenkan_p, kijun_p, senkou_b_p)
        tenkan = ichi["tenkan_sen"]
        kijun = ichi["kijun_sen"]
        span_a = ichi["senkou_span_a"]
        span_b = ichi["senkou_span_b"]
        chikou = ichi["chikou_span"]

        result["tenkan_sen"] = tenkan
        result["kijun_sen"] = kijun
        result["senkou_span_a"] = span_a
        result["senkou_span_b"] = span_b
        result["chikou_span"] = chikou

        # Cloud top/bottom
        cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
        cloud_bottom = pd.concat([span_a, span_b], axis=1).min(axis=1)
        result["cloud_top"] = cloud_top
        result["cloud_bottom"] = cloud_bottom

        # ATR
        atr_values = atr(df, period=atr_period)
        result["atr"] = atr_values

        # ── Condition detection ──

        # TK cross
        tk_bull_cross = (tenkan > kijun) & (tenkan.shift(1) <= kijun.shift(1))
        tk_bear_cross = (tenkan < kijun) & (tenkan.shift(1) >= kijun.shift(1))

        # Price vs cloud
        above_cloud = close > cloud_top * (1 + cloud_buffer)
        below_cloud = close < cloud_bottom * (1 - cloud_buffer)
        in_cloud = ~above_cloud & ~below_cloud

        # Chikou Span confirmation: chikou (current close shifted back) vs
        # price from kijun_period bars ago
        chikou_bull = pd.Series(False, index=df.index)
        chikou_bear = pd.Series(False, index=df.index)
        if require_chikou:
            past_close = close.shift(kijun_p)
            chikou_bull = close > past_close
            chikou_bear = close < past_close
        else:
            chikou_bull = pd.Series(True, index=df.index)
            chikou_bear = pd.Series(True, index=df.index)

        # Tenkan vs Kijun ongoing trend
        tk_bullish = tenkan > kijun
        tk_bearish = tenkan < kijun

        # ── Signal generation with position tracking ──

        signals = np.zeros(n, dtype=int)
        close_vals = close.values
        cloud_top_vals = cloud_top.values
        cloud_bottom_vals = cloud_bottom.values
        atr_vals = atr_values.values

        in_position = 0
        entry_price = 0.0
        peak_price = 0.0

        for i in range(1, n):
            if in_position == 0:
                # Long entry: TK bull cross + above cloud + chikou confirms
                can_long = tk_bull_cross.iloc[i]
                if require_cloud:
                    can_long = can_long and above_cloud.iloc[i]
                if require_chikou:
                    can_long = can_long and chikou_bull.iloc[i]

                # Short entry: TK bear cross + below cloud + chikou confirms
                can_short = tk_bear_cross.iloc[i]
                if require_cloud:
                    can_short = can_short and below_cloud.iloc[i]
                if require_chikou:
                    can_short = can_short and chikou_bear.iloc[i]

                if can_long:
                    signals[i] = 1
                    in_position = 1
                    entry_price = close_vals[i]
                    peak_price = close_vals[i]
                elif can_short:
                    signals[i] = -1
                    in_position = -1
                    entry_price = close_vals[i]
                    peak_price = close_vals[i]

            elif in_position == 1:
                peak_price = max(peak_price, close_vals[i])

                # Exit long: TK bear cross OR price enters/goes below cloud
                exit_tk = tk_bear_cross.iloc[i]
                exit_cloud = not np.isnan(cloud_bottom_vals[i]) and close_vals[i] < cloud_bottom_vals[i]

                # Stop loss
                sl_hit = (not np.isnan(atr_vals[i]) and
                          close_vals[i] < entry_price - atr_vals[i] * self.params["sl_atr_multiplier"])

                # Trailing stop
                trail_hit = (not np.isnan(atr_vals[i]) and
                             close_vals[i] < peak_price - atr_vals[i] * self.params["trail_atr_multiplier"])

                if exit_tk or exit_cloud or sl_hit or trail_hit:
                    signals[i] = -1
                    in_position = 0

            elif in_position == -1:
                peak_price = min(peak_price, close_vals[i])

                # Exit short: TK bull cross OR price enters/goes above cloud
                exit_tk = tk_bull_cross.iloc[i]
                exit_cloud = not np.isnan(cloud_top_vals[i]) and close_vals[i] > cloud_top_vals[i]

                sl_hit = (not np.isnan(atr_vals[i]) and
                          close_vals[i] > entry_price + atr_vals[i] * self.params["sl_atr_multiplier"])

                trail_hit = (not np.isnan(atr_vals[i]) and
                             close_vals[i] > peak_price + atr_vals[i] * self.params["trail_atr_multiplier"])

                if exit_tk or exit_cloud or sl_hit or trail_hit:
                    signals[i] = 1
                    in_position = 0

        result["signal"] = signals

        # SL/TP overlay levels
        sl_mult = self.params["sl_atr_multiplier"]
        tp_mult = self.params["tp_atr_multiplier"]

        result.loc[result["signal"] == 1, "sl_level"] = close - atr_values * sl_mult
        result.loc[result["signal"] == 1, "tp_level"] = close + atr_values * tp_mult

        result.loc[result["signal"] == -1, "sl_level"] = close + atr_values * sl_mult
        result.loc[result["signal"] == -1, "tp_level"] = close - atr_values * tp_mult

        # Clean up
        result["signal"] = result["signal"].fillna(0)
        for col in result.select_dtypes(include=[np.number]).columns:
            if col != "signal":
                result[col] = result[col].ffill().fillna(0)

        return result


register_strategy("ichimoku_cloud", IchimokuCloudStrategy)
