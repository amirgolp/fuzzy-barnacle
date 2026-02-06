"""Order Flow Imbalance Strategy.

Infers buying/selling pressure from price-volume interaction without
requiring Level 2 data. Uses volume-at-price analysis, tick-rule
classification, and accumulation/distribution to detect institutional
order flow imbalance.

Best for: Liquid instruments with reliable volume data, intraday to
daily timeframes, stocks and ETFs.
"""

import numpy as np
import pandas as pd

from quantdash.features.indicators import atr, obv, money_flow_index, chaikin_money_flow
from quantdash.strategies.base import Strategy
from quantdash.strategies.registry import register_strategy


class OrderFlowImbalanceStrategy(Strategy):
    """
    Order flow imbalance strategy using volume analysis.

    Combines multiple volume-based signals to detect institutional
    accumulation (buying) or distribution (selling):

    1. Volume-price divergence: price flat/down but volume surging
       (accumulation) or price up but volume declining (distribution)
    2. Money Flow Index extremes with volume confirmation
    3. Chaikin Money Flow direction
    4. OBV trend vs price trend divergence
    5. Buy/sell volume ratio using close location within bar range

    Enters when multiple flow indicators align, exits on flow reversal.
    """

    name = "order_flow_imbalance"
    description = "Volume-based order flow imbalance detection"
    default_params = {
        "flow_period": 14,             # Period for flow calculations
        "mfi_period": 14,              # Money Flow Index period
        "cmf_period": 20,              # Chaikin Money Flow period
        "obv_ema_period": 20,          # EMA period for OBV smoothing
        "volume_lookback": 20,         # Volume analysis window
        "min_signals": 3,              # Min aligned flow signals to enter
        "mfi_oversold": 20,            # MFI oversold threshold
        "mfi_overbought": 80,          # MFI overbought threshold
        "volume_surge_mult": 1.5,      # Volume surge multiplier
        "atr_period": 14,
        "sl_atr_multiplier": 2.0,
        "tp_atr_multiplier": 3.5,
        "trail_atr_multiplier": 1.5,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate order flow imbalance signals."""
        flow_period = self.params["flow_period"]
        mfi_period = self.params["mfi_period"]
        cmf_period = self.params["cmf_period"]
        obv_ema_p = self.params["obv_ema_period"]
        vol_lookback = self.params["volume_lookback"]
        min_signals = self.params["min_signals"]
        mfi_os = self.params["mfi_oversold"]
        mfi_ob = self.params["mfi_overbought"]
        vol_surge = self.params["volume_surge_mult"]
        atr_period = self.params["atr_period"]

        result = df.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"] if "volume" in df.columns else pd.Series(1, index=df.index)
        n = len(df)

        # ── Component 1: Buy/Sell Volume Classification ──
        # Close Location Value: where did price close within the bar?
        bar_range = high - low
        clv = ((close - low) - (high - close)) / (bar_range + 1e-10)
        # clv: +1 = closed at high (buying pressure), -1 = closed at low (selling)

        buy_volume = volume * ((clv + 1) / 2)     # Estimated buy volume
        sell_volume = volume * ((1 - clv) / 2)     # Estimated sell volume

        rolling_buy = buy_volume.rolling(flow_period).sum()
        rolling_sell = sell_volume.rolling(flow_period).sum()
        flow_ratio = rolling_buy / (rolling_sell + 1e-10)  # >1 = net buying

        result["flow_ratio"] = flow_ratio
        result["clv"] = clv

        # ── Component 2: Money Flow Index ──
        mfi_vals = money_flow_index(df, period=mfi_period)
        result["mfi"] = mfi_vals

        # ── Component 3: Chaikin Money Flow ──
        cmf_vals = chaikin_money_flow(df, period=cmf_period)
        result["cmf"] = cmf_vals

        # ── Component 4: OBV Trend ──
        obv_vals = obv(df)
        obv_ema = obv_vals.ewm(span=obv_ema_p, adjust=False).mean()
        obv_above_ema = obv_vals > obv_ema
        result["obv"] = obv_vals
        result["obv_ema"] = obv_ema

        # ── Component 5: Volume-Price Divergence ──
        price_change = close.pct_change(flow_period)
        volume_change = volume.rolling(flow_period).mean() / volume.rolling(vol_lookback).mean() - 1

        # Accumulation: price flat/down but volume surging
        accumulation_div = (price_change < 0.01) & (volume_change > (vol_surge - 1))
        # Distribution: price up but volume declining
        distribution_div = (price_change > 0.01) & (volume_change < -(vol_surge - 1) * 0.5)

        result["price_change"] = price_change
        result["volume_change"] = volume_change

        # ── Component 6: Volume Surge Detection ──
        avg_vol = volume.rolling(vol_lookback).mean()
        vol_ratio = volume / (avg_vol + 1e-10)
        is_surge = vol_ratio > vol_surge
        result["volume_surge"] = is_surge.astype(int)

        # ATR
        atr_values = atr(df, period=atr_period)
        result["atr"] = atr_values

        # ── Composite Signal Scoring ──
        # Count bullish flow signals
        bull_count = np.zeros(n)
        bear_count = np.zeros(n)

        for i in range(max(flow_period, vol_lookback, cmf_period), n):
            bs = 0
            brs = 0

            # Flow ratio bullish (net buying)
            if not np.isnan(flow_ratio.iloc[i]) and flow_ratio.iloc[i] > 1.2:
                bs += 1
            elif not np.isnan(flow_ratio.iloc[i]) and flow_ratio.iloc[i] < 0.8:
                brs += 1

            # MFI signal
            if not np.isnan(mfi_vals.iloc[i]):
                if mfi_vals.iloc[i] < mfi_os:
                    bs += 1  # Oversold = potential accumulation bounce
                elif mfi_vals.iloc[i] > mfi_ob:
                    brs += 1  # Overbought = potential distribution drop

            # CMF direction
            if not np.isnan(cmf_vals.iloc[i]):
                if cmf_vals.iloc[i] > 0.05:
                    bs += 1
                elif cmf_vals.iloc[i] < -0.05:
                    brs += 1

            # OBV trend
            if obv_above_ema.iloc[i]:
                bs += 1
            else:
                brs += 1

            # Volume-price divergence
            if accumulation_div.iloc[i]:
                bs += 1
            if distribution_div.iloc[i]:
                brs += 1

            # Volume surge with direction
            if is_surge.iloc[i]:
                if clv.iloc[i] > 0.3:
                    bs += 1
                elif clv.iloc[i] < -0.3:
                    brs += 1

            bull_count[i] = bs
            bear_count[i] = brs

        result["bull_flow_count"] = bull_count
        result["bear_flow_count"] = bear_count

        # ── Position management ──
        signals = np.zeros(n, dtype=int)
        close_vals = close.values
        atr_vals = atr_values.values

        in_position = 0
        entry_price = 0.0
        peak_price = 0.0

        sl_mult = self.params["sl_atr_multiplier"]
        trail_mult = self.params["trail_atr_multiplier"]

        for i in range(max(flow_period, vol_lookback, cmf_period), n):
            if in_position == 0:
                if bull_count[i] >= min_signals and bull_count[i] > bear_count[i] + 1:
                    signals[i] = 1
                    in_position = 1
                    entry_price = close_vals[i]
                    peak_price = close_vals[i]
                elif bear_count[i] >= min_signals and bear_count[i] > bull_count[i] + 1:
                    signals[i] = -1
                    in_position = -1
                    entry_price = close_vals[i]
                    peak_price = close_vals[i]

            elif in_position == 1:
                peak_price = max(peak_price, close_vals[i])

                # Exit: flow reversed (more bearish than bullish)
                flow_exit = bear_count[i] > bull_count[i] + 1
                sl_hit = not np.isnan(atr_vals[i]) and close_vals[i] < entry_price - atr_vals[i] * sl_mult
                trail_hit = not np.isnan(atr_vals[i]) and close_vals[i] < peak_price - atr_vals[i] * trail_mult

                if flow_exit or sl_hit or trail_hit:
                    signals[i] = -1
                    in_position = 0

            elif in_position == -1:
                peak_price = min(peak_price, close_vals[i])

                flow_exit = bull_count[i] > bear_count[i] + 1
                sl_hit = not np.isnan(atr_vals[i]) and close_vals[i] > entry_price + atr_vals[i] * sl_mult
                trail_hit = not np.isnan(atr_vals[i]) and close_vals[i] > peak_price + atr_vals[i] * trail_mult

                if flow_exit or sl_hit or trail_hit:
                    signals[i] = 1
                    in_position = 0

        result["signal"] = signals

        # SL/TP overlay
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


register_strategy("order_flow_imbalance", OrderFlowImbalanceStrategy)
