"""Regime Detection Meta-Strategy.

Classifies market into regimes (trending-up, trending-down, mean-reverting,
high-volatility) using a Gaussian Mixture Model approach, then selects the
optimal strategy class for the current regime.

This is a meta-strategy: it doesn't generate raw signals itself but
dynamically switches between trend-following and mean-reversion logic
based on detected market regime.

Best for: Adaptive trading across varying market conditions, reducing
drawdowns from applying wrong strategy to wrong regime.
"""

import numpy as np
import pandas as pd

from quantdash.features.indicators import atr, rsi, bollinger_bands, ema, sma
from quantdash.strategies.base import Strategy
from quantdash.strategies.registry import register_strategy


class RegimeDetectionStrategy(Strategy):
    """
    Market regime detection meta-strategy.

    Classifies market into 4 regimes using observable features:
      1. BULL_TREND: Strong uptrend (high ADX, +DI > -DI, price > EMA)
      2. BEAR_TREND: Strong downtrend (high ADX, -DI > +DI, price < EMA)
      3. MEAN_REVERT: Range-bound/choppy (low ADX, BB squeeze, RSI oscillating)
      4. HIGH_VOL: Regime transition / crisis (high ATR expansion, vol spike)

    For each regime, applies appropriate sub-logic:
      - BULL_TREND: Trend-following long entries on pullbacks to EMA
      - BEAR_TREND: Trend-following short entries on rallies to EMA
      - MEAN_REVERT: Bollinger Band mean-reversion entries
      - HIGH_VOL: Flat / tight stops only, wait for regime clarity

    Regime is determined by a rolling feature vector scored against
    characteristic profiles (no external ML library needed).
    """

    name = "regime_detection"
    description = "Adaptive regime-switching meta-strategy"
    default_params = {
        "regime_lookback": 40,        # Bars to classify regime
        "adx_period": 14,             # ADX for trend strength
        "ema_period": 50,             # EMA for trend direction
        "bb_period": 20,              # Bollinger Band period
        "bb_std": 2.0,                # BB standard deviations
        "rsi_period": 14,             # RSI period
        "vol_lookback": 20,           # Volatility measurement window
        "vol_expansion_mult": 1.5,    # ATR expansion threshold
        "adx_trend_threshold": 25,    # ADX > this = trending
        "adx_range_threshold": 15,    # ADX < this = ranging
        # Trend mode params
        "trend_pullback_pct": 0.02,   # Enter on 2% pullback to EMA
        "trend_ema_fast": 12,
        "trend_ema_slow": 26,
        # Mean reversion params
        "mr_entry_band": 0.9,        # Enter at 90% of BB distance
        "mr_exit_band": 0.3,         # Exit at 30% of BB distance
        # Risk
        "atr_period": 14,
        "sl_atr_multiplier": 2.0,
        "tp_atr_multiplier": 3.5,
        "trail_atr_multiplier": 1.5,
    }

    # Regime constants
    BULL_TREND = 1
    BEAR_TREND = 2
    MEAN_REVERT = 3
    HIGH_VOL = 4

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate regime-adaptive signals."""
        regime_lb = self.params["regime_lookback"]
        adx_period = self.params["adx_period"]
        ema_period = self.params["ema_period"]
        bb_period = self.params["bb_period"]
        bb_std = self.params["bb_std"]
        rsi_period = self.params["rsi_period"]
        vol_lb = self.params["vol_lookback"]
        vol_exp = self.params["vol_expansion_mult"]
        adx_trend_th = self.params["adx_trend_threshold"]
        adx_range_th = self.params["adx_range_threshold"]
        atr_period = self.params["atr_period"]

        result = df.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        n = len(df)

        # ── Compute features ──

        # ADX
        adx_values, plus_di, minus_di = self._compute_adx(df, adx_period)
        result["adx"] = adx_values
        result["plus_di"] = plus_di
        result["minus_di"] = minus_di

        # EMA
        ema_values = ema(df, period=ema_period)
        result["regime_ema"] = ema_values

        # Bollinger Bands
        bb = bollinger_bands(df, period=bb_period, std_dev=bb_std)
        bb_upper = bb["upper"]
        bb_lower = bb["lower"]
        bb_mid = bb["middle"]
        bb_width = (bb_upper - bb_lower) / (bb_mid + 1e-10)
        result["bb_upper"] = bb_upper
        result["bb_lower"] = bb_lower
        result["bb_mid"] = bb_mid
        result["bb_width"] = bb_width

        # RSI
        rsi_values = rsi(df, period=rsi_period)
        result["rsi"] = rsi_values

        # ATR and volatility
        atr_values = atr(df, period=atr_period)
        atr_ma = atr_values.rolling(vol_lb).mean()
        atr_expansion = atr_values / (atr_ma + 1e-10)
        result["atr"] = atr_values
        result["atr_expansion"] = atr_expansion

        # Fast/slow EMA for trend mode
        ema_fast = ema(df, period=self.params["trend_ema_fast"])
        ema_slow = ema(df, period=self.params["trend_ema_slow"])

        # ── Classify regime per bar ──
        regimes = np.full(n, self.MEAN_REVERT, dtype=int)

        for i in range(max(regime_lb, adx_period * 2, bb_period), n):
            adx_val = adx_values.iloc[i]
            pdi = plus_di.iloc[i]
            mdi = minus_di.iloc[i]
            price_vs_ema = close.iloc[i] / (ema_values.iloc[i] + 1e-10) - 1
            atr_exp = atr_expansion.iloc[i]

            if np.isnan(adx_val) or np.isnan(atr_exp):
                continue

            # High volatility regime: ATR expanding rapidly
            if atr_exp > vol_exp:
                regimes[i] = self.HIGH_VOL
            # Strong uptrend
            elif adx_val > adx_trend_th and pdi > mdi and price_vs_ema > 0:
                regimes[i] = self.BULL_TREND
            # Strong downtrend
            elif adx_val > adx_trend_th and mdi > pdi and price_vs_ema < 0:
                regimes[i] = self.BEAR_TREND
            # Range-bound
            elif adx_val < adx_range_th:
                regimes[i] = self.MEAN_REVERT
            # Moderate trend
            elif adx_val >= adx_range_th:
                if pdi > mdi:
                    regimes[i] = self.BULL_TREND
                else:
                    regimes[i] = self.BEAR_TREND

        result["regime"] = regimes

        # ── Generate signals per regime ──
        signals = np.zeros(n, dtype=int)
        close_vals = close.values
        ema_vals = ema_values.values
        bb_upper_vals = bb_upper.values
        bb_lower_vals = bb_lower.values
        bb_mid_vals = bb_mid.values
        ema_f_vals = ema_fast.values
        ema_s_vals = ema_slow.values
        atr_vals = atr_values.values
        rsi_vals = rsi_values.values

        in_position = 0
        entry_price = 0.0
        peak_price = 0.0
        position_regime = 0

        sl_mult = self.params["sl_atr_multiplier"]
        trail_mult = self.params["trail_atr_multiplier"]
        pullback_pct = self.params["trend_pullback_pct"]
        mr_entry = self.params["mr_entry_band"]
        mr_exit = self.params["mr_exit_band"]

        for i in range(max(regime_lb, adx_period * 2, bb_period), n):
            regime = regimes[i]

            if in_position == 0:
                if regime == self.BULL_TREND:
                    # Enter long on pullback toward EMA
                    if not np.isnan(ema_vals[i]):
                        dist = (close_vals[i] - ema_vals[i]) / (ema_vals[i] + 1e-10)
                        # Price pulled back near EMA and fast EMA still above slow
                        if -pullback_pct <= dist <= pullback_pct and ema_f_vals[i] > ema_s_vals[i]:
                            signals[i] = 1
                            in_position = 1
                            entry_price = close_vals[i]
                            peak_price = close_vals[i]
                            position_regime = regime

                elif regime == self.BEAR_TREND:
                    # Enter short on rally toward EMA
                    if not np.isnan(ema_vals[i]):
                        dist = (close_vals[i] - ema_vals[i]) / (ema_vals[i] + 1e-10)
                        if -pullback_pct <= dist <= pullback_pct and ema_f_vals[i] < ema_s_vals[i]:
                            signals[i] = -1
                            in_position = -1
                            entry_price = close_vals[i]
                            peak_price = close_vals[i]
                            position_regime = regime

                elif regime == self.MEAN_REVERT:
                    if not np.isnan(bb_lower_vals[i]) and not np.isnan(bb_upper_vals[i]):
                        bb_range = bb_upper_vals[i] - bb_lower_vals[i]
                        if bb_range > 0:
                            pct_b = (close_vals[i] - bb_lower_vals[i]) / bb_range
                            # Long at lower band
                            if pct_b < (1 - mr_entry) and not np.isnan(rsi_vals[i]) and rsi_vals[i] < 35:
                                signals[i] = 1
                                in_position = 1
                                entry_price = close_vals[i]
                                peak_price = close_vals[i]
                                position_regime = regime
                            # Short at upper band
                            elif pct_b > mr_entry and not np.isnan(rsi_vals[i]) and rsi_vals[i] > 65:
                                signals[i] = -1
                                in_position = -1
                                entry_price = close_vals[i]
                                peak_price = close_vals[i]
                                position_regime = regime

                # HIGH_VOL: stay flat

            elif in_position == 1:
                peak_price = max(peak_price, close_vals[i])

                should_exit = False

                # Regime-specific exit
                if position_regime in (self.BULL_TREND, self.BEAR_TREND):
                    # Trend mode: exit on EMA cross down
                    if ema_f_vals[i] < ema_s_vals[i]:
                        should_exit = True
                elif position_regime == self.MEAN_REVERT:
                    # MR mode: exit near middle band or upper band
                    if not np.isnan(bb_mid_vals[i]):
                        bb_range = bb_upper_vals[i] - bb_lower_vals[i]
                        if bb_range > 0:
                            pct_b = (close_vals[i] - bb_lower_vals[i]) / bb_range
                            if pct_b > mr_exit and pct_b > 0.5:
                                should_exit = True

                # Regime change to HIGH_VOL: tighten or exit
                if regime == self.HIGH_VOL:
                    should_exit = True

                # SL / trailing stop
                if not np.isnan(atr_vals[i]):
                    if close_vals[i] < entry_price - atr_vals[i] * sl_mult:
                        should_exit = True
                    if close_vals[i] < peak_price - atr_vals[i] * trail_mult:
                        should_exit = True

                if should_exit:
                    signals[i] = -1
                    in_position = 0

            elif in_position == -1:
                peak_price = min(peak_price, close_vals[i])

                should_exit = False

                if position_regime in (self.BULL_TREND, self.BEAR_TREND):
                    if ema_f_vals[i] > ema_s_vals[i]:
                        should_exit = True
                elif position_regime == self.MEAN_REVERT:
                    if not np.isnan(bb_mid_vals[i]):
                        bb_range = bb_upper_vals[i] - bb_lower_vals[i]
                        if bb_range > 0:
                            pct_b = (close_vals[i] - bb_lower_vals[i]) / bb_range
                            if pct_b < mr_exit or pct_b < 0.5:
                                should_exit = True

                if regime == self.HIGH_VOL:
                    should_exit = True

                if not np.isnan(atr_vals[i]):
                    if close_vals[i] > entry_price + atr_vals[i] * sl_mult:
                        should_exit = True
                    if close_vals[i] > peak_price + atr_vals[i] * trail_mult:
                        should_exit = True

                if should_exit:
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

    @staticmethod
    def _compute_adx(df: pd.DataFrame, period: int) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Compute ADX, +DI, -DI."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        plus_dm = (high - high.shift(1)).clip(lower=0)
        minus_dm = (low.shift(1) - low).clip(lower=0)
        plus_dm = pd.Series(np.where(plus_dm > minus_dm, plus_dm, 0), index=df.index)
        minus_dm = pd.Series(np.where(minus_dm > plus_dm, minus_dm, 0), index=df.index)

        # Smoothed (Wilder's method)
        atr_smooth = true_range.ewm(span=period, adjust=False).mean()
        plus_dm_smooth = plus_dm.ewm(span=period, adjust=False).mean()
        minus_dm_smooth = minus_dm.ewm(span=period, adjust=False).mean()

        plus_di = 100 * plus_dm_smooth / (atr_smooth + 1e-10)
        minus_di = 100 * minus_dm_smooth / (atr_smooth + 1e-10)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(span=period, adjust=False).mean()

        return adx, plus_di, minus_di


register_strategy("regime_detection", RegimeDetectionStrategy)
