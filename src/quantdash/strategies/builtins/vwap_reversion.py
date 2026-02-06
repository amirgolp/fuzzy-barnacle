"""VWAP Reversion Strategy.

Trades mean-reversion to the Volume Weighted Average Price.
Enters when price deviates significantly from VWAP (measured in
rolling standard deviations), exits when price reverts toward VWAP.

Best for: Intraday and short-term swing trading on liquid instruments.
"""

import numpy as np
import pandas as pd

from quantdash.features.indicators import atr
from quantdash.strategies.base import Strategy
from quantdash.strategies.registry import register_strategy


class VWAPReversionStrategy(Strategy):
    """
    VWAP mean-reversion strategy with band-based entries.

    Uses rolling VWAP with standard deviation bands (analogous to
    Bollinger Bands around VWAP). Enters long when price falls below
    the lower band, short when price rises above the upper band.
    Exits at VWAP or opposing band. Includes volume confirmation
    and ATR-based risk management.
    """

    name = "vwap_reversion"
    description = "VWAP mean-reversion with deviation bands"
    default_params = {
        "vwap_period": 20,          # Rolling VWAP lookback
        "band_mult": 2.0,           # Std dev multiplier for entry bands
        "exit_band_mult": 0.5,      # Std dev multiplier for exit (near VWAP)
        "volume_confirm": True,     # Require above-avg volume on entry
        "volume_lookback": 20,      # Volume average lookback
        "volume_threshold": 1.2,    # Volume must be X times average
        "atr_period": 14,
        "sl_atr_multiplier": 2.0,
        "tp_atr_multiplier": 3.0,
        "trail_atr_multiplier": 1.5,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate VWAP reversion signals."""
        period = self.params["vwap_period"]
        band_mult = self.params["band_mult"]
        exit_mult = self.params["exit_band_mult"]
        vol_confirm = self.params["volume_confirm"]
        vol_lookback = self.params["volume_lookback"]
        vol_threshold = self.params["volume_threshold"]
        atr_period = self.params["atr_period"]

        result = df.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"] if "volume" in df.columns else pd.Series(1, index=df.index)

        # Rolling VWAP: sum(typical_price * volume) / sum(volume) over period
        typical_price = (high + low + close) / 3
        tp_vol = typical_price * volume
        rolling_tp_vol = tp_vol.rolling(window=period).sum()
        rolling_vol = volume.rolling(window=period).sum()
        rolling_vwap = rolling_tp_vol / (rolling_vol + 1e-10)

        # Rolling standard deviation of price around VWAP
        deviation = close - rolling_vwap
        rolling_std = deviation.rolling(window=period).std()

        # VWAP bands
        upper_band = rolling_vwap + band_mult * rolling_std
        lower_band = rolling_vwap - band_mult * rolling_std
        exit_upper = rolling_vwap + exit_mult * rolling_std
        exit_lower = rolling_vwap - exit_mult * rolling_std

        result["vwap"] = rolling_vwap
        result["vwap_upper"] = upper_band
        result["vwap_lower"] = lower_band
        result["vwap_deviation"] = deviation / (rolling_std + 1e-10)

        # Volume filter
        avg_volume = volume.rolling(window=vol_lookback).mean()
        high_volume = volume > (avg_volume * vol_threshold) if vol_confirm else pd.Series(True, index=df.index)
        result["volume_ratio"] = volume / (avg_volume + 1e-10)

        # ATR for risk management
        atr_values = atr(df, period=atr_period)
        result["atr"] = atr_values

        # Entry signals
        long_entry = (close <= lower_band) & high_volume
        short_entry = (close >= upper_band) & high_volume

        # Position tracking with entry/exit logic
        signals = np.zeros(len(df), dtype=int)
        close_vals = close.values
        exit_upper_vals = exit_upper.values
        exit_lower_vals = exit_lower.values
        vwap_vals = rolling_vwap.values
        atr_vals = atr_values.values

        in_position = 0  # 0=flat, 1=long, -1=short
        entry_price = 0.0
        peak_price = 0.0

        for i in range(len(df)):
            if in_position == 0:
                if long_entry.iloc[i]:
                    signals[i] = 1
                    in_position = 1
                    entry_price = close_vals[i]
                    peak_price = close_vals[i]
                elif short_entry.iloc[i]:
                    signals[i] = -1
                    in_position = -1
                    entry_price = close_vals[i]
                    peak_price = close_vals[i]
            elif in_position == 1:
                peak_price = max(peak_price, close_vals[i])
                # Exit long: price reverted to VWAP zone
                if close_vals[i] >= exit_upper_vals[i] if not np.isnan(exit_upper_vals[i]) else False:
                    signals[i] = -1
                    in_position = 0
                # Stop loss
                elif not np.isnan(atr_vals[i]) and close_vals[i] < entry_price - atr_vals[i] * self.params["sl_atr_multiplier"]:
                    signals[i] = -1
                    in_position = 0
                # Trailing stop
                elif not np.isnan(atr_vals[i]) and close_vals[i] < peak_price - atr_vals[i] * self.params["trail_atr_multiplier"]:
                    signals[i] = -1
                    in_position = 0
            elif in_position == -1:
                peak_price = min(peak_price, close_vals[i])
                # Exit short: price reverted to VWAP zone
                if close_vals[i] <= exit_lower_vals[i] if not np.isnan(exit_lower_vals[i]) else False:
                    signals[i] = 1
                    in_position = 0
                # Stop loss
                elif not np.isnan(atr_vals[i]) and close_vals[i] > entry_price + atr_vals[i] * self.params["sl_atr_multiplier"]:
                    signals[i] = 1
                    in_position = 0
                # Trailing stop
                elif not np.isnan(atr_vals[i]) and close_vals[i] > peak_price + atr_vals[i] * self.params["trail_atr_multiplier"]:
                    signals[i] = 1
                    in_position = 0

        result["signal"] = signals

        # SL/TP levels for overlay
        sl_mult = self.params["sl_atr_multiplier"]
        tp_mult = self.params["tp_atr_multiplier"]
        result["sl_distance"] = atr_values * sl_mult
        result["tp_distance"] = atr_values * tp_mult

        result.loc[result["signal"] == 1, "sl_level"] = close - result["sl_distance"]
        result.loc[result["signal"] == 1, "tp_level"] = rolling_vwap  # TP at VWAP

        result.loc[result["signal"] == -1, "sl_level"] = close + result["sl_distance"]
        result.loc[result["signal"] == -1, "tp_level"] = rolling_vwap  # TP at VWAP

        # Clean up
        result["signal"] = result["signal"].fillna(0)
        for col in result.select_dtypes(include=[np.number]).columns:
            if col != "signal":
                result[col] = result[col].ffill().fillna(0)

        return result


register_strategy("vwap_reversion", VWAPReversionStrategy)
