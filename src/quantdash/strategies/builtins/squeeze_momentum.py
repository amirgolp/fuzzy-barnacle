"""Squeeze Momentum Indicator strategy (ported from LazyBear's TradingView Pine Script)."""

import numpy as np
import pandas as pd

from quantdash.strategies.base import Strategy
from quantdash.strategies.registry import register_strategy


class SqueezeMomentum(Strategy):
    """
    Squeeze Momentum Indicator strategy by LazyBear.

    Detects when Bollinger Bands squeeze inside Keltner Channels (low volatility),
    then trades the momentum direction when the squeeze releases.

    Buy when squeeze releases and momentum is positive & increasing.
    Sell when squeeze releases and momentum is negative & decreasing.
    """

    name = "squeeze_momentum"
    description = "Squeeze Momentum (BB/KC squeeze) strategy"
    default_params = {
        "bb_length": 20,
        "bb_mult": 2.0,
        "kc_length": 20,
        "kc_mult": 1.5,
        "use_true_range": True,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        bb_length = self.params["bb_length"]
        bb_mult = self.params["bb_mult"]
        kc_length = self.params["kc_length"]
        kc_mult = self.params["kc_mult"]
        use_tr = self.params["use_true_range"]

        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Bollinger Bands
        basis = close.rolling(window=bb_length).mean()
        dev = bb_mult * close.rolling(window=bb_length).std()
        upper_bb = basis + dev
        lower_bb = basis - dev

        # Keltner Channels
        ma = close.rolling(window=kc_length).mean()
        if use_tr:
            prev_close = close.shift(1)
            tr = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ], axis=1).max(axis=1)
        else:
            tr = high - low
        range_ma = tr.rolling(window=kc_length).mean()
        upper_kc = ma + range_ma * kc_mult
        lower_kc = ma - range_ma * kc_mult

        # Squeeze states
        sqz_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)
        sqz_off = (lower_bb < lower_kc) & (upper_bb > upper_kc)

        # Momentum value: linear regression of (close - avg(avg(highest, lowest), sma))
        highest_high = high.rolling(window=kc_length).max()
        lowest_low = low.rolling(window=kc_length).min()
        sma_close = close.rolling(window=kc_length).mean()
        midline = ((highest_high + lowest_low) / 2 + sma_close) / 2
        delta = close - midline

        # Linear regression over kc_length periods
        momentum = _linreg(delta, kc_length)

        # Signal logic:
        # Buy when squeeze just released (was on, now off) and momentum > 0 and rising
        # Sell when squeeze just released and momentum < 0 and falling
        prev_sqz_on = sqz_on.shift(1).fillna(False)
        mom_rising = momentum > momentum.shift(1)
        mom_falling = momentum < momentum.shift(1)

        n = len(df)
        signal = np.zeros(n, dtype=int)

        for i in range(1, n):
            # Squeeze release: was on, now off or noSqz
            squeeze_released = prev_sqz_on.iloc[i] and not sqz_on.iloc[i]

            if squeeze_released:
                if not np.isnan(momentum.iloc[i]) and momentum.iloc[i] > 0 and mom_rising.iloc[i]:
                    signal[i] = 1  # Buy
                elif not np.isnan(momentum.iloc[i]) and momentum.iloc[i] < 0 and mom_falling.iloc[i]:
                    signal[i] = -1  # Sell
            elif not np.isnan(momentum.iloc[i]):
                # Also generate signals on momentum zero-cross while squeeze is off
                prev_mom = momentum.iloc[i - 1] if i > 0 else 0
                if not sqz_on.iloc[i] and not np.isnan(prev_mom):
                    if momentum.iloc[i] > 0 and prev_mom <= 0:
                        signal[i] = 1  # Momentum crossed above zero
                    elif momentum.iloc[i] < 0 and prev_mom >= 0:
                        signal[i] = -1  # Momentum crossed below zero

        result = df.copy()
        result["signal"] = signal
        result["sqz_momentum"] = momentum
        result["sqz_on"] = sqz_on.astype(int)
        result["sqz_off"] = sqz_off.astype(int)

        return result


def _linreg(series: pd.Series, length: int) -> pd.Series:
    """Rolling linear regression value (endpoint) over length periods."""
    result = pd.Series(np.nan, index=series.index)
    values = series.values

    for i in range(length - 1, len(values)):
        window = values[i - length + 1:i + 1]
        if np.any(np.isnan(window)):
            continue
        x = np.arange(length, dtype=float)
        x_mean = x.mean()
        y_mean = window.mean()
        slope = np.sum((x - x_mean) * (window - y_mean)) / np.sum((x - x_mean) ** 2)
        intercept = y_mean - slope * x_mean
        result.iloc[i] = intercept + slope * (length - 1)

    return result


register_strategy("squeeze_momentum", SqueezeMomentum)
