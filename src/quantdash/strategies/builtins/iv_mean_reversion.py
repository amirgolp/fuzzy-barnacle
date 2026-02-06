"""
IV Mean Reversion Strategy

Uses implied volatility (approximated from historical volatility) to identify
when volatility is at extremes. Trades mean reversion when IV rank is high
(expecting volatility contraction and price consolidation) or low (expecting
volatility expansion and trend continuation).

This is a spot-trading strategy that uses derivatives concepts for edge.

Intraweek/Intramonth friendly with max_holding_bars parameter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantdash.strategies.base import Strategy
from quantdash.strategies.registry import register_strategy


class IVMeanReversionStrategy(Strategy):
    """
    IV-based mean reversion strategy.

    Logic:
    - Calculate historical volatility (HV) as a proxy for IV
    - Compute IV Rank: where current IV sits vs. past year (0-100)
    - When IV Rank > 80 (high vol): sell premium equivalent = short mean reversion
    - When IV Rank < 20 (low vol): prepare for expansion, trend-following mode

    For spot trading:
    - High IV Rank + RSI overbought: Short (price likely to consolidate/drop)
    - High IV Rank + RSI oversold: Long (volatility crush + mean reversion)
    - Low IV Rank: Stay with trend, tighter stops

    Params:
        hv_period: Historical volatility lookback (default: 20)
        iv_rank_period: Period for IV rank calculation (default: 252)
        iv_high_threshold: IV rank threshold for high vol (default: 80)
        iv_low_threshold: IV rank threshold for low vol (default: 20)
        rsi_period: RSI period for confirmation (default: 14)
        rsi_overbought: RSI overbought level (default: 70)
        rsi_oversold: RSI oversold level (default: 30)
    """

    name = "iv_mean_reversion"
    description = "IV-based mean reversion using volatility extremes"
    default_params = {
        "hv_period": 20,
        "iv_rank_period": 252,
        "iv_high_threshold": 80,
        "iv_low_threshold": 20,
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        hv_period = p["hv_period"]
        iv_rank_period = p["iv_rank_period"]
        iv_high = p["iv_high_threshold"]
        iv_low = p["iv_low_threshold"]
        rsi_period = p["rsi_period"]
        rsi_ob = p["rsi_overbought"]
        rsi_os = p["rsi_oversold"]

        close = df["close"]

        # Calculate Historical Volatility (annualized)
        log_returns = np.log(close / close.shift(1))
        hv = log_returns.rolling(window=hv_period).std() * np.sqrt(252)

        # Calculate IV Rank (percentile of current HV vs past year)
        iv_rank = hv.rolling(window=iv_rank_period).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) * 100
            if x.max() != x.min() else 50,
            raw=False
        )

        # Calculate RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        # Calculate expected move based on HV
        # Expected 1-week move = HV * sqrt(5/252) * close
        expected_move_pct = hv * np.sqrt(5 / 252) * 100

        # Generate signals
        signal = pd.Series(0, index=df.index)

        # High IV conditions - mean reversion mode
        high_iv = iv_rank > iv_high

        # Buy: High IV + RSI oversold (vol crush + bounce expected)
        buy_signal = high_iv & (rsi < rsi_os)

        # Sell: High IV + RSI overbought (vol crush + pullback expected)
        sell_signal = high_iv & (rsi > rsi_ob)

        # Low IV conditions - breakout/trend mode
        low_iv = iv_rank < iv_low

        # In low IV, buy breakouts above recent high
        recent_high = close.rolling(window=20).max()
        breakout_up = (close > recent_high.shift(1)) & low_iv

        # Combine signals
        signal = signal.where(~buy_signal, 1)
        signal = signal.where(~breakout_up, 1)
        signal = signal.where(~sell_signal, -1)

        result = pd.DataFrame(index=df.index)
        result["signal"] = signal.fillna(0).astype(int)
        result["iv_rank"] = iv_rank
        result["hv"] = hv
        result["rsi"] = rsi
        result["expected_move_pct"] = expected_move_pct

        # Calculate dynamic SL/TP based on expected move
        result["sl_level"] = close * (1 - expected_move_pct / 100)
        result["tp_level"] = close * (1 + expected_move_pct / 100)

        return result


# Register the strategy
register_strategy("iv_mean_reversion", IVMeanReversionStrategy)
