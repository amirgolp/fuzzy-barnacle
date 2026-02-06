"""
Expected Move Strategy

Uses options-derived expected move calculations to determine if the current
price action is within or outside the market's "expected" range.

When price moves beyond the expected move:
- It may indicate a trend breakout (momentum)
- Or an overextension (mean reversion)

This strategy combines expected move with momentum confirmation for intraweek trades.

Intraweek/Intramonth friendly with max_holding_bars parameter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantdash.strategies.base import Strategy
from quantdash.strategies.registry import register_strategy


class ExpectedMoveStrategy(Strategy):
    """
    Expected Move Breakout/Reversion Strategy.

    Logic:
    - Calculate expected move from historical volatility
    - When price exceeds expected move with momentum confirmation: trend trade
    - When price exceeds expected move with exhaustion signals: fade the move

    The expected move formula (options-derived):
    EM = Stock Price × IV × sqrt(DTE / 365)

    For spot trading, we use HV as IV proxy and a fixed horizon (e.g., 5 days for intraweek).

    Params:
        lookback: Period for volatility calculation (default: 20)
        horizon_days: Expected move horizon in days (default: 5)
        breakout_threshold: Multiplier of expected move for breakout (default: 1.0)
        mean_reversion_threshold: Multiplier for mean reversion (default: 1.5)
        momentum_period: Period for momentum confirmation (default: 5)
        use_atr: Use ATR instead of HV for expected move (default: False)
    """

    name = "expected_move"
    description = "Trade based on expected move breakouts and reversions"
    default_params = {
        "lookback": 20,
        "horizon_days": 5,
        "breakout_threshold": 1.0,
        "mean_reversion_threshold": 1.5,
        "momentum_period": 5,
        "use_atr": False,
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        lookback = p["lookback"]
        horizon = p["horizon_days"]
        breakout_mult = p["breakout_threshold"]
        reversion_mult = p["mean_reversion_threshold"]
        mom_period = p["momentum_period"]
        use_atr = p["use_atr"]

        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Calculate expected move
        if use_atr:
            # ATR-based expected move
            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ], axis=1).max(axis=1)
            atr = tr.rolling(window=lookback).mean()
            # Expected move over horizon = ATR * sqrt(horizon)
            expected_move = atr * np.sqrt(horizon)
        else:
            # HV-based expected move (options formula)
            log_returns = np.log(close / close.shift(1))
            hv = log_returns.rolling(window=lookback).std() * np.sqrt(252)
            expected_move = close * hv * np.sqrt(horizon / 252)

        # Calculate reference price (starting point for expected move)
        # Use the close from 'horizon' days ago as reference
        reference_price = close.shift(horizon)

        # Current move from reference
        actual_move = close - reference_price
        actual_move_pct = (actual_move / reference_price).abs()
        expected_move_pct = expected_move / close

        # Move ratio: actual move / expected move
        move_ratio = (actual_move.abs() / expected_move).fillna(0)

        # Direction of move
        move_up = actual_move > 0
        move_down = actual_move < 0

        # Momentum confirmation
        momentum = close.diff(mom_period)
        momentum_positive = momentum > 0
        momentum_negative = momentum < 0

        # RSI for exhaustion detection
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        # Generate signals
        signal = pd.Series(0, index=df.index)

        # Breakout mode: Price exceeded expected move WITH momentum
        breakout_up = (move_ratio > breakout_mult) & move_up & momentum_positive & (rsi < 70)
        breakout_down = (move_ratio > breakout_mult) & move_down & momentum_negative & (rsi > 30)

        # Mean reversion mode: Price exceeded 1.5x expected move (overextended)
        reversion_from_up = (move_ratio > reversion_mult) & move_up & (rsi > 75)
        reversion_from_down = (move_ratio > reversion_mult) & move_down & (rsi < 25)

        # Assign signals
        # Breakout signals (trend following)
        signal = signal.where(~breakout_up, 1)
        signal = signal.where(~breakout_down, -1)

        # Reversion signals (fade extended moves)
        signal = signal.where(~reversion_from_up, -1)
        signal = signal.where(~reversion_from_down, 1)

        result = pd.DataFrame(index=df.index)
        result["signal"] = signal.fillna(0).astype(int)
        result["expected_move"] = expected_move
        result["actual_move"] = actual_move
        result["move_ratio"] = move_ratio
        result["rsi"] = rsi

        # Dynamic TP/SL based on expected move
        result["sl_level"] = close - expected_move
        result["tp_level"] = close + expected_move

        return result


# Register the strategy
register_strategy("expected_move", ExpectedMoveStrategy)
