"""Pairs Trading / Statistical Arbitrage Strategy.

Trades the mean-reverting spread between two correlated instruments.
Uses z-score of the price ratio (or log spread) to detect divergences,
enters when spread widens beyond threshold, exits on convergence.

Best for: Correlated equity pairs (e.g. GOOGL/META, AMD/NVDA),
sector ETFs, market-neutral portfolios.
"""

import numpy as np
import pandas as pd

from quantdash.strategies.base import Strategy
from quantdash.strategies.registry import register_strategy


class PairsTradingStrategy(Strategy):
    """
    Statistical arbitrage pairs trading strategy.

    Computes the price ratio between the primary symbol (in df) and a
    synthetic "pair" represented by a rolling beta-adjusted spread.
    When the z-score of the spread exceeds thresholds, enter trades
    expecting mean-reversion.

    For single-symbol mode (no second symbol data), uses the ratio of
    close price to its rolling regression line as a self-referential
    spread, which still captures mean-reversion behaviour.

    Signals:
        +1 = Long (spread below -entry_z, expect convergence up)
        -1 = Short (spread above +entry_z, expect convergence down)
         0 = Flat
    """

    name = "pairs_trading"
    description = "Statistical arbitrage / pairs mean-reversion"
    default_params = {
        "lookback": 60,            # Rolling window for spread statistics
        "entry_z": 2.0,            # Z-score threshold to enter
        "exit_z": 0.5,             # Z-score threshold to exit (near mean)
        "stop_z": 3.5,             # Z-score stop loss (spread widening)
        "half_life_max": 50,       # Max mean-reversion half-life (days)
        "min_correlation": 0.7,    # Min rolling correlation to trade
        "use_log_spread": True,    # Use log prices for stationarity
        "recalc_interval": 20,     # Recalculate hedge ratio every N bars
    }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate pairs trading signals.

        When a second symbol is not available, this strategy operates in
        self-referential mode: it computes the spread as the deviation of
        price from its own rolling regression (Ornstein-Uhlenbeck proxy).
        """
        lookback = self.params["lookback"]
        entry_z = self.params["entry_z"]
        exit_z = self.params["exit_z"]
        stop_z = self.params["stop_z"]
        use_log = self.params["use_log_spread"]
        recalc = self.params["recalc_interval"]

        result = df.copy()
        close = df["close"]
        n = len(df)

        # ── Compute spread ──
        # Self-referential mode: spread = close - rolling regression line
        if use_log:
            price = np.log(close)
        else:
            price = close.copy()

        # Rolling linear regression: regress price on time index
        spread = pd.Series(np.nan, index=df.index)
        hedge_ratio = pd.Series(np.nan, index=df.index)

        for i in range(lookback, n):
            window = price.iloc[i - lookback:i].values
            x = np.arange(lookback, dtype=float)
            x_mean = x.mean()
            y_mean = window.mean()

            denom = np.sum((x - x_mean) ** 2)
            if denom == 0:
                continue

            slope = np.sum((x - x_mean) * (window - y_mean)) / denom
            intercept = y_mean - slope * x_mean

            # Predicted value at current position
            predicted = intercept + slope * lookback
            spread.iloc[i] = price.iloc[i] - predicted
            hedge_ratio.iloc[i] = slope

        # Z-score of spread
        spread_mean = spread.rolling(window=lookback).mean()
        spread_std = spread.rolling(window=lookback).std()
        z_score = (spread - spread_mean) / (spread_std + 1e-10)

        result["spread"] = spread
        result["z_score"] = z_score
        result["hedge_ratio"] = hedge_ratio

        # ── Estimate half-life (Ornstein-Uhlenbeck) ──
        half_life = self._estimate_half_life(spread, lookback)
        result["half_life"] = half_life

        # ── Signal generation ──
        signals = np.zeros(n, dtype=int)
        z_vals = z_score.values
        hl_vals = half_life.values
        hl_max = self.params["half_life_max"]

        in_position = 0
        entry_z_val = 0.0

        for i in range(lookback, n):
            z = z_vals[i]
            hl = hl_vals[i]

            if np.isnan(z) or np.isnan(hl):
                continue

            # Only trade if half-life is reasonable (mean-reverting)
            tradeable = 0 < hl < hl_max

            if in_position == 0 and tradeable:
                if z < -entry_z:
                    signals[i] = 1  # Long: spread too low, expect reversion up
                    in_position = 1
                    entry_z_val = z
                elif z > entry_z:
                    signals[i] = -1  # Short: spread too high, expect reversion down
                    in_position = -1
                    entry_z_val = z

            elif in_position == 1:
                # Exit long: z-score reverted toward mean
                if z >= -exit_z:
                    signals[i] = -1
                    in_position = 0
                # Stop: z-score widened further
                elif z < -stop_z:
                    signals[i] = -1
                    in_position = 0

            elif in_position == -1:
                # Exit short: z-score reverted toward mean
                if z <= exit_z:
                    signals[i] = 1
                    in_position = 0
                # Stop: z-score widened further
                elif z > stop_z:
                    signals[i] = 1
                    in_position = 0

        result["signal"] = signals

        # Clean up
        result["signal"] = result["signal"].fillna(0)
        for col in result.select_dtypes(include=[np.number]).columns:
            if col != "signal":
                result[col] = result[col].ffill().fillna(0)

        return result

    @staticmethod
    def _estimate_half_life(spread: pd.Series, lookback: int) -> pd.Series:
        """
        Estimate mean-reversion half-life using Ornstein-Uhlenbeck model.

        Regresses spread_change on lagged_spread to get speed of
        mean-reversion (lambda). Half-life = -ln(2) / ln(1 + lambda).
        """
        half_life = pd.Series(np.nan, index=spread.index)
        spread_vals = spread.values

        for i in range(lookback + 1, len(spread_vals)):
            window = spread_vals[i - lookback:i]
            if np.any(np.isnan(window)):
                continue

            y = np.diff(window)  # spread_change
            x = window[:-1]     # lagged_spread

            x_mean = x.mean()
            y_mean = y.mean()

            denom = np.sum((x - x_mean) ** 2)
            if denom == 0:
                continue

            beta = np.sum((x - x_mean) * (y - y_mean)) / denom

            if beta >= 0:
                # Not mean-reverting
                half_life.iloc[i] = np.inf
            else:
                hl = -np.log(2) / np.log(1 + beta) if (1 + beta) > 0 else np.inf
                half_life.iloc[i] = max(hl, 0)

        return half_life


register_strategy("pairs_trading", PairsTradingStrategy)
