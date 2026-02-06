"""SMA Crossover strategy."""

import pandas as pd

from quantdash.strategies.base import Strategy
from quantdash.strategies.registry import register_strategy


class SMACrossover(Strategy):
    """
    Simple Moving Average crossover strategy.

    Buys when fast SMA crosses above slow SMA, sells on reverse.
    """

    name = "sma_crossover"
    description = "SMA crossover trend-following strategy"
    default_params = {"fast_period": 20, "slow_period": 50}

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        fast = df["close"].rolling(window=self.params["fast_period"]).mean()
        slow = df["close"].rolling(window=self.params["slow_period"]).mean()

        result = df.copy()
        result["fast_ma"] = fast
        result["slow_ma"] = slow
        result["signal"] = 0

        # Crossover: fast crosses above slow = buy (1), below = sell (-1)
        result.loc[fast > slow, "signal"] = 1
        result.loc[fast < slow, "signal"] = -1

        # Only signal on actual crossover points, hold otherwise
        result["signal"] = result["signal"].diff().clip(-1, 1).fillna(0).astype(int)

        return result


register_strategy("sma_crossover", SMACrossover)
