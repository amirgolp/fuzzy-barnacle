"""Mean Reversion (Bollinger Bands) strategy."""

import pandas as pd

from quantdash.strategies.base import Strategy
from quantdash.strategies.registry import register_strategy


class MeanReversion(Strategy):
    """
    Mean reversion strategy using Bollinger Bands.

    Buys when price touches lower band, sells when touching upper band.
    """

    name = "mean_reversion"
    description = "Bollinger Band mean-reversion strategy"
    default_params = {"period": 20, "num_std": 2.0}

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        period = self.params["period"]
        num_std = self.params["num_std"]

        mid = df["close"].rolling(window=period).mean()
        std = df["close"].rolling(window=period).std()
        upper = mid + num_std * std
        lower = mid - num_std * std

        result = df.copy()
        result["bb_upper"] = upper
        result["bb_mid"] = mid
        result["bb_lower"] = lower
        result["signal"] = 0

        result.loc[df["close"] <= lower, "signal"] = 1   # Buy at lower band
        result.loc[df["close"] >= upper, "signal"] = -1  # Sell at upper band

        return result


register_strategy("mean_reversion", MeanReversion)
