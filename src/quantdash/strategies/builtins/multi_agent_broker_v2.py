"""Multi-Agent Broker Strategy V2 - Improved.

Improvements over V1:
- Trend-aware alpha process (doesn't fight strong trends)
- Reduced randomness for more stable behavior
- Better inventory management with trend confirmation
- Lower trading frequency
- Improved entry/exit logic with trend filters
"""

import numpy as np
import pandas as pd

from quantdash.strategies.base import Strategy
from quantdash.strategies.registry import register_strategy


class MultiAgentBrokerV2Strategy(Strategy):
    """
    Improved multi-agent broker strategy with trend awareness.

    Key improvements:
    - Trend-aware alpha (biased toward prevailing trend)
    - Reduced randomness (lower eta)
    - Trend strength filter (don't trade in weak trends)
    - Better inventory management
    - Higher quality trades, fewer whipsaws
    """

    name = "multi_agent_broker_v2"
    description = "Improved multi-agent broker with trend awareness and better risk management"
    default_params = {
        # Alpha process parameters
        "theta": 0.6,                  # Moderate mean reversion
        "eta": 0.03,                   # Slightly higher for some variability
        "sigma": 0.02,

        # Broker parameters
        "kappa": 0.001,
        "b": 0.0001,
        "k": 0.0015,                   # Moderate temp impact
        "phi": 0.25,                   # Moderate inventory penalty
        "a": 0.5,

        # Trading parameters
        "alpha_multiplier": 15.0,      # Moderate multiplier
        "max_inventory": 10.0,         # Standard max inventory
        "observation_period": 100,
        "min_alpha_threshold": 0.015,  # MUCH lower threshold (was 0.03)

        # NEW: Trend awareness
        "trend_period": 50,            # Period for trend detection
        "min_trend_strength": 0.10,    # MUCH lower minimum (was 0.25)
        "trend_bias_weight": 0.5,      # Balanced weight

        # Risk management
        "stop_loss_pct": 2.5,
        "take_profit_pct": 4.0,
    }

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self.current_alpha = 0.0
        self.broker_inventory = 0.0

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals with trend-aware alpha."""
        theta = self.params["theta"]
        eta = self.params["eta"]
        alpha_mult = self.params["alpha_multiplier"]
        obs_period = self.params["observation_period"]
        min_threshold = self.params["min_alpha_threshold"]
        trend_period = self.params["trend_period"]
        min_trend_strength = self.params["min_trend_strength"]
        trend_bias_weight = self.params["trend_bias_weight"]

        result = df.copy()
        close = df["close"].values
        volume = df["volume"].values if "volume" in df.columns else np.ones(len(df))
        n = len(df)

        # Calculate trend strength and direction
        trend_sma = pd.Series(close).rolling(trend_period).mean().values
        trend_direction = np.where(close > trend_sma, 1, -1)

        # Calculate trend strength (normalized price distance from SMA)
        trend_strength = abs(close - trend_sma) / (trend_sma + 1e-10)
        trend_strength = np.clip(trend_strength / trend_strength.std(), 0, 1)

        # Initialize arrays
        alpha_values = np.zeros(n)
        trend_bias = np.zeros(n)
        combined_alpha = np.zeros(n)
        inventory_values = np.zeros(n)

        current_alpha = 0.0
        current_inventory = 0.0

        for i in range(1, n):
            dt = 1.0 / 252.0  # Daily

            # 1. Calculate trend bias (mean-reverting but with trend awareness)
            # Trend bias pulls alpha toward the trend direction
            trend_bias[i] = trend_direction[i] * trend_strength[i]

            # 2. Update Ornstein-Uhlenbeck alpha process with trend influence
            # Standard OU: dα = -θ α dt + η dW
            # Trend-aware: dα = -θ (α - trend_target) dt + η dW
            trend_target = trend_bias[i] * trend_bias_weight

            drift = -theta * (current_alpha - trend_target) * dt
            diffusion = eta * np.sqrt(dt) * np.random.randn()
            current_alpha += drift + diffusion

            # Bound alpha to reasonable range
            current_alpha = np.clip(current_alpha, -0.5, 0.5)

            alpha_values[i] = current_alpha

            # 3. Combine process alpha with trend bias
            # Give more weight to trend in strong trends
            if trend_strength[i] > min_trend_strength:
                combined_alpha[i] = (
                    (1 - trend_bias_weight) * current_alpha +
                    trend_bias_weight * trend_bias[i]
                )
            else:
                # In weak trends, rely more on alpha process
                combined_alpha[i] = current_alpha

            # 4. Update inventory (simplified)
            if combined_alpha[i] > min_threshold:
                inventory_change = 0.1 * dt  # Gradual accumulation
                current_inventory += inventory_change
            elif combined_alpha[i] < -min_threshold:
                inventory_change = -0.1 * dt
                current_inventory += inventory_change

            # Apply inventory limits
            max_inv = self.params["max_inventory"]
            current_inventory = np.clip(current_inventory, -max_inv, max_inv)
            inventory_values[i] = current_inventory

        # Store intermediate values
        result["alpha"] = alpha_values
        result["trend_bias"] = trend_bias
        result["combined_alpha"] = combined_alpha
        result["trend_strength"] = trend_strength
        result["broker_inventory"] = inventory_values

        # 5. Generate signals from combined alpha
        result["signal"] = 0

        # Buy when combined alpha is positive AND trend is favorable
        buy_condition = (
            (combined_alpha > min_threshold) &
            (trend_strength > min_trend_strength / 2)  # Reduced threshold for signals
        )

        # Sell when combined alpha is negative AND trend is favorable
        sell_condition = (
            (combined_alpha < -min_threshold) &
            (trend_strength > min_trend_strength / 2)
        )

        result.loc[buy_condition, "signal"] = 1
        result.loc[sell_condition, "signal"] = -1

        # 6. Apply inventory management
        result = self._apply_inventory_management(result)

        # Clean up NaN and inf values
        result["signal"] = result["signal"].fillna(0)
        for col in result.select_dtypes(include=[np.number]).columns:
            result[col] = result[col].replace([np.inf, -np.inf], np.nan)
            if col != "signal":
                result[col] = result[col].ffill().fillna(0)

        return result

    def _apply_inventory_management(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply inventory management with trend awareness."""
        signals = df["signal"].values.copy()
        inventory = df["broker_inventory"].values
        trend_bias = df["trend_bias"].values
        max_inv = self.params["max_inventory"]

        for i in range(len(signals)):
            inv = inventory[i]

            # More aggressive inventory management
            # If inventory is high and trend is against us, force exit
            if inv > 0.6 * max_inv:
                if trend_bias[i] < -0.2:  # Downtrend starting
                    signals[i] = -1  # Force sell
                elif signals[i] == 1:
                    signals[i] = 0  # Cancel buy

            if inv < -0.6 * max_inv:
                if trend_bias[i] > 0.2:  # Uptrend starting
                    signals[i] = 1  # Force cover
                elif signals[i] == -1:
                    signals[i] = 0  # Cancel sell

            # Hard limits
            if inv > 0.85 * max_inv:
                signals[i] = -1
            elif inv < -0.85 * max_inv:
                signals[i] = 1

        df = df.copy()
        df["signal"] = signals
        return df


register_strategy("multi_agent_broker_v2", MultiAgentBrokerV2Strategy)
