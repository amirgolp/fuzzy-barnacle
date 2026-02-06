"""Multi-Agent Broker Strategy.

Based on research paper: "Multi-Agent Trading with Brokers and an Informed Trader"
Implements a sophisticated trading system where a broker:
1. Infers alpha (information edge) from observed order flow
2. Manages inventory with risk penalties
3. Exploits information asymmetry through optimal trading

The strategy uses an Ornstein-Uhlenbeck process for alpha evolution
and sophisticated inventory management.
"""

import numpy as np
import pandas as pd

from quantdash.strategies.base import Strategy
from quantdash.strategies.registry import register_strategy


class MultiAgentBrokerStrategy(Strategy):
    """
    Multi-agent broker strategy with alpha inference and inventory management.

    Core components:
    - Ornstein-Uhlenbeck alpha process (mean-reverting information signal)
    - Order flow inference from volume/price patterns
    - Dynamic inventory management with risk penalties
    - Optimal trading rate calculation

    Based on Stackelberg game theory where broker competes for informed flow
    while managing inventory risk.
    """

    name = "multi_agent_broker"
    description = "Multi-agent broker with alpha inference and inventory management"
    default_params = {
        # Alpha process parameters (Ornstein-Uhlenbeck)
        "theta": 0.5,              # Mean reversion speed
        "eta": 0.05,               # Volatility of alpha signal (reduced for stability)
        "sigma": 0.02,             # Price volatility

        # Broker parameters
        "kappa": 0.001,            # Liquidity cost
        "b": 0.0001,               # Permanent price impact coefficient
        "k": 0.001,                # Temporary price impact coefficient (increased for less aggressive trading)
        "phi": 0.2,                # Running inventory penalty (increased to limit position size)
        "a": 0.5,                  # Terminal inventory penalty

        # Trading parameters
        "alpha_multiplier": 20.0,  # Signal amplification factor (reduced from 100)
        "max_inventory": 10.0,     # Maximum inventory (increased for more flexibility)
        "observation_period": 100, # Periods for order flow estimation
        "min_alpha_threshold": 0.02, # Minimum alpha to trade (increased for quality trades)

        # Risk management
        "stop_loss_pct": 2.0,      # Stop loss percentage
        "take_profit_pct": 3.0,    # Take profit percentage
    }

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        # Initialize alpha process state
        self.current_alpha = 0.0
        self.last_update_time = 0
        self.broker_inventory = 0.0
        self.cash = 0.0

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals using multi-agent broker logic."""
        theta = self.params["theta"]
        eta = self.params["eta"]
        sigma = self.params["sigma"]
        alpha_mult = self.params["alpha_multiplier"]
        obs_period = self.params["observation_period"]
        min_threshold = self.params["min_alpha_threshold"]

        result = df.copy()
        close = df["close"].values
        volume = df["volume"].values if "volume" in df.columns else np.ones(len(df))
        n = len(df)

        # Initialize arrays for tracking
        alpha_values = np.zeros(n)
        inferred_alpha = np.zeros(n)
        order_flow = np.zeros(n)
        inventory_values = np.zeros(n)
        trading_rate = np.zeros(n)

        # Initialize alpha process
        current_alpha = 0.0
        current_inventory = 0.0

        for i in range(1, n):
            # Time step (assume 1 hour for H1 data, adjust as needed)
            dt = 1.0 / 24.0  # 1 hour in days

            # 1. Update Ornstein-Uhlenbeck alpha process
            # dα(t) = -θ α(t) dt + η dW
            drift = -theta * current_alpha * dt
            diffusion = eta * np.sqrt(dt) * np.random.randn()
            current_alpha += drift + diffusion

            alpha_values[i] = current_alpha

            # 2. Estimate order flow from market data
            # Order flow approximation: use volume-weighted price change
            if i >= obs_period:
                # Calculate weighted price momentum
                price_changes = np.diff(close[i-obs_period:i+1])
                volumes = volume[i-obs_period:i]

                # Volume-weighted order flow proxy
                if volumes.sum() > 0:
                    flow = np.sum(price_changes * volumes) / volumes.sum()
                else:
                    flow = 0.0

                order_flow[i] = flow

                # 3. Infer alpha from observed order flow
                # α = (observed_flow - ω₁ × inventory) / ω₀
                omega0 = self._calculate_omega0(dt)
                omega1 = self._calculate_omega1(dt)

                if abs(omega0) > 1e-10:
                    inferred = (order_flow[i] - omega1 * current_inventory) / omega0
                    # Blend inferred alpha with process alpha (70% inferred, 30% process)
                    current_alpha = 0.7 * inferred + 0.3 * current_alpha
                    inferred_alpha[i] = inferred

            # 4. Calculate optimal trading rate
            # ν(t) = [x/(2k)]α + [b/(2k)]q + inventory_adjustments
            trade_rate = self._calculate_trading_rate(
                current_alpha,
                current_inventory,
                close[i]
            )
            trading_rate[i] = trade_rate

            # 5. Update inventory based on trading rate
            # Inventory change is proportional to trading rate
            inventory_change = trade_rate * dt
            current_inventory += inventory_change

            # Apply inventory limits
            max_inv = self.params["max_inventory"]
            current_inventory = np.clip(current_inventory, -max_inv, max_inv)
            inventory_values[i] = current_inventory

        # Store intermediate values in result
        result["alpha"] = alpha_values
        result["inferred_alpha"] = inferred_alpha
        result["order_flow"] = order_flow
        result["broker_inventory"] = inventory_values
        result["trading_rate"] = trading_rate

        # 6. Generate signals from alpha
        result["signal"] = 0

        # Buy when alpha is significantly positive (expect price increase)
        buy_condition = alpha_values > min_threshold
        sell_condition = alpha_values < -min_threshold

        # Scale signal strength by alpha magnitude
        result.loc[buy_condition, "signal"] = 1
        result.loc[sell_condition, "signal"] = -1

        # Apply inventory-based adjustments
        # Reduce position when inventory is high (risk management)
        result = self._apply_inventory_management(result)

        # Clean up NaN and inf values
        result["signal"] = result["signal"].fillna(0)

        # Replace NaN and inf values in all numeric columns
        for col in result.select_dtypes(include=[np.number]).columns:
            result[col] = result[col].replace([np.inf, -np.inf], np.nan)
            if col != "signal":
                result[col] = result[col].ffill().fillna(0)

        return result

    def _calculate_omega0(self, dt: float) -> float:
        """
        Calculate omega0 (speculative component) for informed trader.

        omega0 represents how aggressively the informed trader exploits alpha.
        Higher kappa (liquidity cost) reduces omega0.
        """
        kappa = self.params["kappa"]
        phi = self.params["phi"]
        T = 1.0  # Time to maturity (assume 1 day)

        # Simplified approximation
        time_factor = np.exp(-phi * T)
        if abs(kappa) > 1e-10:
            return time_factor / kappa
        return 0.0

    def _calculate_omega1(self, dt: float) -> float:
        """
        Calculate omega1 (inventory management component) for informed trader.

        Always negative - represents urgency to unload inventory.
        """
        phi = self.params["phi"]
        T = 1.0  # Time to maturity

        # Always negative to discharge inventory
        return -np.exp(-phi * T) / max(T, 0.01)

    def _calculate_trading_rate(self, alpha: float, inventory: float, price: float) -> float:
        """
        Calculate optimal trading velocity in the lit market.

        Components:
        1. Speculative: exploit alpha signal
        2. Inventory management: reduce position risk
        3. Adjustment: based on broker parameters

        Returns:
            Optimal trading rate (positive = buy, negative = sell)
        """
        k = self.params["k"]
        b = self.params["b"]
        alpha_mult = self.params["alpha_multiplier"]

        if k <= 0:
            return 0.0

        # Speculative component (exploit alpha)
        omega0 = self._calculate_omega0(1.0)
        speculative = (omega0 / (2.0 * k)) * alpha * alpha_mult

        # Inventory management component (reduce risk)
        inventory_mgmt = -(b / (2.0 * k)) * inventory

        # Total trading rate
        rate = speculative + inventory_mgmt

        return rate

    def _apply_inventory_management(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply inventory management rules to signals.

        Reduces or reverses positions when inventory reaches extreme levels.
        Implements risk penalties and position limits.
        """
        signals = df["signal"].values.copy()
        inventory = df["broker_inventory"].values
        max_inv = self.params["max_inventory"]

        for i in range(len(signals)):
            inv = inventory[i]

            # If inventory is too high (overbought), reduce long signals
            if inv > 0.7 * max_inv and signals[i] == 1:
                signals[i] = 0  # Cancel buy signal

            # If inventory is too low (oversold), reduce short signals
            if inv < -0.7 * max_inv and signals[i] == -1:
                signals[i] = 0  # Cancel sell signal

            # Force exit if inventory exceeds limits
            if inv > 0.9 * max_inv:
                signals[i] = -1  # Force sell
            elif inv < -0.9 * max_inv:
                signals[i] = 1  # Force buy (cover short)

        df = df.copy()
        df["signal"] = signals
        return df

    def _calculate_inventory_penalty(self, inventory: float) -> float:
        """
        Calculate inventory penalty cost.

        Penalty increases quadratically with inventory size.
        Combines running penalty (phi) and terminal penalty (a).
        """
        phi = self.params["phi"]
        a = self.params["a"]

        return 0.5 * phi * inventory**2 + 0.5 * a * inventory**2


register_strategy("multi_agent_broker", MultiAgentBrokerStrategy)
