"""RL fine-tuning with PPO (Phase 2 training).

Freezes branch encoders, fine-tunes fusion + output heads using PPO.
Custom Gymnasium environment where observation = fused embedding [256]
and reward = position × return - fees - drawdown penalty.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from quantdash.ml.config import RLConfig

logger = logging.getLogger(__name__)


class TradingEnv:
    """Gymnasium-compatible trading environment for RL fine-tuning.

    Uses pre-computed fused embeddings from the frozen encoder as observations.
    Action space: {0=sell, 1=hold, 2=buy} → position in {-1, 0, +1}.

    Reward = position × bar_return - |Δposition| × fee_bps - λ × drawdown_penalty
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        returns: np.ndarray,
        fee_bps: int = 10,
        lambda_drawdown: float = 0.5,
    ):
        """
        Args:
            embeddings: [N, output_d] fused embeddings from frozen model.
            returns: [N] per-bar returns (close-to-close pct change).
            fee_bps: Trading fee in basis points.
            lambda_drawdown: Drawdown penalty weight.
        """
        if gym is None:
            raise ImportError(
                "gymnasium required for RL. Install with: pip install 'quantdash[ml]'"
            )

        self.embeddings = embeddings.astype(np.float32)
        self.returns = returns.astype(np.float32)
        self.fee_bps = fee_bps / 10_000
        self.lambda_drawdown = lambda_drawdown
        self.n_steps = len(embeddings)

        # Gymnasium interface
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(embeddings.shape[1],), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)  # sell=0, hold=1, buy=2

        self._reset_state()

    def _reset_state(self) -> None:
        self.current_step = 0
        self.position = 0  # -1, 0, or +1
        self.equity = 1.0
        self.peak_equity = 1.0
        self.total_pnl = 0.0

    def _safe_obs(self, idx: int) -> np.ndarray:
        """Return observation with NaN/inf replaced by 0."""
        obs = self.embeddings[idx].copy()
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def reset(self, seed: int | None = None, options: dict | None = None):
        self._reset_state()
        return self._safe_obs(self.current_step), {}

    def step(self, action: int):
        # Map action to position: {0: -1, 1: 0, 2: +1}
        new_position = action - 1

        # Trading cost for position changes
        position_change = abs(new_position - self.position)
        fee = position_change * self.fee_bps

        # PnL from holding position during this bar
        # Clip returns to prevent extreme values (z-scored data can have outliers)
        bar_return = float(np.clip(self.returns[self.current_step], -0.05, 0.05))
        pnl = self.position * bar_return - fee

        # Additive equity tracking (prevents multiplicative overflow)
        self.equity += pnl
        self.equity = max(self.equity, 0.01)  # floor to prevent negative
        self.peak_equity = max(self.peak_equity, self.equity)

        # Drawdown penalty
        drawdown = max(0.0, (self.peak_equity - self.equity) / self.peak_equity)
        drawdown_penalty = self.lambda_drawdown * drawdown

        # Reward (clipped for stable PPO training)
        reward = float(np.clip(pnl - drawdown_penalty, -1.0, 1.0))

        # Update state
        self.position = new_position
        self.total_pnl += pnl
        self.current_step += 1

        # Terminate on end of data or bankruptcy
        terminated = self.current_step >= self.n_steps - 1 or self.equity <= 0.01
        truncated = False

        obs = self._safe_obs(min(self.current_step, self.n_steps - 1))
        info = {
            "equity": float(self.equity),
            "position": self.position,
            "drawdown": float(drawdown),
            "total_pnl": float(self.total_pnl),
        }

        return obs, reward, terminated, truncated, info


def _wrap_as_gymnasium(env: TradingEnv) -> gym.Env:
    """Wrap TradingEnv to fully satisfy Gymnasium API."""

    class _WrappedEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, base_env: TradingEnv):
            super().__init__()
            self._env = base_env
            self.observation_space = base_env.observation_space
            self.action_space = base_env.action_space

        def reset(self, seed=None, options=None):
            return self._env.reset(seed=seed, options=options)

        def step(self, action):
            return self._env.step(action)

    return _WrappedEnv(env)


def extract_fused_embeddings(
    model,
    dataset,
    device: str = "cpu",
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract fused embeddings from frozen model for RL training.

    Args:
        model: Trained TemporalFusionSignalNet.
        dataset: TradingSignalDataset.
        device: Device for inference.
        batch_size: Batch size.

    Returns:
        (embeddings [N, output_d], returns [N])
    """
    from torch.utils.data import DataLoader

    model.eval()
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_embeddings = []
    all_returns = []

    # Returns column index in price features: OHLCV(0-4), returns(5)
    # These are z-score normalized (~N(0,1)). Scale to approximate real
    # 1H return magnitudes (~0.3% std) so the RL reward is realistic.
    RETURN_COL = 5
    RETURN_SCALE = 0.003  # approximate 1H return volatility

    with torch.no_grad():
        for batch in loader:
            price = batch["price"].to(device)
            volume = batch["volume"].to(device)
            pattern = batch["pattern"].to(device)
            news = batch["news"].to(device)
            macro = batch["macro"].to(device)
            cross_asset = batch["cross_asset"].to(device)

            outputs = model(price, volume, pattern, news, macro, cross_asset)
            emb = outputs["fused_embedding"].cpu().numpy()
            # Replace any NaN in embeddings (from model instability)
            emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
            all_embeddings.append(emb)

            # Z-scored returns at last timestep, scaled to real magnitude
            ret_col = min(RETURN_COL, price.shape[2] - 1)
            scaled_returns = price[:, -1, ret_col].cpu().numpy() * RETURN_SCALE
            all_returns.append(scaled_returns)

    embeddings = np.concatenate(all_embeddings, axis=0)
    returns = np.concatenate(all_returns, axis=0)

    return embeddings, returns


def train_ppo(
    model,
    dataset,
    rl_config: RLConfig | None = None,
    fee_bps: int = 10,
    device: str = "cpu",
    save_dir: Path | None = None,
) -> dict:
    """Run PPO fine-tuning on the model.

    1. Freeze encoder branches
    2. Extract fused embeddings
    3. Train PPO agent in TradingEnv
    4. Update fusion + output head weights

    Args:
        model: Pre-trained TemporalFusionSignalNet.
        dataset: Training dataset.
        rl_config: PPO hyperparameters.
        fee_bps: Trading fee in basis points.
        device: Device.
        save_dir: Directory to save RL checkpoints.

    Returns:
        Training metrics dict.
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError:
        raise ImportError(
            "stable-baselines3 required for RL. "
            "Install with: pip install 'quantdash[ml]'"
        )

    if rl_config is None:
        rl_config = RLConfig()

    # 1. Freeze encoders
    model.freeze_encoders()
    logger.info("Froze encoder branches for RL fine-tuning")

    # 2. Extract embeddings
    logger.info("Extracting fused embeddings from frozen encoder...")
    embeddings, returns = extract_fused_embeddings(model, dataset, device=device)
    logger.info("Extracted %d embeddings of dim %d", *embeddings.shape)

    # 3. Create environment
    base_env = TradingEnv(
        embeddings=embeddings,
        returns=returns,
        fee_bps=fee_bps,
        lambda_drawdown=rl_config.lambda_drawdown,
    )
    env = DummyVecEnv([lambda: _wrap_as_gymnasium(base_env)])

    # 4. Train PPO (on CPU — MlpPolicy doesn't benefit from GPU)
    ppo = PPO(
        "MlpPolicy",
        env,
        learning_rate=rl_config.learning_rate,
        n_steps=min(rl_config.n_steps, len(embeddings) - 1),
        batch_size=rl_config.batch_size,
        n_epochs=rl_config.n_epochs,
        gamma=rl_config.gamma,
        verbose=1,
        device="cpu",
    )

    logger.info("Starting PPO training for %d timesteps...", rl_config.total_timesteps)
    ppo.learn(total_timesteps=rl_config.total_timesteps)

    # 5. Save
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        ppo.save(save_dir / "ppo_policy")
        logger.info("Saved PPO policy to %s", save_dir / "ppo_policy")

    return {
        "total_timesteps": rl_config.total_timesteps,
        "final_equity": base_env.equity,
        "total_pnl": base_env.total_pnl,
    }
