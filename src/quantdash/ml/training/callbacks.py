"""Training callbacks: early stopping, checkpoint saving, metric logging."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Args:
        patience: Epochs to wait after last improvement.
        min_delta: Minimum change to qualify as an improvement.
        mode: 'min' (loss) or 'max' (accuracy/F1).
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "max",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value: float | None = None
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        """Check if training should stop.

        Returns True if training should stop.
        """
        if self.best_value is None:
            self.best_value = value
            return False

        if self.mode == "max":
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    "Early stopping triggered after %d epochs without improvement. "
                    "Best: %.4f", self.patience, self.best_value
                )
                return True

        return False


class CheckpointSaver:
    """Save model checkpoints when metric improves.

    Args:
        save_dir: Directory for checkpoint files.
        filename_prefix: Prefix for checkpoint files.
        mode: 'min' or 'max'.
    """

    def __init__(
        self,
        save_dir: Path,
        filename_prefix: str = "model",
        mode: str = "max",
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.filename_prefix = filename_prefix
        self.mode = mode
        self.best_value: float | None = None

    def __call__(
        self,
        value: float,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        epoch: int = 0,
        extra: dict | None = None,
    ) -> bool:
        """Save if metric improved. Returns True if saved."""
        if self.best_value is None:
            improved = True
        elif self.mode == "max":
            improved = value > self.best_value
        else:
            improved = value < self.best_value

        if improved:
            self.best_value = value
            path = self.save_dir / f"{self.filename_prefix}_best.pt"

            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "metric_value": value,
            }
            if optimizer is not None:
                state["optimizer_state_dict"] = optimizer.state_dict()
            if extra is not None:
                state.update(extra)

            torch.save(state, path)
            logger.info(
                "Checkpoint saved: %s (metric=%.4f, epoch=%d)", path, value, epoch
            )
            return True

        return False

    def load_best(self, model: torch.nn.Module) -> dict:
        """Load the best checkpoint into the model."""
        path = self.save_dir / f"{self.filename_prefix}_best.pt"
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint found at {path}")

        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        logger.info("Loaded best checkpoint from %s (epoch=%d)", path, state["epoch"])
        return state


class MetricTracker:
    """Track and log training metrics across epochs."""

    def __init__(self):
        self.history: dict[str, list[float]] = {}

    def update(self, metrics: dict[str, float]) -> None:
        for name, value in metrics.items():
            if name not in self.history:
                self.history[name] = []
            self.history[name].append(value)

    def get(self, name: str) -> list[float]:
        return self.history.get(name, [])

    def last(self, name: str) -> float | None:
        vals = self.history.get(name, [])
        return vals[-1] if vals else None

    def best(self, name: str, mode: str = "max") -> float | None:
        vals = self.history.get(name, [])
        if not vals:
            return None
        return max(vals) if mode == "max" else min(vals)
