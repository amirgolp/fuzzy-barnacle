"""Walk-forward expanding window validation splits.

Generates train/val index pairs using an expanding training window
with fixed-size validation windows. No look-ahead bias — validation
is always strictly after training data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantdash.ml.config import WalkForwardConfig


@dataclass
class SplitFold:
    """A single train/validation fold."""
    fold_idx: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int

    @property
    def train_size(self) -> int:
        return self.train_end - self.train_start

    @property
    def val_size(self) -> int:
        return self.val_end - self.val_start


def walk_forward_splits(
    n_samples: int,
    config: WalkForwardConfig | None = None,
) -> list[SplitFold]:
    """Generate expanding-window walk-forward validation splits.

    Example with n=3500, initial_train=2000, val=500, step=500:
      Fold 0: train=[0:2000], val=[2000:2500]
      Fold 1: train=[0:2500], val=[2500:3000]
      Fold 2: train=[0:3000], val=[3000:3500]

    Args:
        n_samples: Total number of samples in the dataset.
        config: Walk-forward configuration.

    Returns:
        List of SplitFold objects.
    """
    if config is None:
        config = WalkForwardConfig()

    folds = []
    fold_idx = 0
    train_end = config.initial_train_size

    while train_end + config.val_size <= n_samples:
        val_start = train_end
        val_end = min(val_start + config.val_size, n_samples)

        folds.append(SplitFold(
            fold_idx=fold_idx,
            train_start=0,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
        ))

        fold_idx += 1
        train_end += config.step_size

    return folds


def get_fold_indices(
    fold: SplitFold,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a SplitFold to train/val index arrays."""
    train_idx = np.arange(fold.train_start, fold.train_end)
    val_idx = np.arange(fold.val_start, fold.val_end)
    return train_idx, val_idx


def split_dataframe(
    df: pd.DataFrame,
    fold: SplitFold,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame according to a fold."""
    train_df = df.iloc[fold.train_start:fold.train_end]
    val_df = df.iloc[fold.val_start:fold.val_end]
    return train_df, val_df
