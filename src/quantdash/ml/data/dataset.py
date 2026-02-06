"""PyTorch Dataset for trading signal prediction.

Creates windowed samples from pre-computed feature matrices.
Each sample contains all branch inputs and the corresponding label.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset as _Dataset

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    _Dataset = object  # type: ignore[assignment,misc]

from quantdash.ml.config import ModelConfig


class TradingSignalDataset(_Dataset):
    """Windowed dataset for TemporalFusionSignalNet.

    Creates lookback windows for temporal branches (price, volume, cross-asset)
    and point-in-time features for non-temporal branches (pattern, macro/session).

    Each sample returns a dict of tensors keyed by branch name.
    """

    def __init__(
        self,
        price_features: np.ndarray,
        volume_features: np.ndarray,
        pattern_features: np.ndarray,
        macro_session_features: np.ndarray,
        cross_asset_features: np.ndarray | None,
        labels: np.ndarray,
        news_embeddings: np.ndarray | None = None,
        lookback: int = 120,
        config: ModelConfig | None = None,
    ):
        """
        Args:
            price_features: [N, price_channels] — price/indicator features.
            volume_features: [N, 3] — volume, volume_return, volume_accel.
            pattern_features: [N, num_patterns*2] — detected + confidence.
            macro_session_features: [N, num_macro] — macro + session features.
            cross_asset_features: [N, lookback, cross_channels] or None.
            labels: [N] — triple-barrier labels (-1, 0, 1).
            news_embeddings: [N, max_articles, 768] or None.
            lookback: Window size for temporal branches.
            config: Model architecture config.
        """
        if not _HAS_TORCH:
            raise ImportError(
                "PyTorch is required for TradingSignalDataset. "
                "Install with: pip install 'quantdash[ml]'"
            )
        self.price_features = price_features.astype(np.float32)
        self.volume_features = volume_features.astype(np.float32)
        self.pattern_features = pattern_features.astype(np.float32)
        self.macro_session_features = macro_session_features.astype(np.float32)
        # Detect valid (non-NaN) labels BEFORE casting to int
        self._label_valid = ~np.isnan(labels.astype(np.float64))
        # Replace NaN with 0 before int cast to avoid garbage values
        clean_labels = np.where(self._label_valid, labels, 0)
        self.labels = clean_labels.astype(np.int64)
        self.lookback = lookback
        self.config = config or ModelConfig()

        if cross_asset_features is not None:
            self.cross_asset_features = cross_asset_features.astype(np.float32)
        else:
            self.cross_asset_features = None

        if news_embeddings is not None:
            self.news_embeddings = news_embeddings.astype(np.float32)
        else:
            self.news_embeddings = None

        self.n_samples = len(labels)

        # Valid indices: those with enough lookback history and valid labels
        self._valid_indices = np.array([
            i for i in range(self.lookback, self.n_samples)
            if self._label_valid[i]
        ])

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        i = self._valid_indices[idx]

        # Temporal windows: [lookback, channels]
        price_window = self.price_features[i - self.lookback : i]
        volume_window = self.volume_features[i - self.lookback : i]

        # Point-in-time features (at bar i)
        pattern_vec = self.pattern_features[i]
        macro_vec = self.macro_session_features[i]

        sample = {
            "price": torch.from_numpy(price_window),
            "volume": torch.from_numpy(volume_window),
            "pattern": torch.from_numpy(pattern_vec),
            "macro": torch.from_numpy(macro_vec),
            "label": torch.tensor(self.labels[i] + 1, dtype=torch.long),
            # Shift labels from {-1,0,1} → {0,1,2} for CrossEntropy
        }

        # Cross-asset temporal window
        if self.cross_asset_features is not None:
            cross_window = self.cross_asset_features[i - self.lookback : i]
            sample["cross_asset"] = torch.from_numpy(cross_window)
        else:
            # Zero placeholder
            sample["cross_asset"] = torch.zeros(
                self.lookback, self.config.cross_asset_channels, dtype=torch.float32
            )

        # News embeddings
        if self.news_embeddings is not None:
            sample["news"] = torch.from_numpy(self.news_embeddings[i])
        else:
            sample["news"] = torch.zeros(
                self.config.max_articles, self.config.finbert_dim, dtype=torch.float32
            )

        return sample


def create_dataset_from_dataframes(
    price_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    pattern_df: pd.DataFrame,
    macro_session_df: pd.DataFrame,
    labels: pd.Series,
    cross_asset_df: pd.DataFrame | None = None,
    news_embeddings: np.ndarray | None = None,
    lookback: int = 120,
    config: ModelConfig | None = None,
) -> TradingSignalDataset:
    """Convenience constructor from pandas DataFrames.

    Handles NaN filling and converts to numpy arrays.
    """
    # Fill remaining NaNs with 0 (from rolling indicator warmup periods)
    price_arr = price_df.fillna(0).values
    volume_arr = volume_df.fillna(0).values
    pattern_arr = pattern_df.fillna(0).values
    macro_arr = macro_session_df.fillna(0).values
    label_arr = labels.values.astype(float)

    cross_arr = None
    if cross_asset_df is not None:
        cross_arr = cross_asset_df.fillna(0).values
        # Reshape to [N, channels] — will be windowed in __getitem__
        # If already 2D, leave as-is for windowing
        if cross_arr.ndim == 2:
            pass  # will be windowed like price/volume

    return TradingSignalDataset(
        price_features=price_arr,
        volume_features=volume_arr,
        pattern_features=pattern_arr,
        macro_session_features=macro_arr,
        cross_asset_features=cross_arr,
        labels=label_arr,
        news_embeddings=news_embeddings,
        lookback=lookback,
        config=config,
    )


def save_dataset_hdf5(path: Path, **arrays: np.ndarray) -> None:
    """Save numpy arrays to HDF5 for efficient loading."""
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required. Install with: pip install 'quantdash[ml]'")

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for name, arr in arrays.items():
            f.create_dataset(name, data=arr, compression="gzip", compression_opts=4)


def load_dataset_hdf5(path: Path) -> dict[str, np.ndarray]:
    """Load numpy arrays from HDF5."""
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required. Install with: pip install 'quantdash[ml]'")

    arrays = {}
    with h5py.File(path, "r") as f:
        for name in f.keys():
            arrays[name] = f[name][:]
    return arrays
