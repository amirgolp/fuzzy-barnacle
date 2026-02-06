"""FinBERT offline embedding pipeline.

Pre-computes 768-dim embeddings from news articles using ProsusAI/finbert.
Stores results in HDF5 indexed by timestamp for efficient loading.

Usage:
    embeddings = compute_finbert_embeddings(articles)
    save_embeddings_hdf5(path, timestamps, embeddings)
    loaded = load_embeddings_hdf5(path, target_index, max_articles=10)
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FINBERT_MODEL_NAME = "ProsusAI/finbert"
FINBERT_DIM = 768


def compute_finbert_embeddings(
    texts: list[str],
    model_name: str = FINBERT_MODEL_NAME,
    batch_size: int = 16,
    max_length: int = 512,
    device: str | None = None,
) -> np.ndarray:
    """Compute FinBERT embeddings for a list of texts.

    Args:
        texts: List of article texts to embed.
        model_name: HuggingFace model name.
        batch_size: Inference batch size.
        max_length: Max token length (truncated).
        device: 'cuda', 'cpu', or None (auto-detect).

    Returns:
        Array of shape [len(texts), 768].
    """
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        raise ImportError(
            "transformers and torch required. "
            "Install with: pip install 'quantdash[ml]'"
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(
        "Loading FinBERT model %s on %s...", model_name, device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded)
            # Use CLS token embedding
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(cls_embeddings.cpu().numpy())

        if (i // batch_size) % 10 == 0:
            logger.info(
                "Embedded %d / %d articles", min(i + batch_size, len(texts)), len(texts)
            )

    return np.concatenate(all_embeddings, axis=0)


def save_embeddings_hdf5(
    path: Path,
    timestamps: np.ndarray | list[datetime],
    embeddings: np.ndarray,
    article_indices: np.ndarray | None = None,
) -> None:
    """Save embeddings to HDF5 with timestamp index.

    Args:
        path: Output HDF5 file path.
        timestamps: Publication timestamps per article.
        embeddings: [N_articles, 768] embeddings.
        article_indices: Optional mapping of article to bar index.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required. Install with: pip install 'quantdash[ml]'")

    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert timestamps to epoch seconds for storage
    if isinstance(timestamps, list):
        timestamps = np.array(timestamps)

    ts_epoch = np.array([
        t.timestamp() if isinstance(t, datetime) else float(t)
        for t in timestamps
    ])

    with h5py.File(path, "w") as f:
        f.create_dataset("embeddings", data=embeddings, compression="gzip")
        f.create_dataset("timestamps", data=ts_epoch)
        if article_indices is not None:
            f.create_dataset("article_indices", data=article_indices)
        f.attrs["model"] = FINBERT_MODEL_NAME
        f.attrs["dim"] = FINBERT_DIM

    logger.info("Saved %d embeddings to %s", len(embeddings), path)


def load_embeddings_hdf5(
    path: Path,
    target_index: pd.DatetimeIndex | None = None,
    max_articles: int = 10,
    lookback_hours: int = 24,
) -> np.ndarray:
    """Load embeddings from HDF5 and align to target OHLCV timestamps.

    For each bar, collects up to max_articles embeddings from the
    preceding lookback_hours window. Returns zero vectors for missing articles.

    Args:
        path: HDF5 file with embeddings.
        target_index: OHLCV bar timestamps to align to.
        max_articles: Max articles per bar window.
        lookback_hours: Hours to look back for articles per bar.

    Returns:
        Array of shape [N_bars, max_articles, 768].
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required. Install with: pip install 'quantdash[ml]'")

    with h5py.File(path, "r") as f:
        embeddings = f["embeddings"][:]
        ts_epoch = f["timestamps"][:]

    article_times = pd.to_datetime(ts_epoch, unit="s")

    if target_index is None:
        # Return raw embeddings
        return embeddings

    n_bars = len(target_index)
    result = np.zeros((n_bars, max_articles, FINBERT_DIM), dtype=np.float32)

    # Sort articles by time
    sort_idx = np.argsort(ts_epoch)
    sorted_times = article_times[sort_idx]
    sorted_embeddings = embeddings[sort_idx]

    lookback_td = pd.Timedelta(hours=lookback_hours)

    for i, bar_time in enumerate(target_index):
        # Normalize timezone for comparison
        bt = bar_time.tz_localize(None) if bar_time.tzinfo else bar_time

        # Find articles in [bar_time - lookback, bar_time]
        window_start = bt - lookback_td
        mask = (sorted_times >= window_start) & (sorted_times <= bt)
        window_indices = np.where(mask)[0]

        if len(window_indices) > 0:
            # Take most recent max_articles
            selected = window_indices[-max_articles:]
            for j, idx in enumerate(selected):
                result[i, j] = sorted_embeddings[idx]

    return result


def create_empty_embeddings(
    n_bars: int,
    max_articles: int = 10,
) -> np.ndarray:
    """Create zero-filled embeddings array when no news data is available."""
    return np.zeros((n_bars, max_articles, FINBERT_DIM), dtype=np.float32)
