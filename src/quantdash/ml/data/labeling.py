"""Triple-barrier labeling for supervised training targets.

For each bar, look forward and determine which barrier is hit first:
- Upper (TP): entry + tp_mult × ATR → label = +1 (BUY)
- Lower (SL): entry - sl_mult × ATR → label = -1 (SELL)
- Time (max_bars): → label = 0 (HOLD)

Barriers are adjusted for round-trip trading fees.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantdash.ml.config import LabelingConfig


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return true_range.rolling(window=period, min_periods=1).mean()


def triple_barrier_label(
    df: pd.DataFrame,
    config: LabelingConfig | None = None,
) -> pd.Series:
    """Apply triple-barrier labeling to OHLCV data.

    Args:
        df: DataFrame with at least 'open', 'high', 'low', 'close' columns.
        config: Labeling parameters (defaults to LabelingConfig()).

    Returns:
        Series of int labels: +1 (BUY), 0 (HOLD), -1 (SELL).
        NaN for bars without enough forward data.
    """
    if config is None:
        config = LabelingConfig()

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(df)

    atr = compute_atr(df, period=config.atr_period).values

    # Fee adjustment: shift barriers inward by round-trip fee
    fee_adj = close * (config.fee_bps / 10_000) * 2  # round-trip

    labels = np.full(n, np.nan)

    for i in range(n):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue

        entry = close[i]
        upper = entry + config.tp_mult * atr[i] - fee_adj[i]
        lower = entry - config.sl_mult * atr[i] + fee_adj[i]

        # Invalid barriers (fees exceed barrier)
        if upper <= entry or lower >= entry:
            labels[i] = 0
            continue

        end_idx = min(i + config.max_bars, n - 1)
        if i >= n - 1:
            continue

        label = 0  # default: time barrier
        for j in range(i + 1, end_idx + 1):
            hit_upper = high[j] >= upper
            hit_lower = low[j] <= lower

            if hit_upper and hit_lower:
                # Both barriers hit in same bar — use close direction
                if close[j] >= entry:
                    label = 1
                else:
                    label = -1
                break
            elif hit_upper:
                label = 1
                break
            elif hit_lower:
                label = -1
                break

        labels[i] = label

    return pd.Series(labels, index=df.index, name="label", dtype="float64")


def label_distribution(labels: pd.Series) -> dict[str, int | float]:
    """Summarize label distribution."""
    valid = labels.dropna()
    counts = valid.value_counts().to_dict()
    total = len(valid)

    return {
        "total": total,
        "buy": int(counts.get(1.0, 0)),
        "hold": int(counts.get(0.0, 0)),
        "sell": int(counts.get(-1.0, 0)),
        "buy_pct": round(counts.get(1.0, 0) / total * 100, 1) if total > 0 else 0,
        "hold_pct": round(counts.get(0.0, 0) / total * 100, 1) if total > 0 else 0,
        "sell_pct": round(counts.get(-1.0, 0) / total * 100, 1) if total > 0 else 0,
    }
