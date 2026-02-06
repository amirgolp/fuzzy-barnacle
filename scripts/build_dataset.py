#!/usr/bin/env python3
"""Build training dataset for a single asset.

Usage:
    python scripts/build_dataset.py --symbol GC=F
    python scripts/build_dataset.py --symbol BTC-USD --no-cross
    python scripts/build_dataset.py --symbol SPY --save-path data/ml/spy_dataset.h5
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Build ML training dataset")
    parser.add_argument("--symbol", required=True, help="Ticker symbol (e.g., GC=F)")
    parser.add_argument(
        "--save-path", type=Path, default=None,
        help="Output HDF5 path (default: data/ml/{symbol}_dataset.h5)"
    )
    parser.add_argument(
        "--no-cross", action="store_true",
        help="Skip cross-asset data fetching"
    )
    parser.add_argument(
        "--macro-stub", action="store_true",
        help="Use zero-filled macro data (no FRED API key needed)"
    )
    args = parser.parse_args()

    from quantdash.ml.data.builder import build_dataset, get_dataset_path

    save_path = args.save_path or get_dataset_path(args.symbol)

    macro_df = None
    if not args.macro_stub:
        try:
            import os
            if os.environ.get("FRED_API_KEY"):
                from quantdash.ml.data.macro import fetch_all_macro
                macro_df = fetch_all_macro()
                logger.info("Fetched macro data: %d observations", len(macro_df))
            else:
                logger.info("No FRED_API_KEY set, using macro stub")
        except Exception as e:
            logger.warning("Failed to fetch macro data: %s. Using stub.", e)

    dataset = build_dataset(
        symbol=args.symbol,
        macro_df=macro_df,
        save_path=save_path,
        fetch_cross=not args.no_cross,
    )

    logger.info("Dataset built: %d samples", len(dataset))
    logger.info("Saved to: %s", save_path)


if __name__ == "__main__":
    main()
