#!/usr/bin/env python3
"""Pre-compute FinBERT embeddings from news articles for ML training.

Usage:
    python scripts/precompute_finbert.py --symbol GC=F --output data/ml/GCF_news.h5
    python scripts/precompute_finbert.py --symbol BTC-USD --source rss
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


def collect_articles_stub(
    symbol: str,
    start: datetime | None = None,
    end: datetime | None = None,
) -> list[dict]:
    """Placeholder article collector.

    In production, replace with:
    - Google News RSS feed parsing
    - Yahoo Finance news API
    - NewsAPI.org
    - Alpha Vantage news sentiment

    Each article dict should have:
        - "text": str (headline + body)
        - "timestamp": datetime
        - "source": str
    """
    logger.warning(
        "Using stub article collector. "
        "Implement a real news source for production use."
    )
    return []


def main():
    parser = argparse.ArgumentParser(description="Pre-compute FinBERT embeddings")
    parser.add_argument("--symbol", required=True, help="Ticker symbol (e.g., GC=F)")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output HDF5 path (default: data/ml/{symbol}_news.h5)"
    )
    parser.add_argument(
        "--source", default="stub",
        choices=["stub", "rss", "newsapi"],
        help="News source to use"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu/auto)")
    parser.add_argument(
        "--articles-json", type=Path, default=None,
        help="Load articles from JSON file instead of fetching"
    )
    args = parser.parse_args()

    # Determine output path
    if args.output is None:
        safe = args.symbol.replace("=", "").replace("-", "_").replace("^", "")
        args.output = Path("data/ml") / f"{safe}_news.h5"

    # Collect or load articles
    if args.articles_json is not None:
        logger.info("Loading articles from %s", args.articles_json)
        with open(args.articles_json) as f:
            raw_articles = json.load(f)
        articles = [
            {
                "text": a["text"],
                "timestamp": datetime.fromisoformat(a["timestamp"]),
                "source": a.get("source", "unknown"),
            }
            for a in raw_articles
        ]
    else:
        articles = collect_articles_stub(args.symbol)

    if not articles:
        logger.warning("No articles found. Creating empty embeddings file.")
        from quantdash.ml.data.news_embeddings import (
            create_empty_embeddings,
            save_embeddings_hdf5,
        )

        embeddings = create_empty_embeddings(0, max_articles=1)
        save_embeddings_hdf5(
            args.output,
            timestamps=np.array([]),
            embeddings=np.zeros((0, 768), dtype=np.float32),
        )
        logger.info("Empty embeddings saved to %s", args.output)
        return

    # Extract texts and timestamps
    texts = [a["text"] for a in articles]
    timestamps = [a["timestamp"] for a in articles]

    logger.info("Computing FinBERT embeddings for %d articles...", len(texts))

    from quantdash.ml.data.news_embeddings import (
        compute_finbert_embeddings,
        save_embeddings_hdf5,
    )

    embeddings = compute_finbert_embeddings(
        texts, batch_size=args.batch_size, device=args.device
    )

    save_embeddings_hdf5(args.output, timestamps, embeddings)
    logger.info(
        "Done. Saved %d embeddings (%s) to %s",
        len(embeddings), embeddings.shape, args.output
    )


if __name__ == "__main__":
    main()
