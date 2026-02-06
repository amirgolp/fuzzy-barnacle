#!/usr/bin/env python3
"""Run batch inference across configured assets.

Usage:
    python scripts/run_inference.py --symbols GC=F BTC-USD
    python scripts/run_inference.py --all --model-dir models/
    python scripts/run_inference.py --symbols GC=F --output signals.json
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run ML signal inference")
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="Ticker symbols to run inference on"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run inference on all configured assets"
    )
    parser.add_argument(
        "--model-dir", type=Path, default=Path("models"),
        help="Directory containing model checkpoints"
    )
    parser.add_argument(
        "--equity", type=float, default=100_000.0,
        help="Current portfolio equity"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output JSON file for signals"
    )
    parser.add_argument("--device", default=None, help="Inference device")
    args = parser.parse_args()

    from quantdash.ml.config import ASSET_CONFIGS
    from quantdash.ml.inference.batch_runner import run_batch_inference

    if args.all:
        symbols = list(ASSET_CONFIGS.keys())
    elif args.symbols:
        symbols = args.symbols
    else:
        parser.error("Specify --symbols or --all")
        return

    logger.info("Running inference for: %s", symbols)

    results = run_batch_inference(
        symbols=symbols,
        model_dir=args.model_dir,
        equity=args.equity,
        device=args.device,
    )

    # Output
    for r in results:
        if "error" in r:
            logger.warning("%s: %s", r["symbol"], r["error"])
        else:
            logger.info(
                "%s: %s (conf=%.3f, size=%.2f×)",
                r["symbol"], r["action"],
                r["confidence"], r["position"]["size"],
            )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Signals saved to %s", args.output)
    else:
        print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
