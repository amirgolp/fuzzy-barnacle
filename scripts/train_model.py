#!/usr/bin/env python3
"""Train TemporalFusionSignalNet for a single asset.

Usage:
    python scripts/train_model.py --symbol GC=F
    python scripts/train_model.py --symbol BTC-USD --epochs 100 --device cuda
    python scripts/train_model.py --symbol GC=F --dataset data/ml/GCF_dataset.h5
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
    parser = argparse.ArgumentParser(description="Train ML signal model")
    parser.add_argument("--symbol", required=True, help="Ticker symbol")
    parser.add_argument(
        "--dataset", type=Path, default=None,
        help="Pre-built HDF5 dataset path (builds fresh if not provided)"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu/auto)")
    parser.add_argument(
        "--save-dir", type=Path, default=Path("models"),
        help="Directory for checkpoints"
    )
    parser.add_argument(
        "--rl", action="store_true",
        help="Run RL fine-tuning after supervised training"
    )
    args = parser.parse_args()

    import torch

    from quantdash.ml.config import ASSET_CONFIGS, TrainingConfig
    from quantdash.ml.data.splits import walk_forward_splits
    from quantdash.ml.models.signal_net import TemporalFusionSignalNet

    asset_config = ASSET_CONFIGS.get(args.symbol)
    if asset_config is None:
        raise ValueError(f"No config for {args.symbol}. Available: {list(ASSET_CONFIGS.keys())}")

    # Build or load dataset
    if args.dataset is not None and args.dataset.exists():
        from quantdash.ml.data.dataset import TradingSignalDataset, load_dataset_hdf5
        logger.info("Loading dataset from %s", args.dataset)
        arrays = load_dataset_hdf5(args.dataset)
        dataset = TradingSignalDataset(
            price_features=arrays["price"],
            volume_features=arrays["volume"],
            pattern_features=arrays["pattern"],
            macro_session_features=arrays["macro_session"],
            cross_asset_features=arrays.get("cross_asset"),
            labels=arrays["labels"],
            news_embeddings=arrays.get("news"),
            lookback=asset_config.arch_config.lookback_1h,
            config=asset_config.arch_config,
        )
    else:
        logger.info("Building dataset from scratch...")
        from quantdash.ml.data.builder import build_dataset, get_dataset_path
        save_path = args.dataset or get_dataset_path(args.symbol)
        dataset = build_dataset(
            symbol=args.symbol,
            save_path=save_path,
        )

    logger.info("Dataset: %d samples", len(dataset))

    # Create model
    model = TemporalFusionSignalNet(asset_config.arch_config)
    logger.info("Model parameters: %s", f"{model.count_parameters():,}")

    # Training config
    train_config = asset_config.training_config.model_copy()
    if args.epochs is not None:
        train_config.max_epochs = args.epochs
    if args.batch_size is not None:
        train_config.batch_size = args.batch_size
    if args.lr is not None:
        train_config.learning_rate = args.lr

    # Walk-forward splits
    folds = walk_forward_splits(
        len(dataset.labels),
        config=asset_config.walk_forward_config,
    )
    logger.info("Walk-forward folds: %d", len(folds))

    # Supervised training
    from quantdash.ml.training.supervised import SupervisedTrainer

    trainer = SupervisedTrainer(
        model=model,
        train_config=train_config,
        device=args.device,
        save_dir=args.save_dir,
    )

    history = trainer.train_walk_forward(dataset, folds)
    logger.info("Supervised training complete")

    # Optional RL fine-tuning
    if args.rl:
        logger.info("Starting RL fine-tuning...")
        from quantdash.ml.training.rl_finetune import train_ppo

        rl_metrics = train_ppo(
            model=trainer.model,
            dataset=dataset,
            rl_config=asset_config.rl_config,
            fee_bps=asset_config.labeling_config.fee_bps,
            device=args.device or "cpu",
            save_dir=args.save_dir,
        )
        logger.info("RL fine-tuning complete: %s", rl_metrics)

    logger.info("Done. Checkpoints saved to %s", args.save_dir)


if __name__ == "__main__":
    main()
