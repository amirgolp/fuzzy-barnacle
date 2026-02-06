"""Supervised training loop for TemporalFusionSignalNet.

Phase 1 training: Focal Loss + confidence auxiliary loss.
Walk-forward validation, OneCycleLR scheduler, FP16 mixed precision.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from quantdash.ml.config import ModelConfig, TrainingConfig
from quantdash.ml.data.dataset import TradingSignalDataset
from quantdash.ml.data.splits import SplitFold, get_fold_indices
from quantdash.ml.models.signal_net import TemporalFusionSignalNet
from quantdash.ml.training.callbacks import (
    CheckpointSaver,
    EarlyStopping,
    MetricTracker,
)
from quantdash.ml.training.losses import CombinedLoss

logger = logging.getLogger(__name__)


def _compute_metrics(
    all_preds: list[int],
    all_targets: list[int],
) -> dict[str, float]:
    """Compute classification metrics."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    preds = np.array(all_preds)
    targets = np.array(all_targets)

    return {
        "accuracy": accuracy_score(targets, preds),
        "f1_macro": f1_score(targets, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(targets, preds, average="weighted", zero_division=0),
        "precision_macro": precision_score(targets, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(targets, preds, average="macro", zero_division=0),
    }


class SupervisedTrainer:
    """Supervised training for TemporalFusionSignalNet.

    Handles:
    - Training loop with Focal Loss + confidence loss
    - FP16 mixed precision
    - OneCycleLR scheduler
    - Early stopping and checkpointing
    - Metric tracking
    """

    def __init__(
        self,
        model: TemporalFusionSignalNet,
        train_config: TrainingConfig | None = None,
        device: str | None = None,
        save_dir: Path | None = None,
    ):
        self.model = model
        self.config = train_config or TrainingConfig()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        # Loss
        self.criterion = CombinedLoss(
            alpha=self.config.focal_alpha,
            gamma=self.config.focal_gamma,
            confidence_weight=self.config.confidence_loss_weight,
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Mixed precision
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.config.use_fp16 and self.device.type == "cuda")
        self.use_amp = self.config.use_fp16 and self.device.type == "cuda"

        # Callbacks
        self.save_dir = save_dir or Path("models")
        self.checkpoint = CheckpointSaver(self.save_dir, mode="max")
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience, mode="max"
        )
        self.tracker = MetricTracker()

    def _train_epoch(
        self,
        loader: DataLoader,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ) -> dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        total_focal = 0.0
        total_conf = 0.0
        n_batches = 0
        all_preds = []
        all_targets = []

        for batch in loader:
            # Move to device
            price = batch["price"].to(self.device)
            volume = batch["volume"].to(self.device)
            pattern = batch["pattern"].to(self.device)
            news = batch["news"].to(self.device)
            macro = batch["macro"].to(self.device)
            cross_asset = batch["cross_asset"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(price, volume, pattern, news, macro, cross_asset)
                losses = self.criterion(
                    outputs["action_logits"], outputs["confidence"], labels
                )

            self.scaler.scale(losses["total"]).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Step OneCycleLR per batch (not per epoch)
            if scheduler is not None:
                scheduler.step()

            total_loss += losses["total"].item()
            total_focal += losses["focal"].item()
            total_conf += losses["confidence"].item()
            n_batches += 1

            preds = outputs["action_logits"].argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().tolist())

        metrics = _compute_metrics(all_preds, all_targets)
        metrics["loss"] = total_loss / max(n_batches, 1)
        metrics["focal_loss"] = total_focal / max(n_batches, 1)
        metrics["conf_loss"] = total_conf / max(n_batches, 1)

        return metrics

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> dict[str, float]:
        """Run one evaluation epoch."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds = []
        all_targets = []

        for batch in loader:
            price = batch["price"].to(self.device)
            volume = batch["volume"].to(self.device)
            pattern = batch["pattern"].to(self.device)
            news = batch["news"].to(self.device)
            macro = batch["macro"].to(self.device)
            cross_asset = batch["cross_asset"].to(self.device)
            labels = batch["label"].to(self.device)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(price, volume, pattern, news, macro, cross_asset)
                losses = self.criterion(
                    outputs["action_logits"], outputs["confidence"], labels
                )

            total_loss += losses["total"].item()
            n_batches += 1

            preds = outputs["action_logits"].argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().tolist())

        metrics = _compute_metrics(all_preds, all_targets)
        metrics["loss"] = total_loss / max(n_batches, 1)

        return metrics

    def train_fold(
        self,
        dataset: TradingSignalDataset,
        fold: SplitFold,
    ) -> dict[str, list[float]]:
        """Train on a single walk-forward fold.

        Args:
            dataset: Full dataset.
            fold: Train/val split indices.

        Returns:
            Training history dict.
        """
        train_idx, val_idx = get_fold_indices(fold)

        # Map fold indices to valid dataset indices
        valid_indices = dataset._valid_indices
        train_mask = np.isin(valid_indices, train_idx)
        val_mask = np.isin(valid_indices, val_idx)

        train_positions = np.where(train_mask)[0].tolist()
        val_positions = np.where(val_mask)[0].tolist()

        if not train_positions or not val_positions:
            logger.warning("Empty train or val set for fold %d", fold.fold_idx)
            return {}

        train_subset = Subset(dataset, train_positions)
        val_subset = Subset(dataset, val_positions)

        train_loader = DataLoader(
            train_subset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

        # Scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            epochs=self.config.max_epochs,
            steps_per_epoch=len(train_loader),
        )

        logger.info(
            "Fold %d: train=%d, val=%d samples",
            fold.fold_idx, len(train_positions), len(val_positions)
        )

        # Reset early stopping for each fold
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience, mode="max"
        )

        for epoch in range(self.config.max_epochs):
            t0 = time.time()

            train_metrics = self._train_epoch(train_loader, scheduler=scheduler)
            val_metrics = self._eval_epoch(val_loader)

            elapsed = time.time() - t0

            self.tracker.update({
                f"fold{fold.fold_idx}_train_loss": train_metrics["loss"],
                f"fold{fold.fold_idx}_val_loss": val_metrics["loss"],
                f"fold{fold.fold_idx}_val_f1": val_metrics["f1_macro"],
                f"fold{fold.fold_idx}_val_acc": val_metrics["accuracy"],
            })

            logger.info(
                "Fold %d Epoch %d/%d [%.1fs] — "
                "train_loss=%.4f, val_loss=%.4f, val_f1=%.4f, val_acc=%.4f",
                fold.fold_idx, epoch + 1, self.config.max_epochs, elapsed,
                train_metrics["loss"], val_metrics["loss"],
                val_metrics["f1_macro"], val_metrics["accuracy"],
            )

            # Checkpoint best model
            self.checkpoint(
                val_metrics["f1_macro"],
                self.model,
                self.optimizer,
                epoch=epoch,
                extra={"fold": fold.fold_idx, "config": self.model.config.model_dump()},
            )

            # Early stopping
            if self.early_stopping(val_metrics["f1_macro"]):
                break

        return self.tracker.history

    def train_walk_forward(
        self,
        dataset: TradingSignalDataset,
        folds: list[SplitFold],
    ) -> dict:
        """Train across all walk-forward folds.

        For each fold, trains from scratch (or could implement ensemble).
        Returns combined metrics across folds.
        """
        all_histories = []

        for fold in folds:
            logger.info("=" * 60)
            logger.info("Starting fold %d", fold.fold_idx)
            logger.info("=" * 60)

            # Reset model weights for each fold
            self._reset_model()

            history = self.train_fold(dataset, fold)
            all_histories.append(history)

        return {
            "fold_histories": all_histories,
            "tracker": self.tracker.history,
        }

    def _reset_model(self) -> None:
        """Reset model to fresh weights."""
        config = self.model.config
        self.model = TemporalFusionSignalNet(config).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.config.use_fp16 and self.device.type == "cuda"
        )
