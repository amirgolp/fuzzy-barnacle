"""Loss functions for supervised training.

- Focal Loss: handles class imbalance (hold >> buy/sell)
- Confidence Loss: BCE auxiliary loss on confidence head
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification.

    Addresses class imbalance by down-weighting well-classified examples.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Per-class weights [sell, hold, buy]. Default [0.25, 0.5, 0.25].
        gamma: Focusing parameter. Higher = more focus on hard examples.
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(
        self,
        alpha: list[float] | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        if alpha is None:
            alpha = [0.25, 0.5, 0.25]
        self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch, 3] raw logits (not softmaxed)
            targets: [batch] class indices {0=sell, 1=hold, 2=buy}

        Returns:
            Scalar loss
        """
        probs = F.softmax(logits, dim=-1)

        # Gather probability of correct class
        targets_one_hot = F.one_hot(targets, num_classes=3).float()
        p_t = (probs * targets_one_hot).sum(dim=-1)  # [batch]

        # Focal modulation
        focal_weight = (1 - p_t) ** self.gamma

        # Per-class alpha weighting
        alpha = self.alpha.to(logits.device)
        alpha_t = (alpha.unsqueeze(0) * targets_one_hot).sum(dim=-1)  # [batch]

        # Focal loss
        loss = -alpha_t * focal_weight * torch.log(p_t.clamp(min=1e-8))

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class ConfidenceLoss(nn.Module):
    """BCE loss on confidence head.

    Target: 1 if the model's predicted class matches the true label, 0 otherwise.
    Teaches the model to output high confidence when it's correct.
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(
        self,
        confidence: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            confidence: [batch, 1] sigmoid output
            logits: [batch, 3] raw logits
            targets: [batch] class indices

        Returns:
            Scalar BCE loss
        """
        predicted = logits.argmax(dim=-1)  # [batch]
        correct = (predicted == targets).float()  # [batch]

        return self.bce(confidence.squeeze(-1), correct)


class CombinedLoss(nn.Module):
    """Combined Focal Loss + weighted Confidence Loss."""

    def __init__(
        self,
        alpha: list[float] | None = None,
        gamma: float = 2.0,
        confidence_weight: float = 0.1,
    ):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.confidence = ConfidenceLoss()
        self.confidence_weight = confidence_weight

    def forward(
        self,
        logits: torch.Tensor,
        confidence: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            Dict with 'total', 'focal', 'confidence' loss values.
        """
        focal_loss = self.focal(logits, targets)
        conf_loss = self.confidence(confidence, logits, targets)
        total = focal_loss + self.confidence_weight * conf_loss

        return {
            "total": total,
            "focal": focal_loss,
            "confidence": conf_loss,
        }
