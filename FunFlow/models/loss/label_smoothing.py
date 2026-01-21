"""LabelSmoothingLoss: Label smoothing loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from FunFlow.registry import LOSSES


@LOSSES.register("LabelSmoothingLoss")
@LOSSES.register("LabelSmoothing")
class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss to prevent overconfident predictions.

    Uses soft labels instead of hard labels.
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        reduction: str = "mean",
    ):
        """
        Args:
            smoothing: Smoothing coefficient, between 0.0-1.0
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> dict:
        """
        Args:
            logits: Model logits with shape (B, C)
            targets: Labels with shape (B,)

        Returns:
            Loss tensor or dict with loss field
        """
        labels = targets.pop("label", targets.pop("labels", None))

        num_classes = logits.size(-1)

        with torch.no_grad():
            smooth_labels = torch.zeros_like(logits)
            smooth_labels.fill_(self.smoothing / (num_classes - 1))
            smooth_labels.scatter_(1, labels.unsqueeze(1), 1 - self.smoothing)

        log_probs = F.log_softmax(logits, dim=-1)
        loss_tensor = -(smooth_labels * log_probs).sum(dim=-1)

        if self.reduction == "mean":
            loss = loss_tensor.mean()
        elif self.reduction == "sum":
            loss = loss_tensor.sum()
        else:
            loss = loss_tensor

        return loss


__all__ = ["LabelSmoothingLoss"]
