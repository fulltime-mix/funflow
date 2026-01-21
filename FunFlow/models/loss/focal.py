"""FocalLoss: Focal loss for handling class imbalance."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from FunFlow.registry import LOSSES


@LOSSES.register("FocalLoss")
@LOSSES.register("Focal")
class FocalLoss(nn.Module):
    """Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Reduces weight of easy examples and focuses on hard examples.
    """

    def __init__(
        self,
        alpha=None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """
        Args:
            alpha: Class balance weight
                - None: No alpha weighting
                - float: Same weight for all classes (only scales loss)
                - List/Tensor: Per-class weights with shape (C,)
            gamma: Focusing parameter, larger values focus more on hard examples
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                alpha = torch.tensor(alpha)
            elif isinstance(alpha, (int, float)):
                alpha = torch.tensor(alpha)
        self.register_buffer("alpha", alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> dict:
        """
        Args:
            inputs: Model output dict with 'logits' key, shape (B, C)
            targets: Labels or dict containing labels, shape (B,)

        Returns:
            Loss tensor or dict with loss field
        """
        labels = targets.get("label", targets.get("labels", None))
        ce_loss = F.cross_entropy(inputs["logits"], labels, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if self.alpha.dim() == 0:
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha[labels]
            focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            loss = focal_loss.mean()
        elif self.reduction == "sum":
            loss = focal_loss.sum()
        else:
            loss = focal_loss

        return loss


__all__ = ["FocalLoss"]
