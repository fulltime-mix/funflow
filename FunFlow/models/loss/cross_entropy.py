"""CrossEntropyLoss: Standard cross entropy loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from FunFlow.registry import LOSSES


@LOSSES.register("CrossEntropyLoss")
@LOSSES.register("CE")
class CrossEntropyLoss(nn.Module):
    """Cross entropy loss with optional class weights and label smoothing."""

    def __init__(
        self,
        weight: Optional[List[float]] = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        """
        Args:
            weight: Class weights for handling imbalanced classes
            label_smoothing: Label smoothing coefficient, 0.0 for no smoothing
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        weight_tensor = torch.tensor(weight, dtype=torch.float32) if weight else None
        self.register_buffer("weight", weight_tensor)
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> dict:
        """
        Args:
            inputs: Dict containing model logits with shape (B, C)
            targets: Dict containing true labels with shape (B,)

        Returns:
            Loss tensor or dict with loss field
        """
        labels = targets.get("label", targets.get("labels", None))
        weight = (
            self.weight.to(inputs["logits"].device) if self.weight is not None else None
        )
        loss = F.cross_entropy(
            inputs["logits"],
            labels,
            weight=weight,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction,
        )
        return loss


__all__ = ["CrossEntropyLoss"]
