"""LinearHead: Simple linear classification head."""

import torch
import torch.nn as nn

from FunFlow.registry import HEADS


@HEADS.register("LinearHead")
@HEADS.register("Linear")
class LinearHead(nn.Module):
    """Linear classification head with optional dropout.

    Structure: [GlobalPool] -> Dropout -> Linear
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout: float = 0.0,
    ):
        """
        Args:
            in_channels: Input channels/feature dimension
            num_classes: Number of classes
            dropout: Dropout ratio
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature tensor
                - (B, C): 2D features
                - (B, C, H, W): 4D features, will apply global pooling
                - (B, L, C): 3D sequence features, will apply average pooling

        Returns:
            (B, num_classes) logits
        """
        if x.dim() == 4:
            x = x.mean(dim=[2, 3])
        elif x.dim() == 3:
            x = x.mean(dim=1)

        x = self.dropout(x)
        return self.fc(x)


__all__ = ["LinearHead"]
