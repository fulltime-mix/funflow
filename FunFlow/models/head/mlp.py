"""MLPHead: Multi-layer perceptron classification head."""

import torch
import torch.nn as nn
from typing import List

from FunFlow.registry import HEADS


@HEADS.register("MLPHead")
@HEADS.register("MLP")
class MLPHead(nn.Module):
    """Multi-layer perceptron classification head.

    Structure: [GlobalPool] -> (Linear -> BN -> Activation -> Dropout) x N -> Linear
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_dims: List[int] = None,
        dropout: float = 0.5,
        activation: str = "relu",
    ):
        """
        Args:
            in_channels: Input channels
            num_classes: Number of classes
            hidden_dims: Hidden layer dimensions, default [256]
            dropout: Dropout ratio
            activation: Activation function type ('relu', 'gelu')
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        if hidden_dims is None:
            hidden_dims = [256]

        act_fn = nn.ReLU(inplace=True) if activation == "relu" else nn.GELU()

        layers = []
        prev_dim = in_channels

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    act_fn,
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature tensor, supports 2D/3D/4D input

        Returns:
            (B, num_classes) logits
        """
        if x.dim() == 4:
            x = x.mean(dim=[2, 3])
        elif x.dim() == 3:
            x = x.mean(dim=1)

        return self.layers(x)


__all__ = ["MLPHead"]
