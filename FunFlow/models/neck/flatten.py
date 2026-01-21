"""Flatten Neck: Flatten operation."""

import torch
import torch.nn as nn

from FunFlow.registry import NECKS


@NECKS.register("Flatten")
class Flatten(nn.Module):
    """Flatten neck to convert high-dimensional features to 2D (B, C)."""

    def __init__(
        self, in_channels: int = None, start_dim: int = 1, end_dim: int = -1, **kwargs
    ):
        """
        Args:
            in_channels: Input channels (for interface consistency)
            start_dim: Start dimension to flatten
            end_dim: End dimension to flatten
        """
        super().__init__()
        self.out_channels = in_channels
        self.start_dim = start_dim
        self.end_dim = end_dim
        self.flatten = nn.Flatten(start_dim, end_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of any shape

        Returns:
            Flattened tensor
        """
        return self.flatten(x)


__all__ = ["Flatten"]
