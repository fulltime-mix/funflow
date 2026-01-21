"""Identity Neck: Pass-through operation."""

import torch
import torch.nn as nn

from FunFlow.registry import NECKS


@NECKS.register("Identity")
class Identity(nn.Module):
    """Identity transformation that passes input through unchanged."""

    def __init__(self, in_channels: int = None, **kwargs):
        """
        Args:
            in_channels: Input channels (for interface consistency)
        """
        super().__init__()
        self.out_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


__all__ = ["Identity"]
