"""GlobalAveragePooling Neck: Global average pooling."""

import torch
import torch.nn as nn

from FunFlow.registry import NECKS


@NECKS.register("GlobalAveragePooling")
@NECKS.register("GAP")
class GlobalAveragePooling(nn.Module):
    """Global average pooling neck for spatial dimension reduction."""

    def __init__(self, in_channels: int = None, keep_dim: bool = False, **kwargs):
        """
        Args:
            in_channels: Input channels
            keep_dim: Whether to keep dimensions
        """
        super().__init__()
        self.out_channels = in_channels
        self.keep_dim = keep_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor
                - (B, C, H, W): 4D features
                - (B, L, C): 3D features
                - (B, C): 2D features (returns directly)

        Returns:
            (B, C) tensor or same dimensions if keep_dim=True
        """
        if x.dim() == 4:
            if self.keep_dim:
                return x.mean(dim=[2, 3], keepdim=True)
            return x.mean(dim=[2, 3])
        elif x.dim() == 3:
            if self.keep_dim:
                return x.mean(dim=1, keepdim=True)
            return x.mean(dim=1)
        else:
            return x


__all__ = ["GlobalAveragePooling"]
