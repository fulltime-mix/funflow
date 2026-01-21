"""Normalize: Image normalization."""

import torch
import torch.nn as nn
from typing import List, Optional

from FunFlow.registry import PREPROCESSINGS


@PREPROCESSINGS.register("Normalize")
@PREPROCESSINGS.register("ImageNormalize")
class Normalize(nn.Module):
    """Image normalization with mean and std.

    Common presets:
    - ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    - CIFAR10: mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
    """

    PRESETS = {
        "imagenet": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "cifar10": {
            "mean": [0.4914, 0.4822, 0.4465],
            "std": [0.2023, 0.1994, 0.2010],
        },
        "cifar100": {
            "mean": [0.5071, 0.4867, 0.4408],
            "std": [0.2675, 0.2565, 0.2761],
        },
    }

    def __init__(
        self,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        preset: Optional[str] = None,
    ):
        """
        Args:
            mean: Mean values per channel
            std: Standard deviation values per channel
            preset: Preset name ('imagenet', 'cifar10', 'cifar100')
        """
        super().__init__()

        if preset is not None:
            if preset.lower() not in self.PRESETS:
                raise ValueError(
                    f"Unknown preset: {preset}. Available: {list(self.PRESETS.keys())}"
                )
            preset_params = self.PRESETS[preset.lower()]
            mean = preset_params["mean"]
            std = preset_params["std"]

        if mean is None:
            mean = [0.0, 0.0, 0.0]
        if std is None:
            std = [1.0, 1.0, 1.0]

        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor (B, C, H, W), range [0, 1]

        Returns:
            Normalized image tensor
        """
        return (x - self.mean) / self.std


__all__ = ["Normalize"]
