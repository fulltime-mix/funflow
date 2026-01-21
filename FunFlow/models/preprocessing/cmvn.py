"""GlobalCMVN: Global Cepstral Mean and Variance Normalization for audio features."""

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from FunFlow.registry import PREPROCESSINGS
from FunFlow.logger import get_logger

logger = get_logger("FunFlow")


@PREPROCESSINGS.register("GlobalCMVN")
@PREPROCESSINGS.register("CMVN")
class GlobalCMVN(nn.Module):
    """Global Cepstral Mean and Variance Normalization.

    Normalizes input features using: x_norm = (x - mean) / std
    """

    def __init__(
        self,
        cmvn_file: Optional[str] = None,
        norm_var: bool = False,
        eps: float = 1e-8,
    ):
        """
        Args:
            cmvn_file: Path to CMVN statistics file (JSON format)
            norm_var: Whether to normalize variance
            eps: Small constant to prevent division by zero
        """
        super().__init__()
        self.norm_var = norm_var
        self.eps = eps

        self.register_buffer("mean", None)
        self.register_buffer("std", None)

        if cmvn_file is not None:
            self.load_cmvn(cmvn_file)

    def load_cmvn(self, cmvn_file: str):
        """Load CMVN statistics from JSON file."""
        cmvn_path = Path(cmvn_file)
        if not cmvn_path.exists():
            logger.warning(f"CMVN file not found: {cmvn_file}")
            return

        try:
            with open(cmvn_path, "r", encoding="utf-8") as f:
                cmvn_stats = json.load(f)

            mean = torch.tensor(cmvn_stats["mean"], dtype=torch.float32)
            std = torch.tensor(cmvn_stats["std"], dtype=torch.float32)

            self.register_buffer("mean", mean)
            self.register_buffer("std", std)

            logger.info(f"Loaded CMVN from {cmvn_file}, feature dim: {len(mean)}")
        except Exception as e:
            logger.warning(f"Failed to load CMVN from {cmvn_file}: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (B, T, D) or (T, D)

        Returns:
            Normalized features
        """
        if self.mean is None or self.std is None:
            return x

        mean = self.mean.view(1, 1, -1) if x.dim() == 3 else self.mean.view(1, -1)

        if self.norm_var:
            std = self.std.view(1, 1, -1) if x.dim() == 3 else self.std.view(1, -1)
            std = std.clamp(min=self.eps)
            return (x - mean) / std
        else:
            return x - mean


__all__ = ["GlobalCMVN"]
