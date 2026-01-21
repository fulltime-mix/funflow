"""Pooling head with mask-aware pooling for variable-length sequences."""

import torch
import torch.nn as nn
from typing import Optional
import torch.quantization as quant

from FunFlow.registry import HEADS


@HEADS.register("PoolingHead")
class PoolingHead(nn.Module):
    """Mask-aware pooling + linear classification head.

    Supports variable-length sequences through masked global pooling.

    Args:
        in_channels: Input feature dimension
        num_classes: Number of classes
        pooling: Pooling method ('mean' or 'max')
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        pooling: str = "mean",
        dropout: float = 0.5,
        quantization: bool = False,
    ):
        super(PoolingHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.pooling = pooling

        if pooling not in ["mean", "max"]:
            raise ValueError(f"pooling must be 'mean' or 'max', got {pooling}")

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_channels, num_classes)
        self.quant = quant.QuantStub() if quantization else nn.Identity()
        self.dequant = quant.DeQuantStub() if quantization else nn.Identity()

    def _masked_pooling(self, x: torch.Tensor, x_lens: torch.Tensor) -> torch.Tensor:
        """Mask-aware global pooling.

        Args:
            x: Input tensor (B, T, D)
            x_lens: Sequence lengths (B,)

        Returns:
            Pooled features (B, D)
        """
        batch_size, max_len, dim = x.shape
        device = x.device

        mask = torch.arange(max_len, device=device).unsqueeze(0) < x_lens.unsqueeze(1)

        if self.pooling == "mean":
            mask_expanded = mask.unsqueeze(-1).float()
            x_masked = x * mask_expanded

            valid_lengths = x_lens.float().unsqueeze(-1)
            valid_lengths = torch.clamp(valid_lengths, min=1.0)

            pooled = x_masked.sum(dim=1) / valid_lengths

        elif self.pooling == "max":
            mask_expanded = mask.unsqueeze(-1)
            x_masked = x.clone()
            x_masked[~mask_expanded.expand_as(x)] = float("-inf")

            pooled = x_masked.max(dim=1)[0]

        return pooled

    def forward(
        self, x: torch.Tensor, x_lens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input features (B, T, D)
            x_lens: Sequence lengths (B,), None assumes all sequences are max length

        Returns:
            Classification logits (B, num_classes)
        """
        if x_lens is None:
            batch_size, max_len, _ = x.shape
            x_lens = torch.full(
                (batch_size,), max_len, dtype=torch.long, device=x.device
            )

        pooled = self._masked_pooling(x, x_lens)

        pooled = self.quant(pooled)
        pooled = self.dropout(pooled)

        logits = self.fc(pooled)
        logits = self.dequant(logits)

        return logits


if __name__ == "__main__":
    head = PoolingHead(in_channels=256, num_classes=2, pooling="mean")

    x = torch.randn(4, 100, 256)
    x_lens = torch.tensor([100, 95, 87, 73])

    logits = head(x, x_lens)
    print(f"Input shape: {x.shape}")
    print(f"Sequence lengths: {x_lens}")
    print(f"Output logits shape: {logits.shape}")

    logits_no_lens = head(x)
    print(f"Output logits shape (without x_lens): {logits_no_lens.shape}")
