"""
Quantization-friendly TCN (Temporal Convolutional Network) implementation.

Supports QAT (Quantization-Aware Training) with FloatFunctional operations.
Input format: BTD (Batch, Time, Dimension)
Output format: BT'D' (supports downsampling)
"""

import torch
import torch.nn as nn
from typing import List, Optional

try:
    from torch.ao.nn.quantized import FloatFunctional
except ImportError:
    from torch.nn.quantized import FloatFunctional

from FunFlow.registry import BACKBONES

__all__ = ["TCNBackbone", "TemporalBlock"]


class TemporalBlock(nn.Module):
    """TCN building block with residual connection.

    Contains two conv layers with BatchNorm + ReLU + Dropout.
    Supports stride for downsampling.

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        kernel_size: Convolution kernel size.
        stride: Stride for downsampling.
        dilation: Dilation rate.
        dropout: Dropout probability.
        quantization: Enable quantization-friendly operations.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        dropout: float = 0.1,
        quantization: bool = False,
    ):
        super(TemporalBlock, self).__init__()
        self.quantization = quantization

        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = None
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )

        if quantization:
            self.add_func = FloatFunctional()
        else:
            self.add_func = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, C, T).

        Returns:
            Output tensor (B, C', T').
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        if self.quantization and self.add_func is not None:
            out = self.add_func.add(out, residual)
        else:
            out = out + residual

        return out


@BACKBONES.register("TCN")
@BACKBONES.register("TCNBackbone")
class TCNBackbone(nn.Module):
    """Temporal Convolutional Network Backbone.

    Feature extraction for temporal data with downsampling and quantization support.

    Args:
        input_dim: Input feature dimension.
        channels: Channel count for each layer.
        kernel_size: Convolution kernel size.
        dropout: Dropout probability.
        strides: Stride for each layer (default: all 1s).
        quantization: Enable quantization-friendly operations.

    Input:
        x: (B, T, D) - Batch, Time, Dimension
        x_lens: (B,) - Actual sequence lengths (optional)

    Output:
        x: (B, T', D') - Downsampled time and new feature dimension
        x_lens: (B,) - Output sequence lengths
    """

    def __init__(
        self,
        input_dim: int,
        channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.1,
        strides: Optional[List[int]] = None,
        quantization: bool = False,
    ):
        super(TCNBackbone, self).__init__()
        self.input_dim = input_dim
        self.output_dim = channels[-1] if channels else input_dim
        self.quantization = quantization
        self.kernel_size = kernel_size

        if strides is None:
            strides = [1] * len(channels)
        assert len(strides) == len(
            channels
        ), f"strides length ({len(strides)}) must match channels length ({len(channels)})"

        self.strides = strides

        layers = []
        num_levels = len(channels)
        for i in range(num_levels):
            dilation = 2**i
            in_channels = input_dim if i == 0 else channels[i - 1]
            out_channels = channels[i]
            stride = strides[i]

            layers.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    dropout=dropout,
                    quantization=quantization,
                )
            )

        self.network = nn.Sequential(*layers)

    def _compute_output_length(self, input_length: int, layer_idx: int) -> int:
        """Compute output sequence length after specified layer.

        Args:
            input_length: Input sequence length.
            layer_idx: Layer index.

        Returns:
            Output sequence length.
        """
        stride = self.strides[layer_idx]
        kernel_size = self.kernel_size
        dilation = 2**layer_idx
        padding = (kernel_size - 1) * dilation // 2

        output_length = (
            input_length + 2 * padding - dilation * (kernel_size - 1) - 1
        ) // stride + 1
        return output_length

    def _compute_output_lengths(self, x_lens: torch.Tensor) -> torch.Tensor:
        """Compute output lengths after all layers.

        Args:
            x_lens: Input sequence lengths (B,).

        Returns:
            Output sequence lengths (B,).
        """
        device = x_lens.device

        output_lens = x_lens.clone()
        for i in range(len(self.strides)):
            output_lens = torch.tensor(
                [
                    self._compute_output_length(length.item(), i)
                    for length in output_lens
                ],
                device=device,
            )

        return output_lens

    def forward(self, x: torch.Tensor, x_lens: Optional[torch.Tensor] = None) -> tuple:
        """Forward pass.

        Args:
            x: Input tensor (B, T, D).
            x_lens: Sequence lengths (B,), optional.

        Returns:
            x: Output tensor (B, T', D').
            x_lens: Output sequence lengths (B,).
        """
        batch_size = x.size(0)

        x = x.transpose(1, 2)
        x = self.network(x)
        x = x.transpose(1, 2)

        if x_lens is not None:
            x_lens = self._compute_output_lengths(x_lens)
        else:
            x_lens = torch.full(
                (batch_size,), x.size(1), dtype=torch.long, device=x.device
            )

        return x, x_lens

    def get_output_dim(self) -> int:
        """Returns output feature dimension."""
        return self.output_dim


if __name__ == "__main__":
    model = TCNBackbone(
        input_dim=12,
        channels=[64, 128, 256],
        kernel_size=3,
        dropout=0.1,
        strides=[2, 2, 1],
        quantization=False,
    )

    x = torch.randn(4, 2048, 12)
    x_lens = torch.tensor([2048, 1950, 1873, 1635])

    print("Testing with variable length sequences:")
    print(f"Input shape: {x.shape}")
    print(f"Sequence lengths: {x_lens}")

    out, out_lens = model(x, x_lens)
    print(f"Output shape (with x_lens): {out.shape}")
    print(f"Output lengths: {out_lens}")

    out_no_lens, out_lens_no_lens = model(x)
    print(f"Output shape (without x_lens): {out_no_lens.shape}")
    print(f"Output lengths (without x_lens): {out_lens_no_lens}")
