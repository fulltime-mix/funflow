"""
ResNet 1D Backbone

1D convolutional residual network for time series feature extraction.
Input format: BTD (Batch, Time, Dimension)
Output format: BT'D' (with optional downsampling)
"""

import torch
import torch.nn as nn
from typing import List, Optional

try:
    from torch.ao.nn.quantized import FloatFunctional
except ImportError:
    from torch.nn.quantized import FloatFunctional

from FunFlow.registry import BACKBONES

__all__ = ["ResNetBackbone1D", "BottleneckBlock", "BasicBlock"]


class BasicBlock(nn.Module):
    """Basic residual block with two 3x1 convolutions and residual connection.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        stride: Stride for downsampling.
        dropout: Dropout probability.
        quantization: Enable quantization-friendly operations.
    """

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.1,
        quantization: bool = False,
    ):
        super(BasicBlock, self).__init__()
        self.quantization = quantization

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(out_channels),
            )

        if quantization:
            self.add_func = FloatFunctional()
        else:
            self.add_func = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, T).

        Returns:
            Output tensor of shape (B, C', T').
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        if self.quantization and self.add_func is not None:
            out = self.add_func.add(out, residual)
        else:
            out = out + residual
        out = self.relu(out)

        return out


class BottleneckBlock(nn.Module):
    """Bottleneck residual block with 1x1 -> 3x1 -> 1x1 convolution structure.

    Args:
        in_channels: Input channels.
        bottleneck_channels: Bottleneck channels.
        stride: Stride for downsampling.
        dropout: Dropout probability.
        quantization: Enable quantization-friendly operations.
    """

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        bottleneck_channels: int,
        stride: int = 1,
        dropout: float = 0.1,
        quantization: bool = False,
    ):
        super(BottleneckBlock, self).__init__()
        self.quantization = quantization

        out_channels = bottleneck_channels * self.expansion

        self.conv1 = nn.Conv1d(
            in_channels, bottleneck_channels, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(bottleneck_channels)

        self.conv2 = nn.Conv1d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(bottleneck_channels)

        self.conv3 = nn.Conv1d(
            bottleneck_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(out_channels),
            )

        if quantization:
            self.add_func = FloatFunctional()
        else:
            self.add_func = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, T).

        Returns:
            Output tensor of shape (B, C', T').
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        if self.quantization and self.add_func is not None:
            out = self.add_func.add(out, residual)
        else:
            out = out + residual
        out = self.relu(out)

        return out


@BACKBONES.register("ResNet1D")
@BACKBONES.register("ResNetBackbone1D")
class ResNetBackbone1D(nn.Module):
    """ResNet backbone for time series feature extraction with downsampling support.

    Args:
        input_dim: Input feature dimension.
        layers: Number of blocks per stage, e.g., [2, 2, 2, 2].
        base_channels: Base number of channels (default: 64).
        block_type: Block type, 'bottleneck' or 'basic'.
        strides: Stride for each stage (default: [1, 2, 2, 2]).
        dropout: Dropout probability.
        quantization: Enable quantization-friendly operations.

    Input:
        x: Tensor of shape (B, T, D).

    Output:
        x: Tensor of shape (B, T', D').
    """

    def __init__(
        self,
        input_dim: int,
        layers: List[int],
        base_channels: int = 64,
        block_type: str = "bottleneck",
        strides: Optional[List[int]] = None,
        dropout: float = 0.1,
        quantization: bool = False,
    ):
        super(ResNetBackbone1D, self).__init__()
        self.input_dim = input_dim
        self.base_channels = base_channels
        self.dropout = dropout
        self.quantization = quantization

        if block_type == "bottleneck":
            self.block = BottleneckBlock
        elif block_type == "basic":
            self.block = BasicBlock
        else:
            raise ValueError(f"Unknown block_type: {block_type}")

        if strides is None:
            strides = [1] + [2] * (len(layers) - 1)
        assert len(strides) == len(
            layers
        ), f"strides length ({len(strides)}) must match layers length ({len(layers)})"

        self.strides = strides

        self.conv1 = nn.Conv1d(
            input_dim, base_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.in_channels = base_channels
        self.layer1 = self._make_layer(base_channels, layers[0], stride=strides[0])
        self.layer2 = self._make_layer(base_channels * 2, layers[1], stride=strides[1])
        self.layer3 = self._make_layer(base_channels * 4, layers[2], stride=strides[2])
        self.layer4 = self._make_layer(base_channels * 8, layers[3], stride=strides[3])

        self.output_dim = base_channels * 8 * self.block.expansion

    def _make_layer(
        self, channels: int, num_blocks: int, stride: int = 1
    ) -> nn.Sequential:
        """Build a residual layer.

        Args:
            channels: Number of channels for this layer.
            num_blocks: Number of blocks.
            stride: Stride for the first block.

        Returns:
            Sequential containing residual blocks.
        """
        layers = []

        layers.append(
            self.block(
                self.in_channels,
                channels,
                stride=stride,
                dropout=self.dropout,
                quantization=self.quantization,
            )
        )
        self.in_channels = channels * self.block.expansion

        for _ in range(1, num_blocks):
            layers.append(
                self.block(
                    self.in_channels,
                    channels,
                    stride=1,
                    dropout=self.dropout,
                    quantization=self.quantization,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, T, D).

        Returns:
            Output tensor of shape (B, T', D').
        """
        x = x.transpose(1, 2)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.transpose(1, 2)

        return x

    def get_output_dim(self) -> int:
        """Return output feature dimension."""
        return self.output_dim


def resnet18_1d(input_dim: int, **kwargs) -> ResNetBackbone1D:
    """ResNet-18 1D with BasicBlock."""
    return ResNetBackbone1D(
        input_dim=input_dim, layers=[2, 2, 2, 2], block_type="basic", **kwargs
    )


def resnet34_1d(input_dim: int, **kwargs) -> ResNetBackbone1D:
    """ResNet-34 1D with BasicBlock."""
    return ResNetBackbone1D(
        input_dim=input_dim, layers=[3, 4, 6, 3], block_type="basic", **kwargs
    )


def resnet50_1d(input_dim: int, **kwargs) -> ResNetBackbone1D:
    """ResNet-50 1D with BottleneckBlock."""
    return ResNetBackbone1D(
        input_dim=input_dim, layers=[3, 4, 6, 3], block_type="bottleneck", **kwargs
    )


def resnet101_1d(input_dim: int, **kwargs) -> ResNetBackbone1D:
    """ResNet-101 1D with BottleneckBlock."""
    return ResNetBackbone1D(
        input_dim=input_dim, layers=[3, 4, 23, 3], block_type="bottleneck", **kwargs
    )


if __name__ == "__main__":
    print("Testing ResNet18 1D:")
    model = resnet18_1d(input_dim=12, base_channels=32, dropout=0.1)
    print(model)
    print(
        "model parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    x = torch.randn(4, 2048, 12)
    print(f"Input shape: {x.shape}")
    out = model(x)
    print(f"Output shape: {out.shape}")
    print(f"Output dim: {model.get_output_dim()}")

    print("\nTesting ResNet50 1D:")
    model = resnet50_1d(input_dim=12, base_channels=64, dropout=0.1)
    x = torch.randn(4, 2048, 12)
    print(f"Input shape: {x.shape}")
    out = model(x)
    print(f"Output shape: {out.shape}")
    print(f"Output dim: {model.get_output_dim()}")
