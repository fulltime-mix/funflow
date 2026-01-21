"""Quantization-friendly ResNet implementation supporting QAT."""

import torch
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional

try:
    from torch.ao.nn.quantized import FloatFunctional
except ImportError:
    from torch.nn.quantized import FloatFunctional

from FunFlow.registry import BACKBONES

__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "BasicBlock",
    "Bottleneck",
    "ResNetBackbone",
]


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """ResNet BasicBlock for ResNet18/34 with quantization support."""

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        quantization: bool = False,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.quantization = quantization
        if quantization:
            self.skip_add = FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.quantization:
            out = self.skip_add.add(out, identity)
        else:
            out = out + identity

        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """ResNet Bottleneck for ResNet50/101/152 with quantization support."""

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        quantization: bool = False,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.quantization = quantization
        if quantization:
            self.skip_add = FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.quantization:
            out = self.skip_add.add(out, identity)
        else:
            out = out + identity

        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """Quantization-friendly ResNet implementation.

    Args:
        block: BasicBlock or Bottleneck.
        layers: Number of blocks in each stage.
        num_classes: Number of output classes, 0 for no classification head.
        zero_init_residual: Whether to zero-initialize the last BN in residual branches.
        groups: Number of groups for convolutions.
        width_per_group: Width per group.
        replace_stride_with_dilation: Whether to replace strides with dilation.
        norm_layer: Normalization layer.
        quantization: Whether to enable quantization support.
    """

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        quantization: bool = False,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.quantization = quantization

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if num_classes > 0:
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        else:
            self.fc = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                self.quantization,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    quantization=self.quantization,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if self.fc is not None:
            x = self.fc(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)


@BACKBONES.register("ResNet")
@BACKBONES.register("ResNetBackbone")
class ResNetBackbone(ResNet):
    """ResNet backbone wrapper for registry.

    Provides unified backbone interface with support for:
    - Multiple ResNet depths (18, 34, 50, 101, 152)
    - ImageNet pretrained weights
    - Quantization-aware training (QAT)
    - Custom input channels
    - Flexible output (backbone or classification mode)

    Args:
        depth: ResNet depth, supports 18, 34, 50, 101, 152.
        pretrained: Whether to load ImageNet-1K pretrained weights.
        in_channels: Input image channels, default 3 (RGB).
        num_classes: Output classes, 0 for no classification head (backbone only).
        quantization: Whether to enable quantization support for QAT.
        **kwargs: Additional arguments passed to ResNet.
    """

    def __init__(
        self,
        depth: int = 18,
        pretrained: bool = False,
        in_channels: int = 3,
        num_classes: int = 0,
        quantization: bool = False,
        **kwargs,
    ):
        if depth == 18:
            block, layers = BasicBlock, [2, 2, 2, 2]
        elif depth == 34:
            block, layers = BasicBlock, [3, 4, 6, 3]
        elif depth == 50:
            block, layers = Bottleneck, [3, 4, 6, 3]
        elif depth == 101:
            block, layers = Bottleneck, [3, 4, 23, 3]
        elif depth == 152:
            block, layers = Bottleneck, [3, 8, 36, 3]
        else:
            raise ValueError(
                f"Unsupported ResNet depth: {depth}. "
                f"Supported depths: 18, 34, 50, 101, 152"
            )

        super().__init__(
            block=block,
            layers=layers,
            num_classes=num_classes,
            quantization=quantization,
            **kwargs,
        )

        self.depth = depth
        self.block_expansion = block.expansion
        self.out_channels = 512 * block.expansion

        if pretrained:
            self._load_pretrained_weights(depth)

        if in_channels != 3:
            self.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            nn.init.kaiming_normal_(
                self.conv1.weight, mode="fan_out", nonlinearity="relu"
            )
            print(
                f"⚠ Input channels: {in_channels}, conv1 pretrained weights not loaded"
            )

    def _load_pretrained_weights(self, depth):
        """Load ImageNet-1K pretrained weights.

        Args:
            depth: ResNet depth.
        """
        try:
            import torchvision.models as models

            if depth == 18:
                pretrained_model = models.resnet18(weights="IMAGENET1K_V1")
            elif depth == 34:
                pretrained_model = models.resnet34(weights="IMAGENET1K_V1")
            elif depth == 50:
                pretrained_model = models.resnet50(weights="IMAGENET1K_V1")
            elif depth == 101:
                pretrained_model = models.resnet101(weights="IMAGENET1K_V1")
            elif depth == 152:
                pretrained_model = models.resnet152(weights="IMAGENET1K_V1")
            else:
                raise ValueError(f"Unknown depth: {depth}")

            pretrained_dict = pretrained_model.state_dict()
            model_dict = self.state_dict()

            matched_keys = []
            mismatched_keys = []
            missing_keys = []

            for k, v in pretrained_dict.items():
                if k in model_dict:
                    if model_dict[k].shape == v.shape:
                        model_dict[k] = v
                        matched_keys.append(k)
                    else:
                        mismatched_keys.append(
                            f"{k} (shape: {v.shape} vs {model_dict[k].shape})"
                        )

            for k in model_dict.keys():
                if k not in pretrained_dict:
                    missing_keys.append(k)

            self.load_state_dict(model_dict, strict=True)

            print(
                f"✓ Loaded {len(matched_keys)}/{len(pretrained_dict)} pretrained parameters"
            )

            if mismatched_keys:
                print(f"⚠ Shape mismatched keys ({len(mismatched_keys)}):")
                for key in mismatched_keys:
                    print(f"  - {key}")

            if missing_keys:
                print(f"⚠ Missing keys in pretrained weights ({len(missing_keys)}):")
                for key in missing_keys:
                    print(f"  - {key}")

        except Exception as e:
            print(f"⚠ Failed to load pretrained weights: {e}")
            print(f"  → Using random initialization")


if __name__ == "__main__":
    model = ResNetBackbone(
        depth=18, pretrained=True, in_channels=3, num_classes=0, quantization=False
    )
    print(model)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(y.shape)
