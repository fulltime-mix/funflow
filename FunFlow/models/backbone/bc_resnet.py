# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.

import torch
from torch import nn
import torch.nn.functional as F


class SubSpectralNorm(nn.Module):
    """Sub-spectral normalization layer."""

    def __init__(self, num_features, spec_groups=16, affine="Sub", batch=True, dim=2):
        super().__init__()
        self.spec_groups = spec_groups
        self.affine_all = False
        affine_norm = False
        if affine == "Sub":
            affine_norm = True
        elif affine == "All":
            self.affine_all = True
            self.weight = nn.Parameter(torch.ones((1, num_features, 1, 1)))
            self.bias = nn.Parameter(torch.zeros((1, num_features, 1, 1)))
        if batch:
            self.ssnorm = nn.BatchNorm2d(num_features * spec_groups, affine=affine_norm)
        else:
            self.ssnorm = nn.InstanceNorm2d(
                num_features * spec_groups, affine=affine_norm
            )
        self.sub_dim = dim

    def forward(self, x):
        if self.sub_dim in (3, -1):
            x = x.transpose(2, 3)
            x = x.contiguous()
        b, c, h, w = x.size()
        assert h % self.spec_groups == 0
        x = x.view(b, c * self.spec_groups, h // self.spec_groups, w)
        x = self.ssnorm(x)
        x = x.view(b, c, h, w)
        if self.affine_all:
            x = x * self.weight + self.bias
        if self.sub_dim in (3, -1):
            x = x.transpose(2, 3)
            x = x.contiguous()
        return x


class ConvBNReLU(nn.Module):
    """Convolution + BatchNorm + ReLU block."""

    def __init__(
        self,
        in_plane,
        out_plane,
        idx,
        kernel_size=3,
        stride=1,
        groups=1,
        use_dilation=False,
        activation=True,
        swish=False,
        BN=True,
        ssn=False,
    ):
        super().__init__()

        def get_padding(kernel_size, use_dilation):
            rate = 1
            padding_len = (kernel_size - 1) // 2
            if use_dilation and kernel_size > 1:
                rate = int(2**self.idx)
                padding_len = rate * padding_len
            return padding_len, rate

        self.idx = idx

        if isinstance(kernel_size, (list, tuple)):
            padding = []
            rate = []
            for k_size in kernel_size:
                temp_padding, temp_rate = get_padding(k_size, use_dilation)
                rate.append(temp_rate)
                padding.append(temp_padding)
        else:
            padding, rate = get_padding(kernel_size, use_dilation)

        layers = []
        layers.append(
            nn.Conv2d(
                in_plane,
                out_plane,
                kernel_size,
                stride,
                padding,
                rate,
                groups,
                bias=False,
            )
        )
        if ssn:
            layers.append(SubSpectralNorm(out_plane, 5))
        elif BN:
            layers.append(nn.BatchNorm2d(out_plane))
        if swish:
            layers.append(nn.SiLU(True))
        elif activation:
            layers.append(nn.ReLU(True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class BCResBlock(nn.Module):
    """BC-ResNet residual block."""

    def __init__(self, in_plane, out_plane, idx, stride):
        super().__init__()
        self.transition_block = in_plane != out_plane
        kernel_size = (3, 3)

        layers = []
        if self.transition_block:
            layers.append(ConvBNReLU(in_plane, out_plane, idx, 1, 1))
            in_plane = out_plane
        layers.append(
            ConvBNReLU(
                in_plane,
                out_plane,
                idx,
                (kernel_size[0], 1),
                (stride[0], 1),
                groups=in_plane,
                ssn=True,
                activation=False,
            )
        )
        self.f2 = nn.Sequential(*layers)
        self.avg_gpool = nn.AdaptiveAvgPool2d((1, None))

        self.f1 = nn.Sequential(
            ConvBNReLU(
                out_plane,
                out_plane,
                idx,
                (1, kernel_size[1]),
                (1, stride[1]),
                groups=out_plane,
                swish=True,
                use_dilation=True,
            ),
            nn.Conv2d(out_plane, out_plane, 1, bias=False),
            nn.Dropout2d(0.1),
        )

    def forward(self, x):
        shortcut = x
        x = self.f2(x)
        aux_2d_res = x
        x = self.avg_gpool(x)
        x = self.f1(x)
        x = x + aux_2d_res
        if not self.transition_block:
            x = x + shortcut
        x = F.relu(x, True)
        return x


def BCBlockStage(num_layers, last_channel, cur_channel, idx, use_stride):
    """Create a stage of BC-ResNet blocks."""
    stage = nn.ModuleList()
    channels = [last_channel] + [cur_channel] * num_layers
    for i in range(num_layers):
        stride = (2, 1) if use_stride and i == 0 else (1, 1)
        stage.append(BCResBlock(channels[i], channels[i + 1], idx, stride))
    return stage


class BCResNets(nn.Module):
    """BC-ResNet for audio classification."""

    def __init__(self, base_c, num_classes=12):
        """
        Args:
            base_c: Base number of channels
            num_classes: Number of output classes
        """
        super().__init__()
        self.num_classes = num_classes
        self.n = [2, 2, 4, 4]
        self.c = [
            base_c * 2,
            base_c,
            int(base_c * 1.5),
            base_c * 2,
            int(base_c * 2.5),
            base_c * 4,
        ]
        self.s = [1, 2]
        self._build_network()

    def _build_network(self):
        self.cnn_head = nn.Sequential(
            nn.Conv2d(1, self.c[0], 5, (2, 1), 2, bias=False),
            nn.BatchNorm2d(self.c[0]),
            nn.ReLU(True),
        )
        self.BCBlocks = nn.ModuleList([])
        for idx, n in enumerate(self.n):
            use_stride = idx in self.s
            self.BCBlocks.append(
                BCBlockStage(n, self.c[idx], self.c[idx + 1], idx, use_stride)
            )

        self.classifier = nn.Sequential(
            nn.Conv2d(
                self.c[-2],
                self.c[-2],
                (5, 5),
                bias=False,
                groups=self.c[-2],
                padding=(0, 2),
            ),
            nn.Conv2d(self.c[-2], self.c[-1], 1, bias=False),
            nn.BatchNorm2d(self.c[-1]),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.c[-1], self.num_classes, 1),
        )

    def forward(self, x):
        x = self.cnn_head(x)
        print(f"Shape after cnn_head: {x.shape}")
        for i, num_modules in enumerate(self.n):
            for j in range(num_modules):
                x = self.BCBlocks[i][j](x)
                print(f"Shape after BCBlock {i}-{j}: {x.shape}")
        x = self.classifier(x)
        print(f"Shape after classifier: {x.shape}")
        x = x.view(-1, x.shape[1])
        return x


if __name__ == "__main__":
    model = BCResNets(base_c=32, num_classes=2)
    print("=" * 80)
    print("Model Structure:")
    print("=" * 80)
    print(model)
    print("\n")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=" * 80)
    print("Model Parameters:")
    print("=" * 80)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (assuming float32)")
    print("\n")

    print("=" * 80)
    print("Forward Pass Test:")
    print("=" * 80)
    batch_size = 2
    height = 80
    width = 500

    dummy_input = torch.randn(batch_size, 1, height, width)
    print(f"Input shape: {dummy_input.shape}")

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Output shape: {output.shape}")
    print(f"Output:\n{output}")
    print("\n")

    print("=" * 80)
    print("Test completed successfully!")
    print("=" * 80)
