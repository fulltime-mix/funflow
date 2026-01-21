"""TinyViT

Paper: `TinyViT: Fast Pretraining Distillation for Small Vision Transformers`
    - https://arxiv.org/abs/2207.10666

Adapted from official impl at https://github.com/microsoft/Cream/tree/main/TinyViT
"""

__all__ = [
    "TinyVit",
    "TinyVitBackbone",
    "tiny_vit_5m_224",
    "tiny_vit_11m_224",
    "tiny_vit_21m_224",
    "tiny_vit_21m_384",
    "tiny_vit_21m_512",
    "create_tinyvit",
    "TINYVIT_MODELS",
    "MODEL_CONFIGS",
    "PRETRAINED_WEIGHTS",
]

import itertools
from functools import partial
from typing import Dict, List, Optional, Tuple, Union, Type, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import (
    DropPath,
    trunc_normal_,
    use_fused_attn,
    calculate_drop_path_rates,
)
from timm.models._features_fx import register_notrace_module

from FunFlow.registry import BACKBONES


class ConvNorm(torch.nn.Sequential):
    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        ks: int = 1,
        stride: int = 1,
        pad: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bn_weight_init: float = 1,
        device=None,
        dtype=None,
    ):
        dd = {"device": device, "dtype": dtype}
        super().__init__()
        self.conv = nn.Conv2d(
            in_chs, out_chs, ks, stride, pad, dilation, groups, bias=False, **dd
        )
        self.bn = nn.BatchNorm2d(out_chs, **dd)
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self.conv, self.bn
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(
            w.size(1) * self.conv.groups,
            w.size(0),
            w.shape[2:],
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class PatchEmbed(nn.Module):
    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        act_layer: Type[nn.Module],
        device=None,
        dtype=None,
    ):
        dd = {"device": device, "dtype": dtype}
        super().__init__()
        self.stride = 4
        self.conv1 = ConvNorm(in_chs, out_chs // 2, 3, 2, 1, **dd)
        self.act = act_layer()
        self.conv2 = ConvNorm(out_chs // 2, out_chs, 3, 2, 1, **dd)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x


class MBConv(nn.Module):
    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        expand_ratio: float,
        act_layer: Type[nn.Module],
        drop_path: float,
        device=None,
        dtype=None,
    ):
        dd = {"device": device, "dtype": dtype}
        super().__init__()
        mid_chs = int(in_chs * expand_ratio)
        self.conv1 = ConvNorm(in_chs, mid_chs, ks=1, **dd)
        self.act1 = act_layer()
        self.conv2 = ConvNorm(
            mid_chs, mid_chs, ks=3, stride=1, pad=1, groups=mid_chs, **dd
        )
        self.act2 = act_layer()
        self.conv3 = ConvNorm(mid_chs, out_chs, ks=1, bn_weight_init=0.0, **dd)
        self.act3 = act_layer()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.drop_path(x)
        x += shortcut
        x = self.act3(x)
        return x


class PatchMerging(nn.Module):
    def __init__(
        self,
        dim: int,
        out_dim: int,
        act_layer: Type[nn.Module],
        device=None,
        dtype=None,
    ):
        dd = {"device": device, "dtype": dtype}
        super().__init__()
        self.conv1 = ConvNorm(dim, out_dim, 1, 1, 0, **dd)
        self.act1 = act_layer()
        self.conv2 = ConvNorm(out_dim, out_dim, 3, 2, 1, groups=out_dim, **dd)
        self.act2 = act_layer()
        self.conv3 = ConvNorm(out_dim, out_dim, 1, 1, 0, **dd)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        return x


class ConvLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        act_layer: Type[nn.Module],
        drop_path: Union[float, List[float]] = 0.0,
        conv_expand_ratio: float = 4.0,
        device=None,
        dtype=None,
    ):
        dd = {"device": device, "dtype": dtype}
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.blocks = nn.Sequential(
            *[
                MBConv(
                    dim,
                    dim,
                    conv_expand_ratio,
                    act_layer,
                    drop_path[i] if isinstance(drop_path, list) else drop_path,
                    **dd,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class NormMlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        drop: float = 0.0,
        device=None,
        dtype=None,
    ):
        dd = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = norm_layer(in_features, **dd)
        self.fc1 = nn.Linear(in_features, hidden_features, **dd)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, **dd)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(torch.nn.Module):
    """Multi-head attention with relative position bias."""

    fused_attn: torch.jit.Final[bool]
    attention_bias_cache: Dict[str, torch.Tensor]

    def __init__(
        self,
        dim: int,
        key_dim: int,
        num_heads: int = 8,
        attn_ratio: int = 4,
        resolution: Tuple[int, int] = (14, 14),
        device=None,
        dtype=None,
    ):
        dd = {"device": device, "dtype": dtype}
        super().__init__()
        assert isinstance(resolution, tuple) and len(resolution) == 2
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.val_dim = int(attn_ratio * key_dim)
        self.out_dim = self.val_dim * num_heads
        self.attn_ratio = attn_ratio
        self.resolution = resolution
        self.fused_attn = use_fused_attn()

        self.norm = nn.LayerNorm(dim, **dd)
        self.qkv = nn.Linear(dim, num_heads * (self.val_dim + 2 * key_dim), **dd)
        self.proj = nn.Linear(self.out_dim, dim, **dd)

        points = list(itertools.product(range(resolution[0]), range(resolution[1])))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets), **dd)
        )
        self.register_buffer(
            "attention_bias_idxs",
            torch.tensor(idxs, device=device, dtype=torch.long).view(N, N),
            persistent=False,
        )
        self.attention_bias_cache = {}

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and self.attention_bias_cache:
            self.attention_bias_cache = {}

    def get_attention_biases(self, device: torch.device) -> torch.Tensor:
        if torch.jit.is_tracing() or self.training:
            return self.attention_biases[:, self.attention_bias_idxs]
        else:
            device_key = str(device)
            if device_key not in self.attention_bias_cache:
                self.attention_bias_cache[device_key] = self.attention_biases[
                    :, self.attention_bias_idxs
                ]
            return self.attention_bias_cache[device_key]

    def forward(self, x):
        attn_bias = self.get_attention_biases(x.device)
        B, N, _ = x.shape
        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, N, self.num_heads, -1).split(
            [self.key_dim, self.key_dim, self.val_dim], dim=3
        )
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn + attn_bias
            attn = attn.softmax(dim=-1)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, self.out_dim)
        x = self.proj(x)
        return x


class TinyVitBlock(nn.Module):
    """TinyViT Block with window attention and local convolution.

    Args:
        dim: Number of input channels.
        num_heads: Number of attention heads.
        window_size: Window size.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim.
        drop: Dropout rate.
        drop_path: Stochastic depth rate.
        local_conv_size: Kernel size of depthwise conv between Attention and MLP.
        act_layer: Activation function.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        local_conv_size: int = 3,
        act_layer: Type[nn.Module] = nn.GELU,
        device=None,
        dtype=None,
    ):
        dd = {"device": device, "dtype": dtype}
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert window_size > 0, "window_size must be greater than 0"
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        head_dim = dim // num_heads

        window_resolution = (window_size, window_size)
        self.attn = Attention(
            dim,
            head_dim,
            num_heads,
            attn_ratio=1,
            resolution=window_resolution,
            **dd,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.mlp = NormMlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
            **dd,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        pad = local_conv_size // 2
        self.local_conv = ConvNorm(
            dim, dim, ks=local_conv_size, stride=1, pad=pad, groups=dim, **dd
        )

    def forward(self, x):
        B, H, W, C = x.shape
        L = H * W

        shortcut = x
        if H == self.window_size and W == self.window_size:
            x = x.reshape(B, L, C)
            x = self.attn(x)
            x = x.view(B, H, W, C)
        else:
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            padding = pad_b > 0 or pad_r > 0
            if padding:
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_size
            nW = pW // self.window_size
            x = (
                x.view(B, nH, self.window_size, nW, self.window_size, C)
                .transpose(2, 3)
                .reshape(B * nH * nW, self.window_size * self.window_size, C)
            )

            x = self.attn(x)

            x = (
                x.view(B, nH, nW, self.window_size, self.window_size, C)
                .transpose(2, 3)
                .reshape(B, pH, pW, C)
            )

            if padding:
                x = x[:, :H, :W].contiguous()
        x = shortcut + self.drop_path1(x)

        x = x.permute(0, 3, 1, 2)
        x = self.local_conv(x)
        x = x.reshape(B, C, L).transpose(1, 2)

        x = x + self.drop_path2(self.mlp(x))
        return x.view(B, H, W, C)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, mlp_ratio={self.mlp_ratio}"
        )


register_notrace_module(TinyVitBlock)


class TinyVitStage(nn.Module):
    """TinyViT layer for one stage.

    Args:
        dim: Number of input channels.
        out_dim: Output dimension.
        depth: Number of blocks.
        num_heads: Number of attention heads.
        window_size: Window size.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim.
        drop: Dropout rate.
        drop_path: Stochastic depth rate.
        downsample: Downsample layer at the end of stage.
        local_conv_size: Kernel size of depthwise conv.
        act_layer: Activation function.
    """

    def __init__(
        self,
        dim: int,
        out_dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: Union[float, List[float]] = 0.0,
        downsample: Optional[Type[nn.Module]] = None,
        local_conv_size: int = 3,
        act_layer: Type[nn.Module] = nn.GELU,
        device=None,
        dtype=None,
    ):
        dd = {"device": device, "dtype": dtype}
        super().__init__()
        self.depth = depth
        self.out_dim = out_dim

        if downsample is not None:
            self.downsample = downsample(
                dim=dim,
                out_dim=out_dim,
                act_layer=act_layer,
                **dd,
            )
        else:
            self.downsample = nn.Identity()
            assert dim == out_dim

        self.blocks = nn.Sequential(
            *[
                TinyVitBlock(
                    dim=out_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    drop_path=(
                        drop_path[i] if isinstance(drop_path, list) else drop_path
                    ),
                    local_conv_size=local_conv_size,
                    act_layer=act_layer,
                    **dd,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x):
        x = self.downsample(x)
        x = x.permute(0, 2, 3, 1)
        x = self.blocks(x)
        x = x.permute(0, 3, 1, 2)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.out_dim}, depth={self.depth}"


class TinyVit(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        embed_dims: Tuple[int, ...] = (96, 192, 384, 768),
        depths: Tuple[int, ...] = (2, 2, 6, 2),
        num_heads: Tuple[int, ...] = (3, 6, 12, 24),
        window_sizes: Tuple[int, ...] = (7, 7, 14, 7),
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        mbconv_expand_ratio: float = 4.0,
        local_conv_size: int = 3,
        act_layer: Type[nn.Module] = nn.GELU,
        timm_model_name: Optional[str] = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        dd = {"device": device, "dtype": dtype}

        self.depths = depths
        self.num_stages = len(depths)
        self.mlp_ratio = mlp_ratio
        self.timm_model_name = timm_model_name

        self.patch_embed = PatchEmbed(
            in_chs=in_chans,
            out_chs=embed_dims[0],
            act_layer=act_layer,
            **dd,
        )

        dpr = calculate_drop_path_rates(drop_path_rate, sum(depths))

        self.stages = nn.Sequential()
        stride = self.patch_embed.stride
        prev_dim = embed_dims[0]
        self.feature_info = []
        for stage_idx in range(self.num_stages):
            if stage_idx == 0:
                stage = ConvLayer(
                    dim=prev_dim,
                    depth=depths[stage_idx],
                    act_layer=act_layer,
                    drop_path=dpr[: depths[stage_idx]],
                    conv_expand_ratio=mbconv_expand_ratio,
                    **dd,
                )
            else:
                out_dim = embed_dims[stage_idx]
                drop_path_rate = dpr[
                    sum(depths[:stage_idx]) : sum(depths[: stage_idx + 1])
                ]
                stage = TinyVitStage(
                    dim=embed_dims[stage_idx - 1],
                    out_dim=out_dim,
                    depth=depths[stage_idx],
                    num_heads=num_heads[stage_idx],
                    window_size=window_sizes[stage_idx],
                    mlp_ratio=self.mlp_ratio,
                    drop=drop_rate,
                    local_conv_size=local_conv_size,
                    drop_path=drop_path_rate,
                    downsample=PatchMerging,
                    act_layer=act_layer,
                    **dd,
                )
                prev_dim = out_dim
                stride *= 2
            self.stages.append(stage)
            self.feature_info += [
                dict(num_chs=prev_dim, reduction=stride, module=f"stages.{stage_idx}")
            ]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.stages(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return x

    def load_pretrained(self, pretrained_path: Optional[str] = None):
        """Load pretrained weights from timm hub or local file."""
        import timm
        import os

        if pretrained_path:
            pretrained_path = os.path.expanduser(pretrained_path)
            if pretrained_path.endswith(".safetensors"):
                from safetensors.torch import load_file

                state_dict = load_file(pretrained_path)
            else:
                state_dict = torch.load(pretrained_path, map_location="cpu")
                state_dict = state_dict.get("state_dict") or state_dict.get(
                    "model", state_dict
                )
        else:
            if not self.timm_model_name:
                raise ValueError(
                    "timm_model_name required. Use factory functions like tiny_vit_5m_224()."
                )
            timm_model = timm.create_model(self.timm_model_name, pretrained=True)
            state_dict = timm_model.state_dict()

        self.load_state_dict(state_dict, strict=False)

        pretrained_keys = set(state_dict.keys())
        model_keys = set(self.state_dict().keys())
        matched = len(pretrained_keys & model_keys)
        print(f"[TinyVit] Loaded: {matched}/{len(model_keys)} keys matched")
        return self


MODEL_CONFIGS = {
    "tiny_vit_5m_224": {
        "embed_dims": [64, 128, 160, 320],
        "depths": [2, 2, 6, 2],
        "num_heads": [2, 4, 5, 10],
        "window_sizes": [7, 7, 14, 7],
        "drop_path_rate": 0.0,
    },
    "tiny_vit_11m_224": {
        "embed_dims": [64, 128, 256, 448],
        "depths": [2, 2, 6, 2],
        "num_heads": [2, 4, 8, 14],
        "window_sizes": [7, 7, 14, 7],
        "drop_path_rate": 0.1,
    },
    "tiny_vit_21m_224": {
        "embed_dims": [96, 192, 384, 576],
        "depths": [2, 2, 6, 2],
        "num_heads": [3, 6, 12, 18],
        "window_sizes": [7, 7, 14, 7],
        "drop_path_rate": 0.2,
    },
    "tiny_vit_21m_384": {
        "embed_dims": [96, 192, 384, 576],
        "depths": [2, 2, 6, 2],
        "num_heads": [3, 6, 12, 18],
        "window_sizes": [12, 12, 24, 12],
        "drop_path_rate": 0.1,
    },
    "tiny_vit_21m_512": {
        "embed_dims": [96, 192, 384, 576],
        "depths": [2, 2, 6, 2],
        "num_heads": [3, 6, 12, 18],
        "window_sizes": [16, 16, 32, 16],
        "drop_path_rate": 0.1,
    },
}

PRETRAINED_WEIGHTS = {
    "tiny_vit_5m_224.dist_in22k": "tiny_vit_5m_224",
    "tiny_vit_5m_224.dist_in22k_ft_in1k": "tiny_vit_5m_224",
    "tiny_vit_5m_224.in1k": "tiny_vit_5m_224",
    "tiny_vit_11m_224.dist_in22k": "tiny_vit_11m_224",
    "tiny_vit_11m_224.dist_in22k_ft_in1k": "tiny_vit_11m_224",
    "tiny_vit_11m_224.in1k": "tiny_vit_11m_224",
    "tiny_vit_21m_224.dist_in22k": "tiny_vit_21m_224",
    "tiny_vit_21m_224.dist_in22k_ft_in1k": "tiny_vit_21m_224",
    "tiny_vit_21m_224.in1k": "tiny_vit_21m_224",
    "tiny_vit_21m_384.dist_in22k_ft_in1k": "tiny_vit_21m_384",
    "tiny_vit_21m_512.dist_in22k_ft_in1k": "tiny_vit_21m_512",
}


def _create_tinyvit(
    variant: str,
    pretrained: bool = False,
    pretrained_path: Optional[str] = None,
    **kwargs,
):
    """Internal factory to create TinyViT from variant name."""
    if variant in PRETRAINED_WEIGHTS:
        base_model = PRETRAINED_WEIGHTS[variant]
        timm_model_name = variant
    elif variant in MODEL_CONFIGS:
        base_model = variant
        timm_model_name = None
    else:
        raise ValueError(f"Unknown variant: {variant}")

    model_config = MODEL_CONFIGS[base_model].copy()
    model_config.update(kwargs)

    if "timm_model_name" not in model_config and timm_model_name:
        model_config["timm_model_name"] = timm_model_name

    model = TinyVit(**model_config)
    if pretrained:
        model.load_pretrained(pretrained_path)

    return model


def tiny_vit_5m_224(
    pretrained: bool = False,
    pretrained_path: Optional[str] = None,
    variant: str = "dist_in22k_ft_in1k",
    **kwargs,
):
    """TinyViT-5M 224x224. Variants: 'dist_in22k', 'dist_in22k_ft_in1k', 'in1k'."""
    return _create_tinyvit(
        f"tiny_vit_5m_224.{variant}", pretrained, pretrained_path, **kwargs
    )


def tiny_vit_11m_224(
    pretrained: bool = False,
    pretrained_path: Optional[str] = None,
    variant: str = "dist_in22k_ft_in1k",
    **kwargs,
):
    """TinyViT-11M 224x224. Variants: 'dist_in22k', 'dist_in22k_ft_in1k', 'in1k'."""
    return _create_tinyvit(
        f"tiny_vit_11m_224.{variant}", pretrained, pretrained_path, **kwargs
    )


def tiny_vit_21m_224(
    pretrained: bool = False,
    pretrained_path: Optional[str] = None,
    variant: str = "dist_in22k_ft_in1k",
    **kwargs,
):
    """TinyViT-21M 224x224. Variants: 'dist_in22k', 'dist_in22k_ft_in1k', 'in1k'."""
    return _create_tinyvit(
        f"tiny_vit_21m_224.{variant}", pretrained, pretrained_path, **kwargs
    )


def tiny_vit_21m_384(
    pretrained: bool = False,
    pretrained_path: Optional[str] = None,
    variant: str = "dist_in22k_ft_in1k",
    **kwargs,
):
    """TinyViT-21M 384x384. Variant: 'dist_in22k_ft_in1k'."""
    return _create_tinyvit(
        f"tiny_vit_21m_384.{variant}", pretrained, pretrained_path, **kwargs
    )


def tiny_vit_21m_512(
    pretrained: bool = False,
    pretrained_path: Optional[str] = None,
    variant: str = "dist_in22k_ft_in1k",
    **kwargs,
):
    """TinyViT-21M 512x512. Variant: 'dist_in22k_ft_in1k'."""
    return _create_tinyvit(
        f"tiny_vit_21m_512.{variant}", pretrained, pretrained_path, **kwargs
    )


def create_tinyvit(
    model_name: str,
    pretrained: bool = False,
    pretrained_path: Optional[str] = None,
    **kwargs,
):
    """Universal factory to create TinyViT by full name (e.g., 'tiny_vit_5m_224.dist_in22k_ft_in1k')."""
    return _create_tinyvit(model_name, pretrained, pretrained_path, **kwargs)


TINYVIT_MODELS = {
    "tiny_vit_5m_224": tiny_vit_5m_224,
    "tiny_vit_11m_224": tiny_vit_11m_224,
    "tiny_vit_21m_224": tiny_vit_21m_224,
    "tiny_vit_21m_384": tiny_vit_21m_384,
    "tiny_vit_21m_512": tiny_vit_21m_512,
    "create_tinyvit": create_tinyvit,
}


@BACKBONES.register("TinyVit")
@BACKBONES.register("TinyVitBackbone")
class TinyVitBackbone(TinyVit):
    """TinyViT backbone wrapper for registry."""

    def __init__(
        self,
        model_name: str = "tiny_vit_5m_224",
        pretrained: bool = True,
        pretrained_path: Optional[str] = None,
        variant: str = "dist_in22k_ft_in1k",
        **kwargs,
    ):
        full_name = f"{model_name}.{variant}" if "." not in model_name else model_name

        if full_name in PRETRAINED_WEIGHTS:
            base_model = PRETRAINED_WEIGHTS[full_name]
            timm_model_name = full_name
        elif model_name in MODEL_CONFIGS:
            base_model = model_name
            timm_model_name = None
        else:
            raise ValueError(
                f"Unknown model name: {model_name}. "
                f"Available models: {list(MODEL_CONFIGS.keys())}"
            )

        config = MODEL_CONFIGS[base_model].copy()
        config.update(kwargs)

        if "timm_model_name" not in config and timm_model_name:
            config["timm_model_name"] = timm_model_name

        super().__init__(**config)

        if pretrained:
            self.load_pretrained(pretrained_path)


if __name__ == "__main__":
    print("Test 1: Creating TinyVit model directly...")
    model1 = TinyVit(
        in_chans=3,
        embed_dims=[64, 128, 160, 320],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
    )
    dummy_input = torch.randn(1, 3, 224, 224)
    output1 = model1(dummy_input)
    print(f"Output shape: {output1.shape}")

    print("\nTest 2: Using factory function with pretrained weights...")
    model2 = tiny_vit_5m_224(pretrained=True)
    output2 = model2(dummy_input)
    print(f"Output shape: {output2.shape}")

    print("\nAll tests passed!")
