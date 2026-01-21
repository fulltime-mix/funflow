"""NormMlpClassifierHead: MLP head with normalization for ViT models.

Requires timm library.
"""

import torch
import torch.nn as nn
from typing import Optional
from collections import OrderedDict
from functools import partial

from FunFlow.registry import HEADS

try:
    from timm.layers import LayerNorm2d
    from timm.layers.adaptive_avgmax_pool import SelectAdaptivePool2d

    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


if HAS_TIMM:

    @HEADS.register("NormMlpClassifierHead")
    class NormMlpClassifierHead(nn.Module):
        """Pool -> Norm -> MLP classification head for Transformer models."""

        def __init__(
            self,
            in_features: int,
            num_classes: int,
            hidden_size: Optional[int] = None,
            drop_rate: float = 0.0,
            device=None,
            dtype=None,
        ):
            """
            Args:
                in_features: Input feature dimension
                num_classes: Number of classes
                hidden_size: MLP hidden layer size, None to skip
                drop_rate: Dropout ratio
            """
            dd = {"device": device, "dtype": dtype}
            super().__init__()
            self.in_features = in_features
            self.hidden_size = hidden_size
            self.num_features = in_features
            norm_layer = partial(LayerNorm2d, eps=1e-5)

            self.global_pool = SelectAdaptivePool2d(pool_type="avg")
            self.norm = norm_layer(in_features, **dd)
            self.flatten = nn.Flatten(1)

            if hidden_size:
                self.pre_logits = nn.Sequential(
                    OrderedDict(
                        [
                            ("fc", nn.Linear(in_features, hidden_size, **dd)),
                            ("act", nn.GELU()),
                        ]
                    )
                )
                self.num_features = hidden_size
            else:
                self.pre_logits = nn.Identity()

            self.drop = nn.Dropout(drop_rate)
            self.fc = (
                nn.Linear(self.num_features, num_classes, **dd)
                if num_classes > 0
                else nn.Identity()
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.global_pool(x)
            x = self.norm(x)
            x = self.flatten(x)
            x = self.pre_logits(x)
            x = self.drop(x)
            x = self.fc(x)
            return x


__all__ = ["NormMlpClassifierHead"] if HAS_TIMM else []
