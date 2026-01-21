"""Models module providing unified interface and components.

Architecture: Preprocessing -> Backbone -> Neck -> Head -> Loss
"""

from FunFlow.models.base import BaseModel

from FunFlow.models.backbone import (
    ResNet,
    ResNetBackbone,
    TinyVitBackbone,
    TCN,
    TCNBackbone,
    ResNet1D,
    ResNetBackbone1D,
)

from FunFlow.models.head import (
    LinearHead,
    MLPHead,
    PoolingHead,
)

from FunFlow.models.neck import (
    Identity as NeckIdentity,
    GlobalAveragePooling,
    Flatten,
)

from FunFlow.models.preprocessing import (
    GlobalCMVN,
    Normalize,
)

from FunFlow.models.loss import (
    CrossEntropyLoss,
    FocalLoss,
    LabelSmoothingLoss,
)

try:
    from FunFlow.models.backbone.tinyvit import TinyVit

    _HAS_TINYVIT = True
except ImportError:
    _HAS_TINYVIT = False

try:
    from FunFlow.models.head.norm_mlp import NormMlpClassifierHead

    _HAS_NORM_MLP = True
except ImportError:
    _HAS_NORM_MLP = False


__all__ = [
    "BaseModel",
    "ResNet",
    "ResNetBackbone",
    "TinyVitBackbone",
    "TCN",
    "TCNBackbone",
    "ResNet1D",
    "ResNetBackbone1D",
    "LinearHead",
    "MLPHead",
    "PoolingHead",
    "NeckIdentity",
    "GlobalAveragePooling",
    "Flatten",
    "PreprocessingIdentity",
    "GlobalCMVN",
    "Normalize",
    "CrossEntropyLoss",
    "FocalLoss",
    "LabelSmoothingLoss",
]

if _HAS_TINYVIT:
    __all__.append("TinyVit")

if _HAS_NORM_MLP:
    __all__.append("NormMlpClassifierHead")
