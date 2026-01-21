"""Head module for classification heads."""

from FunFlow.models.head.linear import LinearHead
from FunFlow.models.head.mlp import MLPHead
from FunFlow.models.head.pooling_head import PoolingHead

try:
    from FunFlow.models.head.norm_mlp import NormMlpClassifierHead

    _HAS_NORM_MLP = True
except ImportError:
    _HAS_NORM_MLP = False

MLP = MLPHead

__all__ = [
    "LinearHead",
    "MLPHead",
    "MLP",
]

if _HAS_NORM_MLP:
    __all__.append("NormMlpClassifierHead")
