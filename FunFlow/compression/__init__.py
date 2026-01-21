"""Model compression module providing pruning, distillation, and quantization."""

from FunFlow.registry import FUSION_STRATEGIES

from .utils import (
    get_model_sparsity,
    get_layer_sparsity,
    get_model_size,
    compare_model_sizes,
    count_zero_parameters,
    visualize_sparsity,
)

from .QAT import (
    QATQuantizer,
    FusionStrategy,
    ResNetFusionStrategy,
    DefaultFusionStrategy,
    get_fusion_strategy,
)

__all__ = [
    # QAT (Quantization-Aware Training)
    "QATQuantizer",
    "FusionStrategy",
    "ResNetFusionStrategy",
    "AudioCNNFusionStrategy",
    "DefaultFusionStrategy",
    "get_fusion_strategy",
    "FUSION_STRATEGIES",
    # Utils
    "get_model_sparsity",
    "get_layer_sparsity",
    "get_model_size",
    "compare_model_sizes",
    "count_zero_parameters",
    "visualize_sparsity",
]
