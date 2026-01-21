"""Quantization-Aware Training (QAT) module."""

from abc import ABC, abstractmethod
from typing import List, Optional, Type, Dict, Any

import torch
import torch.nn as nn

from FunFlow.logger import get_logger
from FunFlow.registry import FUSION_STRATEGIES

logger = get_logger("FunFlow")


class QATQuantizer:
    """Quantization-Aware Training (QAT) utility class.

    Workflow:
    1. prepare_qat: Prepare QAT model (fuse modules, insert fake quantization nodes)
    2. Train: Use standard training loop for QAT training
    3. convert_qat: Convert trained QAT model to quantized model
    """

    SUPPORTED_BACKENDS = ["fbgemm", "qnnpack", "x86", "onednn"]

    @classmethod
    def prepare_qat(
        cls,
        model: nn.Module,
        backend: str = "fbgemm",
        fuse_modules: bool = True,
        fusion_strategy: Optional[str] = None,
        inplace: bool = False,
    ) -> nn.Module:
        """Prepare QAT model.

        Args:
            model: Original model (needs quantization=True).
            backend: Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM).
            fuse_modules: Whether to fuse modules.
            fusion_strategy: Fusion strategy name.
            inplace: Whether to modify model in-place.

        Returns:
            Prepared QAT model.
        """
        if not inplace:
            import copy

            model = copy.deepcopy(model)

        if backend not in cls.SUPPORTED_BACKENDS:
            logger.warning(
                f"Unknown backend '{backend}', supported: {cls.SUPPORTED_BACKENDS}. "
                f"Using 'fbgemm' as default."
            )
            backend = "fbgemm"

        torch.backends.quantized.engine = backend

        model.train()

        if fuse_modules:
            model = cls._fuse_modules(model, fusion_strategy)

        qconfig = cls._get_qat_config(backend)
        model.qconfig = qconfig

        try:
            from torch.ao.quantization import prepare_qat

            model = prepare_qat(model, inplace=True)
        except ImportError:
            from torch.quantization import prepare_qat

            model = prepare_qat(model, inplace=True)

        logger.info(
            f"[QAT] Model prepared with backend='{backend}', fuse_modules={fuse_modules}"
        )

        return model

    @classmethod
    def convert_qat(cls, model: nn.Module, inplace: bool = False) -> nn.Module:
        """Convert trained QAT model to quantized model.

        Args:
            model: Trained QAT model.
            inplace: Whether to convert in-place.

        Returns:
            Quantized model.
        """
        if not inplace:
            import copy

            model = copy.deepcopy(model)

        model.eval()

        try:
            from torch.ao.quantization import convert

            quantized_model = convert(model, inplace=True)
        except ImportError:
            from torch.quantization import convert

            quantized_model = convert(model, inplace=True)

        logger.info("[QAT] Model converted to quantized format")

        return quantized_model

    @classmethod
    def _fuse_modules(
        cls, model: nn.Module, fusion_strategy: Optional[str] = None
    ) -> nn.Module:
        """Fuse modules in model.

        Args:
            model: Model.
            fusion_strategy: Fusion strategy name.

        Returns:
            Fused model.
        """
        strategy = get_fusion_strategy(model, fusion_strategy)
        strategy_name = strategy.__class__.__name__

        fusion_modules = strategy.get_fusion_modules(model)

        if not fusion_modules:
            logger.warning(
                f"[QAT] No fusion modules found with strategy '{strategy_name}'. "
                f"Skipping fusion."
            )
            return model

        logger.info(f"[QAT] Using fusion strategy: {strategy_name}")
        logger.info(f"[QAT] Fusing {len(fusion_modules)} module groups")

        try:
            from torch.ao.quantization import fuse_modules
        except ImportError:
            from torch.quantization import fuse_modules

        model.eval()

        try:
            fused_model = fuse_modules(model, fusion_modules, inplace=True)
        except Exception as e:
            logger.warning(
                f"[QAT] Module fusion failed: {e}. Continuing without fusion."
            )
            fused_model = model

        fused_model.train()

        return fused_model

    @classmethod
    def _get_qat_config(cls, backend: str):
        """Get QAT configuration.

        Args:
            backend: Quantization backend.

        Returns:
            QConfig object.
        """
        try:
            from torch.ao.quantization import get_default_qat_qconfig

            return get_default_qat_qconfig(backend)
        except ImportError:
            from torch.quantization import get_default_qat_qconfig

            return get_default_qat_qconfig(backend)

    @classmethod
    def convert_save_model(
        cls,
        checkpoint_path: str,
        config: dict | str,
        output_path: str,
        backend: str = "fbgemm",
        save_full_model: bool = False,
    ) -> nn.Module:
        """Convert checkpoint to quantized model and save.

        Args:
            checkpoint_path: QAT checkpoint path.
            config: Config dict or YAML path.
            output_path: Output path for quantized model.
            backend: Quantization backend.
            save_full_model: Save full model instead of state_dict.

        Returns:
            Quantized model.
        """
        from pathlib import Path
        from FunFlow.registry import MODELS

        if isinstance(config, str):
            import yaml

            with open(config, "r") as f:
                config = yaml.safe_load(f)

        print(f"[QAT] Creating model with quantization=True")
        model_cfg = config["model"].copy()
        model_cfg["quantization"] = True
        model = MODELS.build(model_cfg)

        print(f"[QAT] Preparing QAT model (backend={backend})")
        model = cls.prepare_qat(
            model,
            backend=backend,
            fuse_modules=True,
            fusion_strategy=config.get("quantization", {}).get("fusion_strategy", None),
            inplace=True,
        )

        print(f"[QAT] Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = (
            checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
            if isinstance(checkpoint, dict)
            else checkpoint
        )
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        if missing:
            print(f"⚠️Missing keys: {missing}")
        if unexpected:
            print(f"⚠️Unexpected keys: {unexpected}")

        print(f"[QAT] Converting to quantized model")
        model.eval()

        # 获取原始模型大小
        import tempfile

        with tempfile.NamedTemporaryFile(delete=True) as tmp:
            torch.save(model.state_dict(), tmp.name)
            original_size = Path(tmp.name).stat().st_size / (1024 * 1024)  # MB

        quantized_model = cls.convert_qat(model, inplace=False)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if save_full_model:
            torch.save(quantized_model, output_path)
            print(f"[QAT] Saved full quantized model to {output_path}")
        else:
            torch.save(quantized_model.state_dict(), output_path)
            print(f"[QAT] Saved quantized state_dict to {output_path}")

        # 模型大小对比
        quantized_size = output_path.stat().st_size / (1024 * 1024)  # MB
        compression_ratio = original_size / quantized_size if quantized_size > 0 else 0
        print(
            f"[QAT] Model size: {original_size:.2f}MB → {quantized_size:.2f}MB ({compression_ratio:.2f}x compression)"
        )

        return quantized_model

    @classmethod
    def load_converted_model(cls, model_path: str, config: dict | str = None):
        """Load converted quantized model from checkpoint.

        Args:
            model_path: Path to quantized model checkpoint.
            config: Config dict or YAML path (required if loading state_dict).

        Returns:
            Quantized model.
        """
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        if isinstance(checkpoint, nn.Module):
            print(f"[QAT] Loaded full quantized model from {model_path}")
            return checkpoint
        else:
            if config is None:
                raise ValueError("config is required when loading state_dict")

            if isinstance(config, str):
                import yaml

                with open(config, "r") as f:
                    config = yaml.safe_load(f)

            from FunFlow.registry import MODELS

            model_cfg = config["model"].copy()
            model_cfg["quantization"] = True
            model = MODELS.build(model_cfg)

            model = cls.prepare_qat(
                model,
                backend=config.get("quantization", {}).get("backend", "fbgemm"),
                fuse_modules=True,
                fusion_strategy=config.get("quantization", {}).get(
                    "fusion_strategy", None
                ),
                inplace=True,
            )
            model.eval()

            model = cls.convert_qat(model, inplace=True)

            if isinstance(checkpoint, dict) and (
                "state_dict" in checkpoint or "model_state_dict" in checkpoint
            ):
                state_dict = checkpoint.get(
                    "state_dict", checkpoint.get("model_state_dict")
                )
            else:
                state_dict = checkpoint

            missing, unexpected = model.load_state_dict(state_dict, strict=False)

            if missing:
                print(f"⚠️Missing keys: {missing}")
            if unexpected:
                print(f"⚠️Unexpected keys: {unexpected}")
            print(f"[QAT] Loaded quantized state_dict from {model_path}")
            return model


class FusionStrategy(ABC):
    """Module fusion strategy base class.

    Different architectures need different fusion patterns.
    Subclasses must implement get_fusion_modules method.
    """

    @abstractmethod
    def get_fusion_modules(self, model: nn.Module) -> List[List[str]]:
        """Get list of modules to fuse.

        Args:
            model: Model to fuse.

        Returns:
            List of module name lists to fuse.
        """
        pass

    @staticmethod
    def _module_exists(model: nn.Module, name: str) -> bool:
        """Check if module exists."""
        try:
            parts = name.split(".")
            module = model
            for part in parts:
                if module is None:
                    return False
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)
            return module is not None
        except (AttributeError, IndexError, KeyError, TypeError):
            return False

    def validate_fusion_modules(
        self, model: nn.Module, fusion_modules: List[List[str]]
    ) -> List[List[str]]:
        """Validate and filter valid fusion module lists.

        Args:
            model: Model.
            fusion_modules: Original fusion module list.

        Returns:
            Filtered valid fusion module list.
        """
        valid_modules = []
        for module_list in fusion_modules:
            if all(self._module_exists(model, name) for name in module_list):
                valid_modules.append(module_list)
        return valid_modules


class ResNetFusionStrategy(FusionStrategy):
    """ResNet fusion strategy supporting ResNet18/34/50/101."""

    def __init__(self, depth: int = 18):
        """Args:
        depth: ResNet depth (18, 34, 50, 101).
        """
        self.depth = depth
        self.use_bottleneck = depth >= 50

        self.num_blocks = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
        }.get(depth, [2, 2, 2, 2])

    def get_fusion_modules(self, model: nn.Module) -> List[List[str]]:
        """Get ResNet fusion module list."""
        fusion_modules = []

        prefix = self._get_backbone_prefix(model)

        is_qat_resnet = self._is_qat_resnet(model, prefix)
        if is_qat_resnet:
            logger.info("[QAT] Detected QAT-friendly ResNet (with FloatFunctional)")

        first_layer = [f"{prefix}conv1", f"{prefix}bn1"]
        fusion_modules.append(first_layer)

        for layer_idx, num_blocks in enumerate(self.num_blocks, start=1):
            layer_name = f"{prefix}layers.{layer_idx - 1}"

            for block_idx in range(num_blocks):
                block_prefix = f"{layer_name}.{block_idx}"

                if self.use_bottleneck:
                    fusion_modules.extend(
                        [
                            [f"{block_prefix}.conv1", f"{block_prefix}.bn1"],
                            [f"{block_prefix}.conv2", f"{block_prefix}.bn2"],
                            [f"{block_prefix}.conv3", f"{block_prefix}.bn3"],
                        ]
                    )
                    # downsample 层
                    if self._module_exists(model, f"{block_prefix}.downsample"):
                        fusion_modules.append(
                            [
                                f"{block_prefix}.downsample.0",
                                f"{block_prefix}.downsample.1",
                            ]
                        )
                else:
                    # BasicBlock: conv1+bn1, conv2+bn2
                    fusion_modules.extend(
                        [
                            [f"{block_prefix}.conv1", f"{block_prefix}.bn1"],
                            [f"{block_prefix}.conv2", f"{block_prefix}.bn2"],
                        ]
                    )
                    # downsample 层
                    if self._module_exists(model, f"{block_prefix}.downsample"):
                        fusion_modules.append(
                            [
                                f"{block_prefix}.downsample.0",
                                f"{block_prefix}.downsample.1",
                            ]
                        )

        """获取 backbone 的前缀路径"""
        # 检查是否有 backbone 属性
        if hasattr(model, "backbone"):
            return "backbone."
        return ""

    def _is_qat_resnet(self, model: nn.Module, prefix: str) -> bool:
        """Check if model is QAT-friendly ResNet by detecting skip_add attribute."""
        try:
            first_block_name = f"{prefix}layers.0.0"
            if self._module_exists(model, first_block_name):
                parts = first_block_name.split(".")
                module = model
                for part in parts:
                    if part.isdigit():
                        module = module[int(part)]
                    else:
                        module = getattr(module, part)
                return hasattr(module, "skip_add")
        except Exception:
            pass
        return False


class TCNFusionStrategy(FusionStrategy):
    """TCN (Temporal Convolutional Network) fusion strategy."""

    def __init__(self, num_layers: int = 3):
        """Args:
        num_layers: Number of TCN layers (TemporalBlocks).
        """
        self.num_layers = num_layers

    def get_fusion_modules(self, model: nn.Module) -> List[List[str]]:
        """Get TCN fusion module list."""
        fusion_modules = []

        prefix = self._get_backbone_prefix(model)
        network_prefix = f"{prefix}network"

        num_layers = self._detect_num_layers(model, network_prefix)
        if num_layers > 0:
            self.num_layers = num_layers

        logger.info(f"[QAT] TCN detected with {self.num_layers} layers")

        for layer_idx in range(self.num_layers):
            block_prefix = f"{network_prefix}.{layer_idx}"

            fusion_modules.append([f"{block_prefix}.conv1", f"{block_prefix}.bn1"])
            fusion_modules.append([f"{block_prefix}.conv2", f"{block_prefix}.bn2"])

            if self._module_exists(model, f"{block_prefix}.downsample"):
                fusion_modules.append(
                    [f"{block_prefix}.downsample.0", f"{block_prefix}.downsample.1"]
                )

        # 验证并返回有效的融合模块
        return self.validate_fusion_modules(model, fusion_modules)

    def _get_backbone_prefix(self, model: nn.Module) -> str:
        """获取 backbone 的前缀路径"""
        # 检查是否有 backbone 属性
        if hasattr(model, "backbone"):
            return "backbone."
        return ""

    def _detect_num_layers(self, model: nn.Module, network_prefix: str) -> int:
        """Auto-detect number of TCN layers."""
        try:
            parts = network_prefix.split(".")
            module = model
            for part in parts:
                if part:
                    module = getattr(module, part)

            if isinstance(module, nn.Sequential):
                return len(module)
        except (AttributeError, TypeError):
            pass

        return 0


class DefaultFusionStrategy(FusionStrategy):
    """Default fusion strategy that auto-detects Conv+BN+ReLU patterns."""

    def get_fusion_modules(self, model: nn.Module) -> List[List[str]]:
        """Auto-detect modules to fuse by finding Conv-BN-ReLU or Conv-BN patterns."""
        fusion_modules = []

        named_modules = dict(model.named_modules())
        module_names = list(named_modules.keys())

        for name, module in named_modules.items():
            if isinstance(module, nn.Sequential):
                self._find_fusion_in_sequential(name, module, fusion_modules)

        return fusion_modules

    def _find_fusion_in_sequential(
        self, prefix: str, sequential: nn.Sequential, fusion_modules: List[List[str]]
    ):
        """Find fusible modules in Sequential container."""
        children = list(sequential.children())
        child_names = [
            f"{prefix}.{i}" if prefix else str(i) for i in range(len(children))
        ]

        i = 0
        while i < len(children):
            if i + 2 < len(children):
                conv = children[i]
                bn = children[i + 1]
                relu = children[i + 2]

                if self._is_conv(conv) and self._is_bn(bn) and self._is_relu(relu):
                    fusion_modules.append(
                        [child_names[i], child_names[i + 1], child_names[i + 2]]
                    )
                    i += 3
                    continue

            if i + 1 < len(children):
                conv = children[i]
                bn = children[i + 1]

                if self._is_conv(conv) and self._is_bn(bn):
                    fusion_modules.append([child_names[i], child_names[i + 1]])
                    i += 2
                    continue

            i += 1

    @staticmethod
    def _is_conv(module: nn.Module) -> bool:
        """Check if module is Conv layer."""
        return isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d))

    @staticmethod
    def _is_bn(module: nn.Module) -> bool:
        """Check if module is BatchNorm layer."""
        return isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))

    @staticmethod
    def _is_relu(module: nn.Module) -> bool:
        """Check if module is ReLU layer."""
        return isinstance(module, (nn.ReLU, nn.ReLU6))


def get_fusion_strategy(
    model: nn.Module, strategy_name: Optional[str] = None
) -> FusionStrategy:
    """Get appropriate fusion strategy for model.

    Args:
        model: Model instance.
        strategy_name: Strategy name (optional).

    Returns:
        FusionStrategy instance.
    """
    if strategy_name is not None:
        strategy_name = strategy_name.lower()

        if strategy_name.startswith("resnet") and len(strategy_name) > 6:
            try:
                depth = int(strategy_name[6:])
                return ResNetFusionStrategy(depth=depth)
            except ValueError:
                pass

        if strategy_name in FUSION_STRATEGIES:
            strategy_cls = FUSION_STRATEGIES.get(strategy_name)
            return strategy_cls()
        else:
            logger.warning(f"Unknown fusion strategy '{strategy_name}', using default")
            return FUSION_STRATEGIES.get("default")()

    backbone = getattr(model, "backbone", model)
    backbone_name = backbone.__class__.__name__.lower()

    if "resnetbackbone" in backbone_name or "resnet" in backbone_name:
        depth = getattr(backbone, "depth", 18)
        return FUSION_STRATEGIES.get("resnet")(depth=depth)

    if "conv1d" in backbone_name:
        num_layers = len(getattr(backbone, "layers", []))
        return FUSION_STRATEGIES.get("conv1d")(num_layers=num_layers or 4)

    if "tcn" in backbone_name:
        network = getattr(backbone, "network", None)
        num_layers = len(network) if network is not None else 3
        return FUSION_STRATEGIES.get("tcn")(num_layers=num_layers)

    return FUSION_STRATEGIES.get("default")()


def _register_fusion_strategies():
    """Register fusion strategies on first call."""
    strategies = {
        "resnet": ResNetFusionStrategy,
        "tcn": TCNFusionStrategy,
        "default": DefaultFusionStrategy,
    }

    for name, strategy_cls in strategies.items():
        if name not in FUSION_STRATEGIES:
            FUSION_STRATEGIES.register_module(strategy_cls, name, force=False)


try:
    _register_fusion_strategies()
except KeyError:
    pass
