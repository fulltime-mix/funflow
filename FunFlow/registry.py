"""Component registry for unified registration and management."""

from typing import Dict, Any, Optional, Type, Callable, List


class Registry:
    """Generic registry for components.

    Example:
        >>> MODELS = Registry('models')
        >>> @MODELS.register('ResNet18')
        ... class ResNet18(nn.Module):
        ...     pass
        >>> model = MODELS.build({'type': 'ResNet18', 'num_classes': 2})
    """

    def __init__(self, name: str):
        self._name = name
        self._modules: Dict[str, Type] = {}

    @property
    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return len(self._modules)

    def __contains__(self, key: str) -> bool:
        return key in self._modules

    def __repr__(self) -> str:
        return f"Registry(name={self._name}, items={list(self._modules.keys())})"

    def __iter__(self):
        return iter(self._modules)

    def keys(self) -> List[str]:
        return list(self._modules.keys())

    def values(self) -> List[Type]:
        return list(self._modules.values())

    def items(self):
        return self._modules.items()

    def get(self, key: str) -> Optional[Type]:
        return self._modules.get(key)

    def register(self, name: Optional[str] = None, force: bool = False) -> Callable:
        """Register as decorator.

        Example:
            >>> @MODELS.register('MyModel')
            ... class MyModel(nn.Module):
            ...     pass
        """

        def decorator(cls: Type) -> Type:
            module_name = name or cls.__name__
            if not force and module_name in self._modules:
                raise KeyError(f"'{module_name}' already registered in '{self._name}'")
            self._modules[module_name] = cls
            return cls

        return decorator

    def register_module(
        self, module: Type = None, name: Optional[str] = None, force: bool = False
    ):
        """Register as function.

        Example:
            >>> MODELS.register_module(MyModel, 'MyModel')
        """
        if module is not None:
            module_name = name or module.__name__
            if not force and module_name in self._modules:
                raise KeyError(f"'{module_name}' already registered in '{self._name}'")
            self._modules[module_name] = module
            return module
        return self.register(name, force)

    def build(self, cfg: Dict[str, Any], **default_args) -> Any:
        """Build component instance from config.

        Args:
            cfg: Config dict with 'type' key
            **default_args: Default arguments

        Example:
            >>> model = MODELS.build({'type': 'ResNet18', 'num_classes': 2})
        """
        if not isinstance(cfg, dict):
            raise TypeError(f"cfg must be a dict, got {type(cfg)}")
        if "type" not in cfg:
            raise KeyError("cfg must contain 'type' key")

        cfg = cfg.copy()
        module_type = cfg.pop("type")

        if module_type not in self._modules:
            raise KeyError(
                f"'{module_type}' not registered in '{self._name}'. "
                f"Available: {self.keys()}"
            )

        for k, v in default_args.items():
            cfg.setdefault(k, v)

        return self._modules[module_type](**cfg)


MODELS = Registry("models")
PREPROCESSINGS = Registry("preprocessings")
BACKBONES = Registry("backbones")
NECKS = Registry("necks")
HEADS = Registry("heads")
LOSSES = Registry("losses")
DATASETS = Registry("datasets")
OPTIMIZERS = Registry("optimizers")
SCHEDULERS = Registry("schedulers")
HOOKS = Registry("hooks")
EVALUATORS = Registry("evaluators")
INFERENCERS = Registry("inferencers")
EXPORTERS = Registry("exporters")
MODEL_LOADERS = Registry("model_loaders")
FUSION_STRATEGIES = Registry("fusion_strategies")


def build_model(cfg: Dict[str, Any], **kwargs) -> Any:
    """Build model."""
    return MODELS.build(cfg, **kwargs)


def build_preprocessing(cfg: Dict[str, Any], **kwargs) -> Any:
    """Build preprocessing module."""
    return PREPROCESSINGS.build(cfg, **kwargs)


def build_backbone(cfg: Dict[str, Any], **kwargs) -> Any:
    """Build backbone."""
    return BACKBONES.build(cfg, **kwargs)


def build_neck(cfg: Dict[str, Any], **kwargs) -> Any:
    """Build neck."""
    return NECKS.build(cfg, **kwargs)


def build_head(cfg: Dict[str, Any], **kwargs) -> Any:
    """Build head."""
    return HEADS.build(cfg, **kwargs)


def build_loss(cfg: Dict[str, Any], **kwargs) -> Any:
    """Build loss function."""
    return LOSSES.build(cfg, **kwargs)


def build_dataset(cfg: Dict[str, Any], **kwargs) -> Any:
    """Build dataset."""
    return DATASETS.build(cfg, **kwargs)


def build_optimizer(cfg: Dict[str, Any], **kwargs) -> Any:
    """Build optimizer."""
    return OPTIMIZERS.build(cfg, **kwargs)


def build_scheduler(cfg: Dict[str, Any], **kwargs) -> Any:
    """Build learning rate scheduler."""
    return SCHEDULERS.build(cfg, **kwargs)


def build_hook(cfg: Dict[str, Any], **kwargs) -> Any:
    """Build training hook."""
    return HOOKS.build(cfg, **kwargs)


def build_evaluator(cfg: Dict[str, Any], **kwargs) -> Any:
    """Build evaluator."""
    return EVALUATORS.build(cfg, **kwargs)


def build_inferencer(cfg: Dict[str, Any], **kwargs) -> Any:
    """Build inferencer."""
    return INFERENCERS.build(cfg, **kwargs)


def build_exporter(cfg: Dict[str, Any], **kwargs) -> Any:
    """Build exporter."""
    return EXPORTERS.build(cfg, **kwargs)


def build_model_loader(cfg: Dict[str, Any], **kwargs) -> Any:
    """Build model loader."""
    return MODEL_LOADERS.build(cfg, **kwargs)


def build_fusion_strategy(cfg: Dict[str, Any], **kwargs) -> Any:
    """Build fusion strategy."""
    return FUSION_STRATEGIES.build(cfg, **kwargs)
