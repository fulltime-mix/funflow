"""FunFlow - A flexible deep learning training framework."""

__version__ = "0.1.0"
__author__ = "FunFlow Team"

import importlib
import pkgutil
import sys
import os


def import_submodules(package, recursive=True, silent=True):
    """Recursively import all submodules to trigger decorator registration.

    Args:
        package: Package name or package object
        recursive: Recursively import subpackages
        silent: Silently handle import errors

    Returns:
        Dict of imported modules
    """
    if isinstance(package, str):
        try:
            package = importlib.import_module(package)
        except Exception as e:
            if not silent:
                print(f"Failed to import package {package}: {e}")
            return {}

    results = {}
    for loader, name, is_pkg in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        try:
            results[name] = importlib.import_module(name)
        except Exception as e:
            if not silent:
                print(f"Failed to import {name}: {e}")

        if recursive and is_pkg:
            results.update(import_submodules(name, recursive, silent))

    return results


def auto_import_local(silent: bool = True):
    """Auto-import all modules from local/ directory in current working directory.

    Args:
        silent: Silently handle import errors

    Returns:
        Dict of imported modules
    """
    local_path = os.path.join(os.getcwd(), "local")
    if not os.path.isdir(local_path):
        return {}

    if local_path not in sys.path:
        sys.path.insert(0, local_path)

    results = {}
    for root, _, files in os.walk(local_path):
        for file in files:
            if file.endswith(".py") and not file.startswith("_"):
                rel_path = os.path.relpath(os.path.join(root, file), local_path)
                module_name = rel_path[:-3].replace(os.sep, ".")
                try:
                    results[module_name] = importlib.import_module(module_name)
                except Exception as e:
                    if not silent:
                        print(f"Failed to import local module {module_name}: {e}")

    return results


import_submodules(__name__)
auto_import_local(silent=True)
from FunFlow.registry import (
    Registry,
    MODELS,
    PREPROCESSINGS,
    BACKBONES,
    NECKS,
    HEADS,
    LOSSES,
    DATASETS,
    OPTIMIZERS,
    SCHEDULERS,
    HOOKS,
    INFERENCERS,
    EXPORTERS,
    EVALUATORS,
    FUSION_STRATEGIES,
    MODEL_LOADERS,
    build_model,
    build_preprocessing,
    build_backbone,
    build_neck,
    build_head,
    build_loss,
    build_dataset,
    build_optimizer,
    build_scheduler,
    build_hook,
    build_evaluator,
    build_inferencer,
    build_exporter,
    build_fusion_strategy,
    build_model_loader,
)

from FunFlow.logger import (
    setup_logger,
    get_logger,
    reset_logger,
    FunFlowLogger,
    CONSOLE,
)

__all__ = [
    "import_submodules",
    "auto_import_local",
    # 注册表对象
    "Registry",
    "MODELS",
    "PREPROCESSINGS",
    "BACKBONES",
    "NECKS",
    "HEADS",
    "LOSSES",
    "DATASETS",
    "OPTIMIZERS",
    "SCHEDULERS",
    "HOOKS",
    "INFERENCERS",
    "EXPORTERS",
    "EVALUATORS",
    "FUSION_STRATEGIES",
    "MODEL_LOADERS",
    # 构建函数
    "build_model",
    "build_preprocessing",
    "build_backbone",
    "build_neck",
    "build_head",
    "build_loss",
    "build_dataset",
    "build_optimizer",
    "build_scheduler",
    "build_hook",
    "build_evaluator",
    "build_inferencer",
    "build_exporter",
    "build_fusion_strategy",
    "build_model_loader",
    # 日志工具
    "setup_logger",
    "get_logger",
    "reset_logger",
    "FunFlowLogger",
    "CONSOLE",
]
