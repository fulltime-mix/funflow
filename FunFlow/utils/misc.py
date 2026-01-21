"""Miscellaneous utility functions"""

import os
import random
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def set_random_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seed

    Args:
        seed: Random seed
        deterministic: Enable deterministic mode (may impact performance)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True


def get_timestamp() -> str:
    """Get timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters

    Args:
        model: PyTorch model
        trainable_only: Count only trainable parameters

    Returns:
        Total parameter count
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def make_divisible(value: int, divisor: int = 8) -> int:
    """Make value divisible by divisor

    Args:
        value: Original value
        divisor: Divisor

    Returns:
        Nearest value divisible by divisor
    """
    new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


def get_device(device: Optional[str] = None) -> torch.device:
    """Get device

    Args:
        device: Device string ('cuda', 'cpu', 'cuda:0', etc.)

    Returns:
        torch.device object
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def get_model_info(model: nn.Module, input_shape: Tuple = (1, 3, 224, 224)) -> dict:
    """Get model information

    Args:
        model: PyTorch model
        input_shape: Input shape

    Returns:
        Dict with parameter count, FLOPs, etc.
    """
    info = {
        "parameters": count_parameters(model, trainable_only=False),
        "trainable_parameters": count_parameters(model, trainable_only=True),
    }

    try:
        from thop import profile

        device = next(model.parameters()).device
        dummy_input = torch.randn(*input_shape).to(device)
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        info["flops"] = flops
        info["macs"] = flops / 2
    except ImportError:
        pass
    except Exception:
        pass

    return info


def format_number(num: int) -> str:
    """Format number to human-readable string

    Args:
        num: Number

    Returns:
        Formatted string (e.g., '1.23M', '456.78K')
    """
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    return str(num)


class AverageMeter:
    """Compute and store average and current value"""

    def __init__(self, name: str = "", fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class Timer:
    """Timer for measuring elapsed time"""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.start_time = None
        self.end_time = None
        self.elapsed = 0

    def start(self) -> "Timer":
        self.start_time = datetime.now()
        return self

    def stop(self) -> float:
        self.end_time = datetime.now()
        self.elapsed = (self.end_time - self.start_time).total_seconds()
        return self.elapsed

    def __enter__(self) -> "Timer":
        return self.start()

    def __exit__(self, *args) -> None:
        self.stop()


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate"""
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Set learning rate"""
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def freeze_layers(model: nn.Module, freeze_names: list, verbose: bool = True) -> int:
    """Freeze specified layers with substring or regex matching

    Args:
        model: PyTorch model
        freeze_names: Layer names to freeze, supports:
            - Substring match: 'layer1' freezes all params containing 'layer1'
            - Regex match: 'regex:layer[0-2]' uses regex pattern
        verbose: Print freezing info

    Returns:
        Number of frozen parameters
    """
    import re
    import logging

    if not freeze_names:
        return 0

    logger = logging.getLogger("FunFlow")

    compiled_patterns = {}
    for pattern in freeze_names:
        if pattern.startswith("regex:"):
            try:
                compiled_patterns[pattern] = re.compile(pattern[6:])
            except re.error as e:
                raise ValueError(f'Invalid regex pattern "{pattern[6:]}": {e}')

    frozen_params = []
    frozen_count = 0
    unfrozen_params = []

    for name, param in model.named_parameters():
        should_freeze = False
        matched_pattern = None

        for pattern in freeze_names:
            if pattern in compiled_patterns:

                if compiled_patterns[pattern].search(name):
                    should_freeze = True
                    matched_pattern = pattern
                    break
            else:
                if pattern in name:
                    should_freeze = True
                    matched_pattern = pattern
                    break

        if should_freeze and param.requires_grad:
            param.requires_grad = False
            frozen_count += 1
            frozen_params.append((name, matched_pattern))
        elif not should_freeze and param.requires_grad:
            unfrozen_params.append(name)

    if verbose and frozen_count > 0:
        logger.debug(f"=== Frozen {frozen_count} parameters ===")
        pattern_groups = {}
        for param_name, pattern in frozen_params:
            if pattern not in pattern_groups:
                pattern_groups[pattern] = []
            pattern_groups[pattern].append(param_name)

        for pattern, params in pattern_groups.items():
            logger.debug(f"Pattern: {pattern} (matched {len(params)} params)")
            for p in params:
                logger.debug(f"  - {p}")
        logger.debug("=" * 40)

        if unfrozen_params:
            logger.debug(f"\n=== Unfrozen {len(unfrozen_params)} parameters ===")
            for p in unfrozen_params:
                logger.debug(f"  - {p}")
            logger.debug("=" * 40)

    elif verbose and frozen_count == 0:
        logger.debug(
            f"⚠️  Warning: No parameters matched freeze patterns: {freeze_names}"
        )
        logger.debug("Available parameter prefixes (first 10):")
        param_names = [name for name, _ in model.named_parameters()]
        for name in param_names[:10]:
            logger.debug(f"  - {name}")
        if len(param_names) > 10:
            logger.debug(f"  ... and {len(param_names) - 10} more parameters")
        logger.debug("=" * 40)

    return frozen_count


def init_weights(module: nn.Module, method: str = "kaiming") -> None:
    """Initialize model weights

    Args:
        module: PyTorch module
        method: Init method ('kaiming', 'xavier', 'normal', 'constant')
    """
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        if method == "kaiming":
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif method == "xavier":
            nn.init.xavier_normal_(module.weight)
        elif method == "normal":
            nn.init.normal_(module.weight, mean=0, std=0.02)
        elif method == "constant":
            nn.init.constant_(module.weight, 0)

        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def calculate_class_weights(labels: list, method: str = "inverse") -> torch.Tensor:
    """Calculate class weights for imbalanced datasets

    Args:
        labels: Label list
        method: Calculation method ('inverse', 'sqrt_inverse', 'effective_number')

    Returns:
        Class weight tensor
    """
    labels = np.array(labels)
    classes, counts = np.unique(labels, return_counts=True)
    n_classes = len(classes)
    n_samples = len(labels)

    if method == "inverse":
        weights = n_samples / (n_classes * counts)
    elif method == "sqrt_inverse":
        weights = np.sqrt(n_samples / (n_classes * counts))
    elif method == "effective_number":
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / effective_num
    else:
        weights = np.ones(n_classes)

    weights = weights / weights.sum() * n_classes

    return torch.FloatTensor(weights)
