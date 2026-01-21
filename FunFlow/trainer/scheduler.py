"""Learning rate schedulers

Scheduler stepping strategies:
- Step-level: Call scheduler.step() after each optimization step
- Epoch-level: Call scheduler.step() after each epoch

Use infer_scheduler_stepping() to automatically determine scheduler type.
"""

import math
from typing import Dict, Any, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from FunFlow.registry import SCHEDULERS

STEP_LEVEL_SCHEDULERS = {
    "OneCycleLR",
    "CyclicLR",
    "LinearWarmupLR",
}

EPOCH_LEVEL_SCHEDULERS = {
    "CosineAnnealingLR",
    "StepLR",
    "MultiStepLR",
    "ExponentialLR",
    "ReduceLROnPlateau",
    "CosineAnnealingWarmRestarts",
    "WarmupCosineAnnealingLR",
    "WarmupMultiStepLR",
    "PolynomialLR",
}


def infer_scheduler_stepping(
    scheduler: Union[_LRScheduler, None] = None,
    scheduler_type: str = None,
) -> bool:
    """Infer scheduler stepping method

    Args:
        scheduler: Scheduler instance (priority)
        scheduler_type: Scheduler type name

    Returns:
        True: epoch-level stepping
        False: step-level stepping
    """
    if scheduler_type:
        if scheduler_type in STEP_LEVEL_SCHEDULERS:
            return False
        if scheduler_type in EPOCH_LEVEL_SCHEDULERS:
            return True

    if scheduler is not None:
        class_name = scheduler.__class__.__name__
        if class_name in STEP_LEVEL_SCHEDULERS:
            return False
        if class_name in EPOCH_LEVEL_SCHEDULERS:
            return True

        step_indicators = ["total_steps", "step_size_up", "step_size_down"]
        for attr in step_indicators:
            if hasattr(scheduler, attr):
                return False

        epoch_indicators = ["T_max", "milestones", "T_0"]
        for attr in epoch_indicators:
            if hasattr(scheduler, attr):
                return True

    return True


SCHEDULERS.register_module(
    torch.optim.lr_scheduler.StepLR, "StepLR"
)
SCHEDULERS.register_module(
    torch.optim.lr_scheduler.MultiStepLR, "MultiStepLR"
)
SCHEDULERS.register_module(
    torch.optim.lr_scheduler.ExponentialLR, "ExponentialLR"
)
SCHEDULERS.register_module(
    torch.optim.lr_scheduler.CosineAnnealingLR, "CosineAnnealingLR"
)
SCHEDULERS.register_module(
    torch.optim.lr_scheduler.ReduceLROnPlateau, "ReduceLROnPlateau"
)
SCHEDULERS.register_module(
    torch.optim.lr_scheduler.OneCycleLR, "OneCycleLR"
)
SCHEDULERS.register_module(
    torch.optim.lr_scheduler.CyclicLR, "CyclicLR"
)
SCHEDULERS.register_module(
    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts, "CosineAnnealingWarmRestarts"
)


def build_scheduler(
    optimizer: Optimizer,
    cfg: Dict[str, Any],
) -> _LRScheduler:
    """Build learning rate scheduler

    Args:
        optimizer: Optimizer
        cfg: Scheduler config

    Returns:
        Scheduler instance
    """
    cfg = cfg.copy()
    scheduler_type = cfg.pop("type")

    if scheduler_type in SCHEDULERS:
        scheduler_cls = SCHEDULERS.get(scheduler_type)
    else:
        scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_type)

    return scheduler_cls(optimizer, **cfg)


@SCHEDULERS.register("WarmupCosineAnnealingLR")
class WarmupCosineAnnealingLR(_LRScheduler):
    """Warmup cosine annealing scheduler

    Args:
        optimizer: Optimizer
        warmup_epochs: Warmup epochs
        max_epochs: Total epochs
        warmup_lr: Initial warmup learning rate
        min_lr: Minimum learning rate
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_lr: float = 1e-6,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_lr = warmup_lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / self.warmup_epochs
            return [
                self.warmup_lr + (base_lr - self.warmup_lr) * alpha
                for base_lr in self.base_lrs
            ]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / (
                self.max_epochs - self.warmup_epochs
            )
            return [
                self.min_lr
                + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


@SCHEDULERS.register("LinearWarmupLR")
class LinearWarmupLR(_LRScheduler):
    """Linear warmup scheduler

    Args:
        optimizer: Optimizer
        warmup_steps: Warmup steps
        warmup_lr: Initial warmup learning rate
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        warmup_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.warmup_lr = warmup_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            alpha = self.last_epoch / self.warmup_steps
            return [
                self.warmup_lr + (base_lr - self.warmup_lr) * alpha
                for base_lr in self.base_lrs
            ]
        return self.base_lrs


@SCHEDULERS.register("WarmupMultiStepLR")
class WarmupMultiStepLR(_LRScheduler):
    """Warmup multi-step scheduler

    Args:
        optimizer: Optimizer
        milestones: Epochs to decrease learning rate
        gamma: Learning rate decay factor
        warmup_epochs: Warmup epochs
        warmup_lr: Initial warmup learning rate
    """

    def __init__(
        self,
        optimizer: Optimizer,
        milestones: list,
        gamma: float = 0.1,
        warmup_epochs: int = 0,
        warmup_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.milestones = sorted(milestones)
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / self.warmup_epochs
            return [
                self.warmup_lr + (base_lr - self.warmup_lr) * alpha
                for base_lr in self.base_lrs
            ]
        else:
            factor = 1.0
            for milestone in self.milestones:
                if self.last_epoch >= milestone:
                    factor *= self.gamma
            return [base_lr * factor for base_lr in self.base_lrs]


@SCHEDULERS.register("PolynomialLR")
class PolynomialLR(_LRScheduler):
    """Polynomial learning rate scheduler

    Args:
        optimizer: Optimizer
        max_epochs: Total epochs
        power: Polynomial power
        min_lr: Minimum learning rate
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_epochs: int,
        power: float = 0.9,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        self.max_epochs = max_epochs
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.max_epochs:
            return [self.min_lr for _ in self.base_lrs]

        factor = (1 - self.last_epoch / self.max_epochs) ** self.power
        return [max(base_lr * factor, self.min_lr) for base_lr in self.base_lrs]
