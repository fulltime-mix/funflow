"""Optimizer builder module"""

from typing import Dict, Any, Optional, List
import re

import torch
import torch.nn as nn
from torch.optim import Optimizer

from FunFlow.registry import OPTIMIZERS

OPTIMIZERS.register_module(torch.optim.SGD, "SGD")
OPTIMIZERS.register_module(torch.optim.Adam, "Adam")
OPTIMIZERS.register_module(torch.optim.AdamW, "AdamW")
OPTIMIZERS.register_module(torch.optim.RMSprop, "RMSprop")
OPTIMIZERS.register_module(torch.optim.Adagrad, "Adagrad")
OPTIMIZERS.register_module(torch.optim.Adadelta, "Adadelta")


def build_optimizer(
    model: nn.Module,
    cfg: Dict[str, Any],
    paramwise_cfg: Optional[Dict[str, Any]] = None,
) -> Optimizer:
    """Build optimizer

    Args:
        model: Model
        cfg: Optimizer config
        paramwise_cfg: Parameter-wise config (optional)

    Returns:
        Optimizer instance
    """
    cfg = cfg.copy()
    optimizer_type = cfg.pop("type")

    if paramwise_cfg:
        params = get_paramwise_params(model, cfg.get("lr", 1e-3), paramwise_cfg)
    else:
        params = filter(lambda p: p.requires_grad, model.parameters())

    if optimizer_type in OPTIMIZERS:
        optimizer_cls = OPTIMIZERS.get(optimizer_type)
    else:
        optimizer_cls = getattr(torch.optim, optimizer_type)

    return optimizer_cls(params, **cfg)


def get_paramwise_params(
    model: nn.Module,
    base_lr: float,
    paramwise_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Get parameter-wise config for differential learning rates

    Args:
        model: Model
        base_lr: Base learning rate
        paramwise_cfg: Parameter-wise config

    Returns:
        List of parameter groups
    """
    backbone_lr_mult = paramwise_cfg.get("backbone_lr_mult", 1.0)
    head_lr_mult = paramwise_cfg.get("head_lr_mult", 1.0)
    neck_lr_mult = paramwise_cfg.get("neck_lr_mult", 1.0)
    bias_lr_mult = paramwise_cfg.get("bias_lr_mult", 1.0)
    bias_decay_mult = paramwise_cfg.get("bias_decay_mult", 0.0)
    norm_decay_mult = paramwise_cfg.get("norm_decay_mult", 0.0)
    custom_patterns = paramwise_cfg.get("custom_patterns", {})
    base_weight_decay = paramwise_cfg.get("base_weight_decay", None)

    compiled_patterns = {}
    for pattern in custom_patterns.keys():
        if pattern.startswith("regex:"):
            try:
                compiled_patterns[pattern] = re.compile(pattern[6:])
            except re.error as e:
                raise ValueError(f'Invalid regex pattern "{pattern[6:]}": {e}')

    param_groups_dict = {}

    for param_name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        lr_mult = 1.0
        decay_mult = 1.0
        group_name = "other"

        matched = False
        for pattern, pattern_cfg in custom_patterns.items():
            if pattern in compiled_patterns:
                if compiled_patterns[pattern].search(param_name):
                    matched = True
            elif pattern in param_name:
                matched = True

            if matched:
                lr_mult = pattern_cfg.get("lr_mult", 1.0)
                decay_mult = pattern_cfg.get("decay_mult", 1.0)
                group_name = f'custom_{pattern.replace("regex:", "")}'
                break

        if not matched:
            if "backbone" in param_name:
                lr_mult = backbone_lr_mult
                group_name = "backbone"
            elif "head" in param_name:
                lr_mult = head_lr_mult
                group_name = "head"
            elif "neck" in param_name:
                lr_mult = neck_lr_mult
                group_name = "neck"

        if param_name.endswith(".bias"):
            lr_mult *= bias_lr_mult
            decay_mult = bias_decay_mult
            group_name = f"{group_name}_bias"
        elif any(
            f".{kw}." in param_name or param_name.startswith(f"{kw}.")
            for kw in ["bn", "norm", "ln", "gn", "in"]
        ):
            decay_mult = norm_decay_mult
            group_name = f"{group_name}_norm"

        group_key = (lr_mult, decay_mult, group_name)
        if group_key not in param_groups_dict:
            param_groups_dict[group_key] = {
                "params": [],
                "lr": base_lr * lr_mult,
                "group_name": group_name,
            }
            if base_weight_decay is not None:
                param_groups_dict[group_key]["weight_decay"] = (
                    base_weight_decay * decay_mult
                )

        param_groups_dict[group_key]["params"].append(param)

    param_groups = sorted(param_groups_dict.values(), key=lambda x: x["group_name"])

    return param_groups
