"""Checkpoint utilities for loading and saving models"""

import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    map_location: str = "cpu",
    strict: bool = True,
    strip_prefix: str = "",
) -> Dict[str, Any]:
    """Load model checkpoint

    Args:
        model: PyTorch model
        checkpoint_path: Checkpoint file path
        map_location: Device mapping
        strict: Strict state dict matching
        strip_prefix: Key prefix to remove (e.g., 'module.')

    Returns:
        Additional checkpoint metadata
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
        checkpoint = {"state_dict": state_dict}

    if strip_prefix:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith(strip_prefix):
                new_state_dict[k[len(strip_prefix) :]] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict

    if hasattr(model, "module"):
        model.module.load_state_dict(state_dict, strict=strict)
    else:
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(state_dict.keys())

        if any(k.startswith("module.") for k in ckpt_keys) and not any(
            k.startswith("module.") for k in model_keys
        ):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        elif any(k.startswith("module.") for k in model_keys) and not any(
            k.startswith("module.") for k in ckpt_keys
        ):
            state_dict = {"module." + k: v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=strict)

    return checkpoint


def save_checkpoint(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    best_metric: Optional[float] = None,
    config: Optional[Dict] = None,
    **kwargs,
) -> None:
    """Save model checkpoint

    Args:
        model: PyTorch model
        checkpoint_path: Save path
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        step: Current step
        best_metric: Best metric value
        config: Configuration dict
        **kwargs: Additional data to save
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "module"):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    checkpoint = {
        "model_state_dict": model_state_dict,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if epoch is not None:
        checkpoint["epoch"] = epoch

    if step is not None:
        checkpoint["step"] = step

    if best_metric is not None:
        checkpoint["best_metric"] = best_metric

    if config is not None:
        checkpoint["config"] = config

    checkpoint.update(kwargs)

    torch.save(checkpoint, checkpoint_path)
