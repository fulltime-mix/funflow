#!/usr/bin/env python
"""Export utility functions."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import yaml

logger = logging.getLogger(__name__)


def prepare_dummy_input(
    config: Optional[Dict[str, Any]] = None,
    input_shape: Optional[Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]] = None,
    device: str = "cpu",
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """Prepare dummy input for export.

    Args:
        config: Model config dict
        input_shape: Input shape tuple or dict
        device: Device to place input on

    Returns:
        Dummy input tensor or dict
    """
    device = torch.device(device)

    if input_shape is not None:
        if isinstance(input_shape, dict):
            return {
                k: torch.randn(*shape, device=device)
                for k, shape in input_shape.items()
            }
        return torch.randn(*input_shape, device=device)

    if config is not None:
        shape = _infer_shape_from_config(config)
        if shape:
            return torch.randn(*shape, device=device)

    logger.warning("Using default input shape (1, 16, 1000)")
    return torch.randn(1, 16, 1000, device=device)


def _infer_shape_from_config(config: Dict[str, Any]) -> Optional[Tuple[int, ...]]:
    """Infer input shape from config."""
    if "dataset" in config:
        dataset_cfg = config["dataset"]
        if "input_dim" in dataset_cfg and "seq_len" in dataset_cfg:
            return (1, dataset_cfg["input_dim"], dataset_cfg["seq_len"])

    if "model" in config:
        model_cfg = config["model"]
        if "backbone" in model_cfg:
            backbone_cfg = model_cfg["backbone"]
            if "input_dim" in backbone_cfg:
                return (1, backbone_cfg["input_dim"], 1000)
    return None


def get_output_path(
    checkpoint_path: str,
    output_dir: Optional[str] = None,
    suffix: str = ".onnx",
) -> str:
    """Generate output path.

    Args:
        checkpoint_path: Checkpoint file path
        output_dir: Output directory (default: same as checkpoint)
        suffix: File suffix with extension

    Returns:
        Output file path
    """
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir) if output_dir else checkpoint_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir / f"{checkpoint_path.stem}{suffix}")


def print_result(result: Dict[str, Any]) -> None:
    """Print export result.

    Args:
        result: Export result dict
    """
    print("\n" + "=" * 50)
    if result.get("success"):
        print(f"✓ Export: {result['output_path']}")
        print(
            f"  Size: {result['file_size_mb']:.2f} MB | Time: {result['export_time_ms']:.1f} ms"
        )
        print(f"  Verified: {'✓' if result.get('verified') else '✗'}")
    else:
        print(f"✗ Export failed: {result.get('error', 'Unknown error')}")
    print("=" * 50 + "\n")


def load_export_config(config_path: str) -> Dict[str, Any]:
    """Load export config from YAML/JSON file.

    Args:
        config_path: Path to config file

    Returns:
        Config dict
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        if config_path.suffix in [".yaml", ".yml"]:
            return yaml.safe_load(f)
        elif config_path.suffix == ".json":
            return json.load(f)
        else:
            raise ValueError(f"Unsupported format: {config_path.suffix}")


def parse_dynamic_axes(dynamic_axes_str: str) -> Dict[str, Dict[int, str]]:
    """Parse dynamic axes string.

    Args:
        dynamic_axes_str: String like "input:0=batch,2=seq;output:0=batch"

    Returns:
        Dynamic axes dict
    """
    if not dynamic_axes_str:
        return {}

    result = {}
    for spec in dynamic_axes_str.split(";"):
        spec = spec.strip()
        if ":" not in spec:
            continue
        name, axes = spec.split(":", 1)
        result[name.strip()] = {
            int(k): v
            for k, v in (item.split("=") for item in axes.split(",") if "=" in item)
        }
    return result


def parse_list(value: str) -> List[str]:
    """Parse comma-separated string.

    Args:
        value: Comma-separated string

    Returns:
        List of strings
    """
    return [v.strip() for v in value.split(",")] if value else []


def parse_input_shape(
    shape_str: str,
) -> Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]:
    """Parse input shape string.

    Args:
        shape_str: Shape string like "1,16,1000" or "data:1,16,1000;mask:1,1000"

    Returns:
        Shape tuple or dict
    """
    if ":" in shape_str:
        return {
            k.strip(): tuple(map(int, v.split(",")))
            for k, v in (p.split(":") for p in shape_str.split(";"))
        }
    return tuple(map(int, shape_str.split(",")))
