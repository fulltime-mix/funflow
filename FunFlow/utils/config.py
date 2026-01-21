"""Configuration management utilities

Provides unified config loading, saving, and merging with inheritance support.

Usage:
    from FunFlow.utils import load_config, save_config, merge_configs

    config = load_config('config.yaml')  # Supports _base_ inheritance
    save_config(config, 'output/config.yaml')
    merged = merge_configs(base_config, override_config)
"""

import os
import copy
from pathlib import Path
from typing import Any, Dict, Union

import yaml


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML config with inheritance support

    Supports _base_ field for config inheritance:
        _base_: base_config.yaml
        or
        _base_:
          - base1.yaml
          - base2.yaml

    Args:
        config_path: Config file path

    Returns:
        Config dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    if "_base_" in config:
        base_configs = config.pop("_base_")
        if isinstance(base_configs, str):
            base_configs = [base_configs]

        merged_config = {}
        for base_path in base_configs:
            if not os.path.isabs(base_path):
                base_path = config_path.parent / base_path
            base_config = load_config(base_path)
            merged_config = merge_configs(merged_config, base_config)

        config = merge_configs(merged_config, config)

    return config


def save_config(config: Dict[str, Any], save_path: Union[str, Path]) -> None:
    """Save config to YAML file

    Args:
        config: Config dictionary
        save_path: Save path (directories auto-created)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def merge_configs(
    base_config: Dict[str, Any],
    override_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Recursively merge configs, override_config overwrites base_config

    Args:
        base_config: Base config
        override_config: Override config

    Returns:
        Merged config (deep copy)
    """
    result = copy.deepcopy(base_config)

    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result
