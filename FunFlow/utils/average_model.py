# Copyright 2019 Mobvoi Inc. All Rights Reserved.
# Author: di.wu@mobvoi.com (DI WU)
"""Model averaging script for PyTorch checkpoints"""

import os
import argparse
import glob
from typing import Dict, List, Tuple, Optional

import yaml
import numpy as np
import torch


def get_args():
    parser = argparse.ArgumentParser(
        description="Average multiple model checkpoints based on validation metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dst_model", required=True, help="Output path for the averaged model"
    )
    parser.add_argument(
        "--src_path",
        required=True,
        help="Directory containing model checkpoints and YAML metric files",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cv_loss",
        help="Metric name to use for model selection (e.g., cv_loss, cv_accuracy, cv_f1). "
        "If not found, falls back to cv_accuracy",
    )
    parser.add_argument(
        "--num", default=5, type=int, help="Number of best models to average"
    )
    parser.add_argument(
        "--min_epoch",
        default=0,
        type=int,
        help="Minimum epoch to consider for averaging",
    )
    parser.add_argument(
        "--max_epoch",
        default=65536,
        type=int,
        help="Maximum epoch to consider for averaging",
    )
    parser.add_argument(
        "--maximize",
        action="store_true",
        help="Select models with highest metric values (for accuracy, F1, etc.). "
        "By default, selects lowest values (for loss metrics)",
    )

    args = parser.parse_args()
    print(f"Arguments: {args}")
    return args


def load_yaml_metrics(yaml_path: str) -> Optional[Dict]:
    """Load metrics from YAML file"""
    try:
        with open(yaml_path, "r") as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    except Exception as e:
        print(f"Warning: Failed to load {yaml_path}: {e}")
        return None


def find_yaml_files(src_path: str) -> List[str]:
    """Find all epoch YAML files"""
    yamls = glob.glob(os.path.join(src_path, "epoch_*.yaml"))

    # If not found, try *.yaml format (excluding config.yaml)
    if not yamls:
        all_yamls = glob.glob(os.path.join(src_path, "*.yaml"))
        yamls = [y for y in all_yamls if not y.endswith("config.yaml")]

    return yamls


def get_metric_from_yaml(yaml_data: Dict, metric_name: str) -> Optional[float]:
    """Extract metric value from YAML data

    Returns:
        Metric value with fallback: metric_name -> cv_loss -> cv_accuracy
    """
    if metric_name in yaml_data:
        return yaml_data[metric_name]

    # Fallback logic
    if metric_name != "cv_loss" and "cv_loss" in yaml_data:
        print(f"Warning: '{metric_name}' not found, using 'cv_loss' as fallback")
        return yaml_data["cv_loss"]

    if "cv_accuracy" in yaml_data:
        print(f"Warning: '{metric_name}' not found, using 'cv_accuracy' as fallback")
        return yaml_data["cv_accuracy"]

    return None


def collect_validation_scores(
    src_path: str, metric_name: str, min_epoch: int, max_epoch: int
) -> List[Tuple[int, float]]:
    """Collect validation scores from YAML files

    Returns:
        List of (epoch, metric_value) tuples
    """
    yaml_files = find_yaml_files(src_path)

    if not yaml_files:
        raise ValueError(f"No YAML metric files found in {src_path}")

    print(f"Found {len(yaml_files)} YAML files")

    val_scores = []
    for yaml_path in yaml_files:
        yaml_data = load_yaml_metrics(yaml_path)
        if yaml_data is None:
            continue

        epoch = yaml_data.get("epoch")
        if epoch is None:
            print(f"Warning: No epoch found in {yaml_path}, skipping")
            continue

        if not (min_epoch <= epoch <= max_epoch):
            continue

        metric_value = get_metric_from_yaml(yaml_data, metric_name)
        if metric_value is None:
            print(f"Warning: No valid metric found in {yaml_path}, skipping")
            continue

        val_scores.append((epoch, metric_value))
        print(f"Epoch {epoch}: {metric_name} = {metric_value:.6f}")

    if not val_scores:
        raise ValueError(
            f"No valid scores found for metric '{metric_name}'. "
            f"Available metrics in YAML files might be different."
        )

    return val_scores


def find_checkpoint_path(src_path: str, epoch: int) -> Optional[str]:
    """Find checkpoint file for given epoch"""
    path = os.path.join(src_path, f"epoch_{epoch}.pth")
    if os.path.exists(path):
        return path

    # Try X.pt format
    path = os.path.join(src_path, f"{epoch}.pt")
    if os.path.exists(path):
        return path

    return None


def select_best_models(
    val_scores: List[Tuple[int, float]], num_models: int, maximize: bool
) -> List[int]:
    """Select top-N model epochs

    Args:
        val_scores: List of (epoch, metric_value) tuples
        num_models: Number of models to select
        maximize: Select highest if True, lowest if False

    Returns:
        List of selected epoch numbers
    """
    scores_array = np.array(val_scores)
    if maximize:
        # For accuracy, F1, etc. - higher is better
        sort_indices = np.argsort(scores_array[:, 1])[::-1]
    else:
        # For loss - lower is better
        sort_indices = np.argsort(scores_array[:, 1])

    sorted_scores = scores_array[sort_indices]
    selected_scores = sorted_scores[:num_models]

    print(f"\n{'='*60}")
    print(f"Best {len(selected_scores)} model scores (sorted):")
    for epoch, score in selected_scores:
        print(f"  Epoch {int(epoch):3d}: {score:.6f}")
    print(f"{'='*60}\n")

    return selected_scores[:, 0].astype(int).tolist()


def load_checkpoint(checkpoint_path: str) -> Dict:
    """Load checkpoint and extract state dict"""
    print(f"Loading: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


def average_checkpoints(checkpoint_paths: List[str]) -> Dict:
    """Average multiple model checkpoints"""
    if not checkpoint_paths:
        raise ValueError("No checkpoint paths provided for averaging")

    print(f"\nAveraging {len(checkpoint_paths)} models...")

    averaged_state = None
    for path in checkpoint_paths:
        state_dict = load_checkpoint(path)

        if averaged_state is None:
            averaged_state = state_dict
        else:
            for key in averaged_state.keys():
                averaged_state[key] += state_dict[key]

    num_models = len(checkpoint_paths)
    for key in averaged_state.keys():
        if averaged_state[key] is not None:
            averaged_state[key] = torch.true_divide(averaged_state[key], num_models)

    return averaged_state


def main():
    args = get_args()

    print(f"\n{'='*60}")
    print(f"Model Averaging Configuration:")
    print(f"  Source path: {args.src_path}")
    print(f"  Metric: {args.metric}")
    print(f"  Number of models: {args.num}")
    print(f"  Epoch range: [{args.min_epoch}, {args.max_epoch}]")
    print(f"  Selection mode: {'maximize' if args.maximize else 'minimize'}")
    print(f"{'='*60}\n")

    val_scores = collect_validation_scores(
        args.src_path, args.metric, args.min_epoch, args.max_epoch
    )

    selected_epochs = select_best_models(val_scores, args.num, args.maximize)

    checkpoint_paths = []
    for epoch in selected_epochs:
        path = find_checkpoint_path(args.src_path, epoch)
        if path is None:
            print(f"Warning: Checkpoint for epoch {epoch} not found, skipping")
        else:
            checkpoint_paths.append(path)

    if not checkpoint_paths:
        raise ValueError("No valid checkpoints found for selected epochs")

    averaged_state = average_checkpoints(checkpoint_paths)

    print(f"\nSaving averaged model to: {args.dst_model}")
    torch.save(averaged_state, args.dst_model)

    print(f"\n{'='*60}")
    print(f"✓ Successfully averaged {len(checkpoint_paths)} models")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
