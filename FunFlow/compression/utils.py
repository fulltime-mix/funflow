"""Model compression utility functions."""

import os
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import onnx


def get_onnx_input_name(model_path: str) -> str:
    """Auto-detect input name from ONNX model.

    Args:
        model_path: ONNX model path.

    Returns:
        Input tensor name.
    """
    try:
        model = onnx.load(model_path)
        if len(model.graph.input) > 0:
            input_name = model.graph.input[0].name
            print(f"Auto-detected input name: {input_name}")
            return input_name
        else:
            print("Warning: No input found in model, using default 'input'")
            return "input"
    except Exception as e:
        print(f"Warning: Failed to load ONNX model to detect input name: {e}")
        print("Using default input name: 'input'")
        return "input"


def get_model_sparsity(model: nn.Module) -> float:
    """Calculate model global sparsity.

    Args:
        model: PyTorch model.

    Returns:
        Sparsity (0-1).
    """
    total_params = 0
    zero_params = 0

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
            if hasattr(module, "weight"):
                if hasattr(module, "weight_mask"):
                    weight = module.weight_mask * module.weight_orig
                else:
                    weight = module.weight

                total_params += weight.numel()
                zero_params += (weight == 0).sum().item()

    return zero_params / total_params if total_params > 0 else 0.0


def get_layer_sparsity(module: nn.Module) -> float:
    """Calculate layer sparsity.

    Args:
        module: PyTorch module.

    Returns:
        Sparsity.
    """
    if not hasattr(module, "weight"):
        return 0.0

    weight = module.weight
    total = weight.numel()
    zeros = (weight == 0).sum().item()

    return zeros / total if total > 0 else 0.0


def get_model_size(model: nn.Module, unit: str = "MB") -> float:
    """Get model size.

    Args:
        model: PyTorch model.
        unit: Unit ('B', 'KB', 'MB', 'GB').

    Returns:
        Model size.
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size

    units = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
    return total_size / units.get(unit, 1024**2)


def get_model_file_size(model_path: str, unit: str = "MB") -> float:
    """Get model file size.

    Args:
        model_path: Model file path.
        unit: Unit ('B', 'KB', 'MB', 'GB').

    Returns:
        File size.
    """
    if not os.path.exists(model_path):
        return 0.0

    file_size = os.path.getsize(model_path)
    units = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
    return file_size / units.get(unit, 1024**2)


def compare_model_sizes(
    original_model: nn.Module,
    compressed_model: nn.Module,
    unit: str = "MB",
) -> Dict[str, Any]:
    """Compare original and compressed model sizes.

    Args:
        original_model: Original model.
        compressed_model: Compressed model.
        unit: Unit ('B', 'KB', 'MB', 'GB').

    Returns:
        Dict containing comparison results.
    """
    original_size = get_model_size(original_model, unit)
    compressed_size = get_model_size(compressed_model, unit)

    original_params = sum(p.numel() for p in original_model.parameters())
    compressed_params = sum(p.numel() for p in compressed_model.parameters())

    compressed_nonzero = sum(
        (p != 0).sum().item() for p in compressed_model.parameters()
    )

    return {
        "original_size": original_size,
        "compressed_size": compressed_size,
        "size_reduction": (original_size - compressed_size) / original_size * 100,
        "original_params": original_params,
        "compressed_params": compressed_params,
        "compressed_nonzero_params": compressed_nonzero,
        "param_reduction": (original_params - compressed_nonzero)
        / original_params
        * 100,
        "compression_ratio": (
            original_size / compressed_size if compressed_size > 0 else float("inf")
        ),
        "unit": unit,
    }


def count_zero_parameters(model: nn.Module) -> Dict[str, int]:
    """Count zero parameters in model.

    Args:
        model: PyTorch model.

    Returns:
        Statistics dict.
    """
    total_params = 0
    zero_params = 0
    layer_stats = {}

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
            if hasattr(module, "weight"):
                if hasattr(module, "weight_mask"):
                    weight = module.weight_mask * module.weight_orig
                else:
                    weight = module.weight

                total = weight.numel()
                zeros = (weight == 0).sum().item()

                total_params += total
                zero_params += zeros

                if zeros > 0:
                    layer_stats[name] = {
                        "total": total,
                        "zeros": zeros,
                        "sparsity": zeros / total,
                    }

    return {
        "total_params": total_params,
        "zero_params": zero_params,
        "nonzero_params": total_params - zero_params,
        "global_sparsity": zero_params / total_params if total_params > 0 else 0,
        "layer_stats": layer_stats,
    }


def visualize_sparsity(
    model: nn.Module,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Visualize layer-wise sparsity.

    Args:
        model: PyTorch model.
        save_path: Path to save figure.
        show: Whether to display figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Cannot visualize sparsity.")
        return

    layer_names = []
    sparsities = []

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
            sparsity = get_layer_sparsity(module)
            layer_names.append(name)
            sparsities.append(sparsity * 100)

    if not layer_names:
        print("No prunable layers found.")
        return

    fig, ax = plt.subplots(figsize=(12, max(6, len(layer_names) * 0.3)))

    y_pos = range(len(layer_names))
    ax.barh(y_pos, sparsities, align="center", color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(layer_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Sparsity (%)")
    ax.set_title("Layer-wise Sparsity Distribution")
    ax.axvline(
        x=sum(sparsities) / len(sparsities),
        color="red",
        linestyle="--",
        label=f"Avg: {sum(sparsities)/len(sparsities):.1f}%",
    )
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Sparsity visualization saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def print_model_summary(model: nn.Module, title: str = "Model Summary"):
    """Print model summary information.

    Args:
        model: PyTorch model.
        title: Summary title.
    """
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable:        {total_params - trainable_params:,}")

    size_mb = get_model_size(model, "MB")
    print(f"Model size:           {size_mb:.2f} MB")

    sparsity = get_model_sparsity(model)
    print(f"Global sparsity:      {sparsity*100:.2f}%")

    conv_layers = sum(
        1 for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Conv1d))
    )
    linear_layers = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    bn_layers = sum(
        1 for m in model.modules() if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d))
    )

    print(f"Conv layers:          {conv_layers}")
    print(f"Linear layers:        {linear_layers}")
    print(f"BatchNorm layers:     {bn_layers}")

    print(f"{'='*60}\n")


def sparse_to_dense(model: nn.Module) -> nn.Module:
    """Convert sparse model to dense (remove masks).

    Used for exporting to formats that don't support sparsity like ONNX.

    Args:
        model: Sparse model.

    Returns:
        Dense model (weights stay sparse but no mask).
    """
    import torch.nn.utils.prune as prune

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
            try:
                prune.remove(module, "weight")
            except ValueError:
                pass

    return model


def measure_inference_time(
    model: nn.Module,
    input_shape: tuple,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    device: str = "cuda",
) -> Dict[str, float]:
    """Measure model inference time.

    Args:
        model: Model.
        input_shape: Input shape (without batch dimension).
        num_iterations: Number of measurement iterations.
        warmup_iterations: Number of warmup iterations.
        device: Device ('cuda' or 'cpu').

    Returns:
        Inference time statistics.
    """
    import time

    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(1, *input_shape).to(device)

    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(dummy_input)

    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            _ = model(dummy_input)

            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()

            times.append((end - start) * 1000)

    return {
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": (sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times))
        ** 0.5,
        "fps": 1000 / (sum(times) / len(times)),
    }


def export_pruned_model(
    model: nn.Module,
    save_path: str,
    remove_masks: bool = True,
) -> str:
    """Export pruned model.

    Args:
        model: Pruned model.
        save_path: Save path.
        remove_masks: Whether to remove pruning masks.

    Returns:
        Save path.
    """
    if remove_masks:
        model = sparse_to_dense(model)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "sparsity": get_model_sparsity(model),
            "model_size_mb": get_model_size(model, "MB"),
        },
        save_path,
    )

    print(f"Pruned model saved to: {save_path}")
    print(f"  Sparsity: {get_model_sparsity(model)*100:.2f}%")
    print(f"  Size: {get_model_size(model, 'MB'):.2f} MB")

    return save_path
