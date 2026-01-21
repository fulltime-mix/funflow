#!/usr/bin/env python
"""Model loaders for different formats"""

import logging

import torch
import onnxruntime as ort

from FunFlow.registry import MODEL_LOADERS

logger = logging.getLogger(__name__)


@MODEL_LOADERS.register("pytorch")
def load_pytorch_model(
    config: dict,
    checkpoint_path: str,
    device: str = "cuda",
    num_threads: int = None,
) -> torch.nn.Module:
    """Load PyTorch model

    Args:
        config: Configuration dict
        checkpoint_path: Checkpoint path
        device: Target device
        num_threads: PyTorch thread count (None for auto, >=1 for specific)

    Returns:
        Loaded model
    """
    from FunFlow.registry import MODELS

    if num_threads is not None and num_threads >= 1:
        torch.set_num_threads(num_threads)
        logger.info(f"PyTorch threads set to: {num_threads}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = config["model"]

    model = MODELS.build(model_config)

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint.get("state_dict", checkpoint.get("model", checkpoint)),
    )

    incompatible_keys = model.load_state_dict(state_dict, strict=False)

    if incompatible_keys.missing_keys:
        logger.warning(f"Missing keys: {incompatible_keys.missing_keys}")

    if incompatible_keys.unexpected_keys:
        logger.warning(f"Unexpected keys: {incompatible_keys.unexpected_keys}")

    return model.to(device).eval()


@MODEL_LOADERS.register("onnx")
def load_onnx_model(
    config: dict,
    checkpoint_path: str,
    device: str = "cpu",
    num_threads: int = 1,
):
    """Load ONNX model

    Args:
        config: Configuration dict (unused, kept for interface compatibility)
        checkpoint_path: ONNX model path
        device: Target device ('cuda' or 'cpu')
        num_threads: Intra-op parallelism (default 1 to avoid thread pool conflicts)

    Returns:
        ONNX Runtime InferenceSession
    """

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess_options.intra_op_num_threads = num_threads
    sess_options.inter_op_num_threads = 1
    logger.info(f"ONNX Runtime threads: intra_op={num_threads}, inter_op={1}")

    providers = ["CPUExecutionProvider"]
    if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        logger.info("Using CUDA execution provider for ONNX")
    else:
        logger.info("Using CPU execution provider for ONNX")

    session = ort.InferenceSession(
        checkpoint_path, sess_options=sess_options, providers=providers
    )

    logger.info(f"ONNX model loaded from: {checkpoint_path}")
    logger.info(
        f"Input: {session.get_inputs()[0].name}, shape: {session.get_inputs()[0].shape}"
    )
    logger.info(
        f"Output: {session.get_outputs()[0].name}, shape: {session.get_outputs()[0].shape}"
    )

    return session


@MODEL_LOADERS.register("qat")
def load_qat_model(
    config: dict,
    checkpoint_path: str,
    device: str = "cpu",
    num_threads: int = None,
) -> torch.nn.Module:
    """Load Quantization-Aware Training (QAT) model

    Args:
        config: Configuration dict
        checkpoint_path: Checkpoint path
        device: Target device
        num_threads: PyTorch thread count

    Returns:
        Loaded quantized model
    """
    import FunFlow.models  # noqa: F401
    from FunFlow.registry import MODELS
    from FunFlow.compression.QAT import QATQuantizer

    if num_threads is not None:
        torch.set_num_threads(num_threads)
        logger.info(f"PyTorch threads set to: {num_threads}")

    model = QATQuantizer.load_converted_model(
        model_path=checkpoint_path,
        config=config,
    )

    return model.eval()
