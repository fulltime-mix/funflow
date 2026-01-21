#!/usr/bin/env python
"""ONNX exporter for models with dict input/output."""

import os
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from .base_exporter import BaseExporter
from FunFlow.registry import EXPORTERS

logger = logging.getLogger(__name__)


@EXPORTERS.register("onnx")
class ONNXExporter(BaseExporter):
    """ONNX exporter with configurable parameters."""

    SUPPORTED_PARAMS = [
        "opset_version",
        "input_names",
        "output_names",
        "output_keys",
        "dynamic_axes",
        "simplify",
        "rtol",
        "atol",
    ]

    def __init__(
        self,
        device: str = "cpu",
        opset_version: int = 13,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        output_keys: List[str] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        simplify: bool = False,
        rtol: float = 1e-3,
        atol: float = 1e-5,
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)

        self.opset_version = opset_version
        self.input_names = input_names
        self.output_names = output_names
        self.output_keys = output_keys or ["logits"]
        self.dynamic_axes = dynamic_axes
        self.simplify = simplify
        self.rtol = rtol
        self.atol = atol
        self.export_kwargs = kwargs

    def export(
        self,
        model: nn.Module,
        dummy_input: Union[torch.Tensor, Dict[str, torch.Tensor]],
        output_path: str,
        verify: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Export model to ONNX format.

        Args:
            model: Model to export
            dummy_input: Example input
            output_path: Output file path
            verify: Whether to verify exported model
            **kwargs: Additional parameters

        Returns:
            Export result dict
        """
        result = {
            "output_path": "",
            "success": False,
            "verified": False,
            "file_size_mb": 0.0,
            "export_time_ms": 0.0,
            "error": None,
        }

        try:
            start_time = time.time()

            model = model.to(self.device).eval()
            output_path = (
                output_path if output_path.endswith(".onnx") else output_path + ".onnx"
            )
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            dummy_input = self._to_device(dummy_input)

            input_names = kwargs.get("input_names", self.input_names)
            output_names = kwargs.get("output_names", self.output_names)
            output_keys = kwargs.get("output_keys", self.output_keys)
            dynamic_axes = kwargs.get("dynamic_axes", self.dynamic_axes)
            simplify = kwargs.get("simplify", self.simplify)

            if input_names is None:
                input_names = (
                    list(dummy_input.keys())
                    if isinstance(dummy_input, dict)
                    else ["input"]
                )
            if output_names is None:
                output_names = output_keys

            if dynamic_axes is None:
                dynamic_axes = {
                    n: {0: "batch_size"} for n in input_names + output_names
                }

            export_model = self._wrap_model(model, dummy_input, output_keys)
            export_input = (
                tuple(dummy_input.values())
                if isinstance(dummy_input, dict)
                else dummy_input
            )

            torch.onnx.export(
                export_model,
                export_input,
                output_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=self.opset_version,
                do_constant_folding=True,
                **self.export_kwargs,
            )

            if simplify:
                self._simplify(output_path)

            result["output_path"] = output_path
            result["success"] = True
            result["file_size_mb"] = os.path.getsize(output_path) / (1024 * 1024)
            result["export_time_ms"] = (time.time() - start_time) * 1000

            if verify:
                result["verified"] = self.verify(
                    output_path, model, dummy_input, **kwargs
                )

            logger.info(
                f"Export completed: {output_path} ({result['file_size_mb']:.2f} MB)"
            )

        except Exception as e:
            logger.error(f"Export failed: {e}", exc_info=True)
            result["error"] = str(e)

        return result

    def verify(
        self,
        output_path: str,
        model: nn.Module,
        dummy_input: Union[torch.Tensor, Dict[str, torch.Tensor]],
        **kwargs,
    ) -> bool:
        """Verify exported ONNX model.

        Args:
            output_path: Path to exported model
            model: Original PyTorch model
            dummy_input: Example input
            **kwargs: Additional parameters

        Returns:
            Whether verification passed
        """
        try:
            import onnx
            import onnxruntime as ort
            import numpy as np

            onnx.checker.check_model(onnx.load(output_path))
            logger.info("  ✓ ONNX structure check passed")

            session = ort.InferenceSession(output_path)
            logger.info("  ✓ ONNX Runtime load passed")

            output_keys = kwargs.get("output_keys", self.output_keys)
            rtol = kwargs.get("rtol", self.rtol)
            atol = kwargs.get("atol", self.atol)

            model.eval()
            with torch.no_grad():
                pt_output = model(dummy_input)
                pt_output = (
                    [pt_output[k].cpu().numpy() for k in output_keys]
                    if isinstance(pt_output, dict)
                    else [pt_output.cpu().numpy()]
                )

            if isinstance(dummy_input, dict):
                onnx_input = {k: v.cpu().numpy() for k, v in dummy_input.items()}
            else:
                onnx_input = {session.get_inputs()[0].name: dummy_input.cpu().numpy()}
            onnx_output = session.run(None, onnx_input)

            for i, (pt, onnx) in enumerate(zip(pt_output, onnx_output)):
                if not np.allclose(pt, onnx, rtol=rtol, atol=atol):
                    logger.warning(
                        f"  ✗ Output {i} mismatch, max diff: {np.abs(pt - onnx).max():.6f}"
                    )
                    return False
                logger.info(f"  ✓ Output {i} match")

            return True

        except ImportError:
            logger.warning("  ℹ onnxruntime not installed, skipping verification")
            return True
        except Exception as e:
            logger.error(f"  ✗ Verification failed: {e}")
            return False

    def _wrap_model(
        self,
        model: nn.Module,
        dummy_input: Union[torch.Tensor, Dict[str, torch.Tensor]],
        output_keys: List[str],
    ) -> nn.Module:
        """Wrap model to convert dict IO to tuple IO."""
        input_keys = list(dummy_input.keys()) if isinstance(dummy_input, dict) else None

        class Wrapper(nn.Module):
            def __init__(self, model, input_keys, output_keys):
                super().__init__()
                self.model = model
                self.input_keys = input_keys
                self.output_keys = output_keys

            def forward(self, *args):
                inputs = (
                    dict(zip(self.input_keys, args)) if self.input_keys else args[0]
                )
                outputs = self.model(inputs)
                if isinstance(outputs, dict):
                    return (
                        outputs[self.output_keys[0]]
                        if len(self.output_keys) == 1
                        else tuple(outputs[k] for k in self.output_keys)
                    )
                return outputs

        return Wrapper(model, input_keys, output_keys)

    def _simplify(self, path: str) -> None:
        """Simplify ONNX model."""
        try:
            import onnx
            from onnxsim import simplify

            model, ok = simplify(onnx.load(path))
            if ok:
                onnx.save(model, path)
                logger.info("  ✓ Model simplified")
        except Exception as e:
            logger.warning(f"  ℹ Simplification skipped: {e}")


__all__ = ["ONNXExporter"]
