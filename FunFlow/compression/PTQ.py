"""Model quantization supporting Post-Training Quantization (PTQ)."""

from typing import Dict, Any, Optional, List, Union, Callable, Tuple
from pathlib import Path
from abc import ABC, abstractmethod
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class PTQQuantizerBase(ABC):
    """PTQ quantizer base class defining quantization interface.

    Subclasses must implement:
        - dynamic_quantize: Runtime quantization
        - static_quantize: Pre-calibrated quantization
    """

    @abstractmethod
    def dynamic_quantize(
        self, model_path: str, output_path: str, **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """Dynamic quantization.

        Args:
            model_path: Input model path.
            output_path: Output model path.
            **kwargs: Backend-specific parameters.

        Returns:
            Tuple of (quantized model path, quantization info dict).
        """
        pass

    @abstractmethod
    def static_quantize(
        self,
        model_path: str,
        output_path: str,
        calibration_data_reader: Any,
        **kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        """Static quantization with calibration data.

        Args:
            model_path: Input model path.
            output_path: Output model path.
            calibration_data_reader: Calibration data reader.
            **kwargs: Backend-specific parameters.

        Returns:
            Tuple of (quantized model path, quantization info dict).
        """
        pass

    @staticmethod
    def get_model_size(model_path: str) -> float:
        """Get model file size in MB.

        Args:
            model_path: Model file path.

        Returns:
            Model size in MB.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        size_bytes = os.path.getsize(model_path)
        size_mb = size_bytes / (1024 * 1024)
        return size_mb

    @staticmethod
    def compare_model_sizes(
        original_path: str, quantized_path: str
    ) -> Dict[str, float]:
        """Compare original and quantized model sizes.

        Args:
            original_path: Original model path.
            quantized_path: Quantized model path.

        Returns:
            Dict containing size info and compression ratio.
        """
        original_size = PTQQuantizerBase.get_model_size(original_path)
        quantized_size = PTQQuantizerBase.get_model_size(quantized_path)
        compression_ratio = original_size / quantized_size if quantized_size > 0 else 0

        return {
            "original_size_mb": original_size,
            "quantized_size_mb": quantized_size,
            "compression_ratio": compression_ratio,
            "size_reduction_percent": (
                (1 - quantized_size / original_size) * 100 if original_size > 0 else 0
            ),
        }

    @staticmethod
    def save_quantization_info(
        quant_info: Dict[str, Any],
        output_path: str,
        save_json: bool = True,
        save_txt: bool = True,
    ):
        """Save quantization info to file.

        Args:
            quant_info: Quantization info dict.
            output_path: Output file path (without extension).
            save_json: Save as JSON format.
            save_txt: Save as readable text format.
        """
        if save_json:
            json_path = f"{output_path}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(quant_info, f, indent=2, ensure_ascii=False)
            print(f"  Quantization info (JSON) saved to: {json_path}")

        if save_txt:
            txt_path = f"{output_path}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write("QUANTIZATION INFORMATION REPORT\n")
                f.write("=" * 80 + "\n\n")

                f.write("Basic Information:\n")
                f.write("-" * 80 + "\n")
                for key in [
                    "quantization_method",
                    "weight_type",
                    "activation_type",
                    "original_model",
                    "quantized_model",
                ]:
                    if key in quant_info:
                        f.write(f"  {key}: {quant_info[key]}\n")
                f.write("\n")

                if "model_size" in quant_info:
                    f.write("Model Size:\n")
                    f.write("-" * 80 + "\n")
                    size_info = quant_info["model_size"]
                    f.write(
                        f"  Original Size: {size_info['original_size_mb']:.2f} MB\n"
                    )
                    f.write(
                        f"  Quantized Size: {size_info['quantized_size_mb']:.2f} MB\n"
                    )
                    f.write(
                        f"  Compression Ratio: {size_info['compression_ratio']:.2f}x\n"
                    )
                    f.write(
                        f"  Size Reduction: {size_info['size_reduction_percent']:.1f}%\n"
                    )
                    f.write("\n")

                if "quantization_summary" in quant_info:
                    summary = quant_info["quantization_summary"]
                    f.write("Quantization Summary:\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"  Total Nodes: {summary.get('total_nodes', 0)}\n")
                    f.write(f"  Quantized Nodes: {summary.get('quantized_nodes', 0)}\n")
                    f.write(f"  FP32 Nodes: {summary.get('fp32_nodes', 0)}\n")
                    f.write(f"  Fused Nodes: {summary.get('fused_nodes', 0)}\n")
                    f.write("\n")

                if "quantized_nodes" in quant_info and quant_info["quantized_nodes"]:
                    f.write("Quantized Nodes:\n")
                    f.write("-" * 80 + "\n")
                    for node in quant_info["quantized_nodes"]:
                        f.write(f"  - {node}\n")
                    f.write("\n")

                if "fp32_nodes" in quant_info and quant_info["fp32_nodes"]:
                    f.write("FP32 (Not Quantized) Nodes:\n")
                    f.write("-" * 80 + "\n")
                    for node in quant_info["fp32_nodes"]:
                        f.write(f"  - {node}\n")
                    f.write("\n")

                if "fused_nodes" in quant_info and quant_info["fused_nodes"]:
                    f.write("Fused Nodes:\n")
                    f.write("-" * 80 + "\n")
                    for fusion in quant_info["fused_nodes"]:
                        if isinstance(fusion, dict):
                            f.write(f"  - {fusion.get('name', 'Unknown')}:\n")
                            f.write(
                                f"    Layers: {', '.join(fusion.get('layers', []))}\n"
                            )
                        else:
                            f.write(f"  - {fusion}\n")
                    f.write("\n")

                f.write("=" * 80 + "\n")

            print(f"  Quantization info (TXT) saved to: {txt_path}")


class ONNXPTQQuantizer(PTQQuantizerBase):
    """ONNX Runtime PTQ quantizer supporting dynamic and static quantization.

    Dynamic quantization:
        - Quantizes weights and activations at runtime
        - No calibration data needed
        - Good for RNN/LSTM/Transformer models

    Static quantization:
        - Pre-calibrates quantization parameters
        - Requires calibration dataset
        - Better for CNN models
    """

    def __init__(self):
        """Initialize ONNX PTQ quantizer."""
        self._check_dependencies()

    def _check_dependencies(self):
        """Check if dependencies are installed."""
        try:
            import onnxruntime
            from onnxruntime.quantization import quantize_dynamic, quantize_static
        except ImportError as e:
            raise ImportError(
                "ONNX Runtime is not installed. Please install it:\n"
                "  pip install onnxruntime\n"
                f"Original error: {e}"
            )

    def dynamic_quantize(
        self,
        model_path: str,
        output_path: str,
        weight_type: str = "QInt8",
        op_types_to_quantize: Optional[List[str]] = None,
        **kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        """ONNX dynamic quantization.

        Args:
            model_path: Input ONNX model path.
            output_path: Output quantized model path.
            weight_type: Weight quantization type ('QInt8' or 'QUInt8').
            op_types_to_quantize: Op types to quantize (e.g., ['MatMul', 'Gemm']).
            **kwargs: Additional ONNX Runtime quantization parameters.

        Returns:
            Tuple of (quantized model path, quantization info dict).
        """
        from onnxruntime.quantization import quantize_dynamic, QuantType

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if weight_type not in ["QInt8", "QUInt8"]:
            raise ValueError(
                f"Invalid weight_type: {weight_type}. Must be 'QInt8' or 'QUInt8'"
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        weight_type_enum = (
            QuantType.QInt8 if weight_type == "QInt8" else QuantType.QUInt8
        )

        if op_types_to_quantize is None:
            op_types_to_quantize = ["MatMul", "Gemm", "Gather", "LSTM", "GRU"]
            print(
                f"  Using safe op types for CPU compatibility: {op_types_to_quantize}"
            )

        print(f"[ONNX PTQ] Applying dynamic quantization...")
        print(f"  Input model: {model_path}")
        print(f"  Output model: {output_path}")
        print(f"  Weight type: {weight_type}")
        print(f"  Op types to quantize: {op_types_to_quantize}")

        quantize_dynamic(
            model_input=model_path,
            model_output=output_path,
            weight_type=weight_type_enum,
            op_types_to_quantize=op_types_to_quantize,
            **kwargs,
        )

        quant_info = self._analyze_quantized_model(
            original_model_path=model_path,
            quantized_model_path=output_path,
            quantization_method="dynamic",
            weight_type=weight_type,
            activation_type="dynamic",
        )

        size_info = self.compare_model_sizes(model_path, output_path)
        quant_info["model_size"] = size_info

        print(f"  Original size: {size_info['original_size_mb']:.2f} MB")
        print(f"  Quantized size: {size_info['quantized_size_mb']:.2f} MB")
        print(f"  Compression ratio: {size_info['compression_ratio']:.2f}x")
        print(f"  Size reduction: {size_info['size_reduction_percent']:.1f}%")
        print(f"[ONNX PTQ] Dynamic quantization completed!")

        return output_path, quant_info

    def static_quantize(
        self,
        model_path: str,
        output_path: str,
        calibration_data_reader: Any,
        weight_type: str = "QInt8",
        activation_type: str = "QUInt8",
        **kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        """ONNX static quantization.

        Args:
            model_path: Input ONNX model path.
            output_path: Output quantized model path.
            calibration_data_reader: Calibration data reader.
            weight_type: Weight quantization type ('QInt8' or 'QUInt8').
            activation_type: Activation quantization type ('QInt8' or 'QUInt8').
            **kwargs: Additional ONNX Runtime quantization parameters.

        Returns:
            Tuple of (quantized model path, quantization info dict).
        """
        from onnxruntime.quantization import (
            quantize_static,
            QuantType,
            CalibrationMethod,
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if weight_type not in ["QInt8", "QUInt8"]:
            raise ValueError(
                f"Invalid weight_type: {weight_type}. Must be 'QInt8' or 'QUInt8'"
            )

        if activation_type not in ["QInt8", "QUInt8"]:
            raise ValueError(
                f"Invalid activation_type: {activation_type}. Must be 'QInt8' or 'QUInt8'"
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        weight_type_enum = (
            QuantType.QInt8 if weight_type == "QInt8" else QuantType.QUInt8
        )
        activation_type_enum = (
            QuantType.QInt8 if activation_type == "QInt8" else QuantType.QUInt8
        )

        if "calibrate_method" in kwargs and isinstance(kwargs["calibrate_method"], str):
            calibrate_method_str = kwargs["calibrate_method"]
            calibrate_method_map = {
                "MinMax": CalibrationMethod.MinMax,
                "Entropy": CalibrationMethod.Entropy,
                "Percentile": CalibrationMethod.Percentile,
                "Distribution": CalibrationMethod.Distribution,
            }
            if calibrate_method_str in calibrate_method_map:
                kwargs["calibrate_method"] = calibrate_method_map[calibrate_method_str]
            else:
                print(
                    f"Warning: Unknown calibrate_method '{calibrate_method_str}', using MinMax"
                )
                kwargs["calibrate_method"] = CalibrationMethod.MinMax

        print(f"[ONNX PTQ] Applying static quantization...")
        print(f"  Input model: {model_path}")
        print(f"  Output model: {output_path}")
        print(f"  Weight type: {weight_type}")
        print(f"  Activation type: {activation_type}")
        if "calibrate_method" in kwargs:
            print(f"  Calibration method: {kwargs['calibrate_method']}")

        quantize_static(
            model_input=model_path,
            model_output=output_path,
            calibration_data_reader=calibration_data_reader,
            weight_type=weight_type_enum,
            activation_type=activation_type_enum,
            **kwargs,
        )

        quant_info = self._analyze_quantized_model(
            original_model_path=model_path,
            quantized_model_path=output_path,
            quantization_method="static",
            weight_type=weight_type,
            activation_type=activation_type,
        )

        size_info = self.compare_model_sizes(model_path, output_path)
        quant_info["model_size"] = size_info

        print(f"  Original size: {size_info['original_size_mb']:.2f} MB")
        print(f"  Quantized size: {size_info['quantized_size_mb']:.2f} MB")
        print(f"  Compression ratio: {size_info['compression_ratio']:.2f}x")
        print(f"  Size reduction: {size_info['size_reduction_percent']:.1f}%")
        print(f"[ONNX PTQ] Static quantization completed!")

        return output_path, quant_info

    def _analyze_quantized_model(
        self,
        original_model_path: str,
        quantized_model_path: str,
        quantization_method: str,
        weight_type: str,
        activation_type: str = None,
    ) -> Dict[str, Any]:
        """Analyze quantized ONNX model.

        Args:
            original_model_path: Original model path.
            quantized_model_path: Quantized model path.
            quantization_method: Quantization method ('dynamic' or 'static').
            weight_type: Weight quantization type.
            activation_type: Activation quantization type.

        Returns:
            Dict containing quantization details.
        """
        try:
            import onnx
        except ImportError:
            print(
                "Warning: onnx package not installed. Skipping detailed model analysis."
            )
            return {
                "quantization_method": quantization_method,
                "weight_type": weight_type,
                "activation_type": activation_type,
                "original_model": original_model_path,
                "quantized_model": quantized_model_path,
            }

        try:
            original_model = onnx.load(original_model_path)
            quantized_model = onnx.load(quantized_model_path)
        except Exception as e:
            print(f"Warning: Failed to load ONNX models for analysis: {e}")
            return {
                "quantization_method": quantization_method,
                "weight_type": weight_type,
                "activation_type": activation_type,
                "original_model": original_model_path,
                "quantized_model": quantized_model_path,
            }

        original_nodes = {node.op_type for node in original_model.graph.node}
        quantized_nodes = []
        fp32_nodes = []
        fused_nodes = []

        quantized_op_types = {
            "QLinearConv",
            "QLinearMatMul",
            "QLinearAdd",
            "QLinearMul",
            "QuantizeLinear",
            "DequantizeLinear",
            "QAttention",
            "QLinearAveragePool",
            "QLinearGlobalAveragePool",
            "QLinearSigmoid",
            "QLinearLeakyRelu",
            "QLinearConcat",
        }

        fused_op_types = {
            "FusedConv",
            "FusedMatMul",
            "FusedGemm",
            "QLinearConv",
            "QLinearMatMul",
        }

        for node in quantized_model.graph.node:
            node_info = f"{node.name} ({node.op_type})"

            if node.op_type in quantized_op_types:
                quantized_nodes.append(node_info)
            elif (
                node.op_type in ["Conv", "MatMul", "Gemm", "Add", "Mul"]
                and node.op_type in original_nodes
            ):
                fp32_nodes.append(node_info)

            if node.op_type in fused_op_types:
                fused_nodes.append(
                    {
                        "name": node.name,
                        "type": node.op_type,
                        "inputs": list(node.input),
                        "outputs": list(node.output),
                    }
                )

        quant_info = {
            "quantization_method": quantization_method,
            "weight_type": weight_type,
            "activation_type": activation_type,
            "original_model": original_model_path,
            "quantized_model": quantized_model_path,
            "quantization_summary": {
                "total_nodes": len(quantized_model.graph.node),
                "quantized_nodes": len(quantized_nodes),
                "fp32_nodes": len(fp32_nodes),
                "fused_nodes": len(fused_nodes),
            },
            "quantized_nodes": quantized_nodes,
            "fp32_nodes": fp32_nodes,
            "fused_nodes": fused_nodes,
        }

        return quant_info

    @staticmethod
    def create_calibration_data_reader(
        calibration_loader: DataLoader,
        num_calibration_batches: int = 100,
        input_name: str = "input",
    ):
        """Create ONNX Runtime calibration data reader.

        Args:
            calibration_loader: PyTorch DataLoader for calibration data.
            num_calibration_batches: Number of calibration batches.
            input_name: ONNX model input tensor name.

        Returns:
            CalibrationDataReader instance.
        """
        from onnxruntime.quantization import CalibrationDataReader

        class CustomCalibrationDataReader(CalibrationDataReader):
            """Custom calibration data reader."""

            def __init__(self, data_loader, num_batches, input_name):
                self.data_loader = data_loader
                self.num_batches = num_batches
                self.input_name = input_name
                self.iterator = None
                self.count = 0

            def get_next(self):
                """Get next calibration batch."""
                if self.iterator is None:
                    self.iterator = iter(self.data_loader)

                if self.count >= self.num_batches:
                    return None

                try:
                    batch = next(self.iterator)
                    self.count += 1

                    inputs = self._extract_inputs(batch)

                    if isinstance(inputs, torch.Tensor):
                        inputs = inputs.cpu().numpy()

                    return {self.input_name: inputs}
                except StopIteration:
                    return None

            def _extract_inputs(self, batch):
                """Extract input data from batch."""
                if isinstance(batch, dict):
                    for key in ["feats", "images", "input", "inputs"]:
                        if key in batch:
                            return batch[key]
                    raise ValueError(
                        f"Cannot find input data in batch. Available keys: {list(batch.keys())}"
                    )
                elif isinstance(batch, (list, tuple)):
                    return batch[0]
                else:
                    return batch

        return CustomCalibrationDataReader(
            calibration_loader, num_calibration_batches, input_name
        )
