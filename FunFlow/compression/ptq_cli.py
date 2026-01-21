#!/usr/bin/env python3
"""PTQ quantization command-line tool."""

import argparse
import sys
from pathlib import Path

from FunFlow.compression.PTQ import ONNXPTQQuantizer
from FunFlow.compression.utils import get_onnx_input_name


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="PTQ (Post-Training Quantization) CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["dynamic", "static"],
        required=True,
        help="Quantization method: dynamic or static",
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to input ONNX model"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to output quantized model"
    )
    parser.add_argument(
        "--weight_type",
        type=str,
        default="QInt8",
        choices=["QInt8", "QUInt8"],
        help="Weight quantization type (default: QInt8)",
    )

    # 静态量化特有参数
    parser.add_argument(
        "--activation_type",
        type=str,
        default="QInt8",
        choices=["QInt8", "QUInt8"],
        help="Activation quantization type for static quantization (default: QInt8)",
    )
    parser.add_argument(
        "--calibration_data",
        type=str,
        help="Path to calibration data JSONL file (required for static quantization)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file (required for static quantization)",
    )
    parser.add_argument(
        "--num_calibration_batches",
        type=int,
        default=100,
        help="Number of calibration batches (default: 100)",
    )
    parser.add_argument(
        "--save_quant_info",
        action="store_true",
        help="Save quantization information to file",
    )
    parser.add_argument(
        "--quant_info_path",
        type=str,
        help="Path to save quantization info (without extension). "
        "If not specified, will use output_path base name",
    )
    parser.add_argument(
        "--per_channel", action="store_true", help="Use per-channel quantization"
    )
    parser.add_argument(
        "--reduce_range", action="store_true", help="Reduce quantization range"
    )
    parser.add_argument(
        "--calibrate_method",
        type=str,
        default="MinMax",
        choices=["MinMax", "Entropy"],
        help="Calibration method for static quantization (default: MinMax)",
    )
    parser.add_argument(
        "--op_types_to_quantize",
        type=str,
        nargs="+",
        help="List of op types to quantize (default: MatMul, Gemm, Gather, LSTM, GRU for dynamic quantization)",
    )

    return parser.parse_args()


def create_calibration_data_reader(
    calibration_data_path: str,
    config_path: str,
    num_calibration_batches: int,
    input_name: str,
):
    """Create calibration data reader.

    Args:
        calibration_data_path: Path to calibration JSONL file.
        config_path: Path to config file.
        num_calibration_batches: Number of calibration batches.
        input_name: Input tensor name.

    Returns:
        CalibrationDataReader instance.
    """
    import yaml
    from torch.utils.data import DataLoader
    from FunFlow.datasets.dataset import build_dataset

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_cfg = config.get("data", {})
    val_conf = data_cfg.get("val", {}).get("conf", {})

    dataset = build_dataset(
        data_file=calibration_data_path, conf=val_conf, partition=False
    )

    data_loader = DataLoader(dataset, batch_size=None, num_workers=0, pin_memory=False)

    return ONNXPTQQuantizer.create_calibration_data_reader(
        calibration_loader=data_loader,
        num_calibration_batches=num_calibration_batches,
        input_name=input_name,
    )


def main():
    """Main function."""
    args = parse_args()

    if not Path(args.model_path).exists():
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)

    quantizer = ONNXPTQQuantizer()

    quant_kwargs = {
        "per_channel": args.per_channel,
        "reduce_range": args.reduce_range,
    }

    if args.method == "dynamic" and args.op_types_to_quantize:
        quant_kwargs["op_types_to_quantize"] = args.op_types_to_quantize

    input_name = get_onnx_input_name(args.model_path)

    if args.method == "dynamic":
        print("=" * 80)
        print("ONNX Dynamic Quantization")
        print("=" * 80)

        quantized_path, quant_info = quantizer.dynamic_quantize(
            model_path=args.model_path,
            output_path=args.output_path,
            weight_type=args.weight_type,
            **quant_kwargs,
        )

    elif args.method == "static":
        if not args.calibration_data:
            print("Error: --calibration_data is required for static quantization")
            sys.exit(1)
        if not args.config:
            print("Error: --config is required for static quantization")
            sys.exit(1)

        if not Path(args.calibration_data).exists():
            print(f"Error: Calibration data not found: {args.calibration_data}")
            sys.exit(1)
        if not Path(args.config).exists():
            print(f"Error: Config file not found: {args.config}")
            sys.exit(1)

        print("=" * 80)
        print("ONNX Static Quantization")
        print("=" * 80)

        print(f"\nPreparing calibration data...")
        print(f"  Calibration data: {args.calibration_data}")
        print(f"  Number of batches: {args.num_calibration_batches}")

        try:
            calibration_reader = create_calibration_data_reader(
                calibration_data_path=args.calibration_data,
                config_path=args.config,
                num_calibration_batches=args.num_calibration_batches,
                input_name=input_name,
            )
        except Exception as e:
            print(f"Error: Failed to create calibration data reader: {e}")
            sys.exit(1)

        quant_kwargs["calibrate_method"] = args.calibrate_method

        quantized_path, quant_info = quantizer.static_quantize(
            model_path=args.model_path,
            output_path=args.output_path,
            calibration_data_reader=calibration_reader,
            weight_type=args.weight_type,
            activation_type=args.activation_type,
            **quant_kwargs,
        )

    else:
        print(f"Error: Unknown quantization method: {args.method}")
        sys.exit(1)

    if args.save_quant_info:
        if args.quant_info_path:
            info_path = args.quant_info_path
        else:
            info_path = str(Path(args.output_path).with_suffix(""))
            info_path = f"{info_path}_quant_info"

        print("\n" + "=" * 80)
        print("Saving Quantization Information")
        print("=" * 80)

        ONNXPTQQuantizer.save_quantization_info(
            quant_info=quant_info,
            output_path=info_path,
            save_json=True,
            save_txt=True,
        )

    print("\n" + "=" * 80)
    print("Quantization Completed Successfully!")
    print("=" * 80)
    print(f"Quantized model: {quantized_path}")

    if args.save_quant_info:
        print(f"Quantization info: {info_path}.{{json,txt}}")


if __name__ == "__main__":
    main()
