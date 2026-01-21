"""Model export command line interface."""

import argparse
import logging

from FunFlow.export import (
    prepare_dummy_input,
    get_output_path,
    print_result,
    load_export_config,
    parse_dynamic_axes,
    parse_list,
    parse_input_shape,
)
from FunFlow.registry import EXPORTERS

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Model Export CLI")

    parser.add_argument(
        "--checkpoint", "-c", 
        required=True, 
        help="Model checkpoint path"
    )
    parser.add_argument(
        "--output", "-o", 
        help="Output path (default: <checkpoint>.<format>)"
    )
    parser.add_argument(
        "--config", 
        help="Model config path"
    )
    parser.add_argument(
        "--export_config", 
        help="Export config file (YAML/JSON)"
    )
    parser.add_argument(
        "--format",
        "-f",
        default="onnx",
        help=f"Export format (default: onnx). Available: {list(EXPORTERS.keys())}",
    )

    parser.add_argument(
        "--input_shape", 
        help='Input shape: "1,16,1000" or "data:1,16,1000;mask:1,1000"'
    )
    parser.add_argument(
        "--device", 
        default="cpu", 
        help="Device"
    )
    parser.add_argument(
        "--model_loader", 
        default="pytorch", 
        help="Model loader type"
    )

    parser.add_argument(
        "--opset_version", 
        type=int, 
        default=13, 
        help="ONNX opset version"
    )
    parser.add_argument(
        "--input_names", 
        help="Input names (comma-separated)"
    )
    parser.add_argument(
        "--output_names", 
        help="Output names (comma-separated)"
    )
    parser.add_argument(
        "--output_keys", 
        help="Output keys from model dict output"
    )
    parser.add_argument(
        "--dynamic_axes", 
        help='Dynamic axes: "input:0=batch;output:0=batch"'
    )
    parser.add_argument(
        "--simplify", 
        action="store_true", 
        help="Simplify ONNX model"
    )

    parser.add_argument(
        "--no_verify", 
        action="store_true", 
        help="Skip verification"
    )
    parser.add_argument(
        "--rtol", type=float, default=1e-3)
    parser.add_argument(
        "--atol", type=float, default=1e-5)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.format not in EXPORTERS:
        logger.error(
            f"Unknown export format: '{args.format}'. Available: {list(EXPORTERS.keys())}"
        )
        return 1

    export_cfg = load_export_config(args.export_config) if args.export_config else {}

    exporter_params = {
        "device": args.device,
        "opset_version": args.opset_version,
        "input_names": (
            parse_list(args.input_names)
            if args.input_names
            else export_cfg.get("input_names")
        ),
        "output_names": (
            parse_list(args.output_names)
            if args.output_names
            else export_cfg.get("output_names")
        ),
        "output_keys": (
            parse_list(args.output_keys)
            if args.output_keys
            else export_cfg.get("output_keys")
        ),
        "dynamic_axes": (
            parse_dynamic_axes(args.dynamic_axes)
            if args.dynamic_axes
            else export_cfg.get("dynamic_axes")
        ),
        "simplify": args.simplify or export_cfg.get("simplify", False),
        "rtol": args.rtol,
        "atol": args.atol,
    }

    for k, v in export_cfg.items():
        if k not in exporter_params:
            exporter_params[k] = v

    exporter_cls = EXPORTERS.get(args.format)
    exporter = exporter_cls(**exporter_params)

    logger.info(f"Loading model from {args.checkpoint}")
    exporter.load_model(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        model_loader=args.model_loader,
    )

    input_shape = parse_input_shape(args.input_shape) if args.input_shape else None
    dummy_input = prepare_dummy_input(
        config=exporter.config, input_shape=input_shape, device=args.device
    )
    input_shape_info = (
        dummy_input.shape 
        if hasattr(dummy_input, 'shape') 
        else {k: v.shape for k, v in dummy_input.items()}
    )
    logger.info(f"Input shape: {input_shape_info}")

    output_path = args.output or get_output_path(
        args.checkpoint, suffix=f".{args.format}"
    )
    logger.info(f"Exporting to {output_path} (format: {args.format})")
    result = exporter.export(
        exporter.model, dummy_input, output_path, verify=not args.no_verify
    )

    print_result(result)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    exit(main())
