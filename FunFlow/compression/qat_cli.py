#!/usr/bin/env python3
import argparse
import sys

from FunFlow.compression.QAT import QATQuantizer
from FunFlow.logger import get_logger

logger = get_logger("QAT-CLI")


def main():
    parser = argparse.ArgumentParser(
        description="Convert QAT checkpoint to quantized model"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to QAT checkpoint"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument(
        "--output", type=str, required=True, help="Output path for quantized model"
    )
    parser.add_argument(
        "--save-full-model",
        action="store_true",
        help="Save full model instead of state_dict",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="fbgemm",
        choices=["fbgemm", "qnnpack", "x86", "onednn"],
    )

    args = parser.parse_args()

    try:
        QATQuantizer.convert_save_model(
            checkpoint_path=args.checkpoint,
            config=args.config,
            output_path=args.output,
            backend=args.backend,
            save_full_model=args.save_full_model,
        )
        return 0
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
