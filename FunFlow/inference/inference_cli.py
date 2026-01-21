"""General inference command line tool."""

import argparse
import logging
from pathlib import Path

from FunFlow import build_inferencer
from FunFlow.inference import (
    parse_jsonl,
    collect_files,
    save_results,
    print_summary,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Inference CLI")

    parser.add_argument(
        "--inferencer",
        "-I",
        required=True,
        help="Registered inferencer type name, e.g., 'GIPDInferencer'",
    )
    parser.add_argument(
        "--checkpoint", "-c", required=True, help="Model checkpoint path"
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Input file/directory/jsonl"
    )

    parser.add_argument(
        "--config", help="Config path (default: checkpoint_dir/config.yaml)"
    )
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument(
        "--extensions",
        default=".txt,.wav,.mp3",
        help="File extensions for directory input",
    )
    parser.add_argument(
        "--file_field", default="file_path", help="File path field name in jsonl"
    )
    parser.add_argument(
        "--gt_fields",
        help='Ground truth field names in jsonl (comma-separated), e.g., "label,hoehn_yahr"',
    )
    parser.add_argument("--no_eval", action="store_true", help="Skip evaluation")
    parser.add_argument(
        "--return_raw", action="store_true", help="Return raw model outputs"
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=None,
        help="Number of threads for inference (None=auto, 0=single thread)",
    )

    args = parser.parse_args()

    inferencer_config = {
        "type": args.inferencer,
        "device": args.device,
        "num_threads": args.num_threads,
    }
    inferencer = build_inferencer(inferencer_config)
    inferencer.load_model(
        args.checkpoint, config_path=args.config, num_threads=args.num_threads
    )

    input_path = Path(args.input)
    ground_truths = None

    if input_path.suffix == ".jsonl":
        gt_fields = args.gt_fields.split(",") if args.gt_fields else None
        file_paths, ground_truths = parse_jsonl(
            args.input, file_field=args.file_field, gt_fields=gt_fields
        )
        logger.info(f"Loaded {len(file_paths)} samples from {args.input}")
    else:
        extensions = tuple(args.extensions.split(","))
        file_paths = collect_files(args.input, extensions=extensions)
        logger.info(f"Found {len(file_paths)} files")

    if not file_paths:
        logger.error("No input files found")
        return

    results = inferencer.predict(
        file_paths,
        ground_truths=ground_truths,
        batch_size=args.batch_size,
        return_raw=args.return_raw,
    )

    if not isinstance(results, list):
        results = [results]

    metrics = {}
    if ground_truths and not args.no_eval:
        try:
            metrics = inferencer.evaluate(results)
        except NotImplementedError:
            logger.warning("Evaluation not implemented for this inferencer")

    stats = inferencer.get_stats()

    if args.output:
        save_results(results, args.output, metrics=metrics, stats=stats)

    print_summary(results, metrics=metrics, stats=stats)


if __name__ == "__main__":
    main()
