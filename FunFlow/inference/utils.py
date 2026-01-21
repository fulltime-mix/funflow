"""Inference module utility functions."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Dict, Any, Union, Optional, Tuple

if TYPE_CHECKING:
    from FunFlow.inference.base import InferenceResult

logger = logging.getLogger(__name__)


def parse_jsonl(
    jsonl_path: str,
    file_field: str = "file_path",
    gt_fields: Optional[List[str]] = None,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Parse JSONL file to extract file paths and ground truths.

    Args:
        jsonl_path: Path to JSONL file
        file_field: Field name for file path
        gt_fields: List of ground truth field names

    Returns:
        Tuple of (file_paths, ground_truths)
    """
    file_paths = []
    ground_truths = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())

            raw = item.get("raw", item)
            if isinstance(raw, str):
                raw = json.loads(raw)

            file_path = raw.get(file_field)
            if not file_path:
                continue

            file_paths.append(file_path)

            if gt_fields:
                gt = {k: raw.get(k) for k in gt_fields if k in raw}
            else:
                gt = {k: v for k, v in raw.items() if k != file_field}
            ground_truths.append(gt if gt else None)

    return file_paths, ground_truths


def collect_files(
    source: Union[str, Path, List[str]],
    extensions: tuple = (".wav", ".mp3", ".flac"),
    recursive: bool = True,
) -> List[str]:
    """Collect input files from source.

    Args:
        source: File/directory path or list of files
        extensions: Supported file extensions
        recursive: Whether to search subdirectories

    Returns:
        List of file paths
    """
    if isinstance(source, list):
        return [str(p) for p in source if Path(p).suffix.lower() in extensions]

    source = Path(source)
    if source.is_file():
        return [str(source)] if source.suffix.lower() in extensions else []

    if source.is_dir():
        pattern = "**/*" if recursive else "*"
        files = []
        for ext in extensions:
            files.extend(source.glob(f"{pattern}{ext}"))
        return sorted(str(f) for f in files)

    return []


def save_results(
    results: List["InferenceResult"],
    output_path: Union[str, Path],
    metrics: Optional[Dict[str, float]] = None,
    stats: Optional[Dict[str, Any]] = None,
) -> None:
    """Save inference results to JSON file.

    Args:
        results: List of InferenceResult
        output_path: Output file path
        metrics: Evaluation metrics
        stats: Inference statistics
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {}

    total = len(results)
    success = sum(1 for r in results if r.status == "success")
    failed = total - success
    output_data["summary"] = {
        "total": total,
        "success": success,
        "failed": failed,
        "success_rate": round(success / total * 100, 2) if total > 0 else 0.0,
    }

    if metrics:
        output_data["metrics"] = metrics

    if stats and stats.get("num_samples", 0) > 0:
        timing = {"num_samples": stats["num_samples"]}
        for key in ["preprocess_ms", "forward_ms", "postprocess_ms", "total_ms"]:
            if key in stats:
                timing[key] = stats[key]
        if "throughput" in stats:
            timing["throughput"] = stats["throughput"]
        output_data["timing"] = timing

    output_data["results"] = [r.to_dict() for r in results]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved to {output_path}")


def print_summary(
    results: List["InferenceResult"],
    metrics: Optional[Dict[str, float]] = None,
    stats: Optional[Dict[str, Any]] = None,
) -> None:
    """Print inference summary.

    Args:
        results: List of InferenceResult
        metrics: Evaluation metrics
        stats: Inference statistics
    """
    total = len(results)
    success = sum(1 for r in results if r.status == "success")
    failed = total - success

    print("\n" + "=" * 50)
    print("Inference Summary")
    print("=" * 50)
    print(f"Total samples: {total}")
    print(f"Success: {success} ({success/total*100:.1f}%)")
    print(f"Failed: {failed} ({failed/total*100:.1f}%)")

    if metrics:
        print("\nEvaluation Metrics:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    if stats and stats.get("num_samples", 0) > 0:
        print("\nTiming Statistics:")
        for key in ["preprocess_ms", "forward_ms", "postprocess_ms", "total_ms"]:
            if key in stats:
                s = stats[key]
                print(f"  {key}: {s['mean']:.2f} ± {s['std']:.2f}")
        if "throughput" in stats:
            print(f"  Throughput: {stats['throughput']:.2f} samples/sec")

    print("=" * 50 + "\n")
