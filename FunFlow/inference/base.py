"""Lightweight inference module with base classes for task-specific inferencers."""

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Single sample inference result.

    Args:
        file_path: Path to input file
        predictions: Prediction results
        status: Inference status ('success' or 'failed')
        error: Error message if failed
        ground_truth: Ground truth labels
        raw_outputs: Raw model outputs
        timing_ms: Timing information in milliseconds
    """

    file_path: str
    predictions: Dict[str, Any]
    status: str = "success"
    error: Optional[str] = None
    ground_truth: Optional[Dict[str, Any]] = None
    raw_outputs: Optional[Dict[str, Any]] = None
    timing_ms: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "file": self.file_path,
            "predictions": self.predictions,
            "status": self.status,
        }
        if self.error:
            d["error"] = self.error
        if self.ground_truth:
            d["ground_truth"] = self.ground_truth
        if self.timing_ms:
            d["timing_ms"] = self.timing_ms
        return d

    @classmethod
    def failed(cls, file_path: str, error: str) -> "InferenceResult":
        return cls(file_path=file_path, predictions={}, status="failed", error=error)


@dataclass
class InferenceStats:
    """Inference statistics tracker."""

    preprocess_times: List[float] = field(default_factory=list)
    forward_times: List[float] = field(default_factory=list)
    postprocess_times: List[float] = field(default_factory=list)

    def add(self, preprocess: float, forward: float, postprocess: float):
        self.preprocess_times.append(preprocess)
        self.forward_times.append(forward)
        self.postprocess_times.append(postprocess)

    def summary(self) -> Dict[str, Any]:
        n = len(self.preprocess_times)
        if n == 0:
            return {"num_samples": 0}

        def _stat(arr):
            return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}

        total = (
            np.array(self.preprocess_times)
            + np.array(self.forward_times)
            + np.array(self.postprocess_times)
        )
        return {
            "num_samples": n,
            "preprocess_ms": _stat([t * 1000 for t in self.preprocess_times]),
            "forward_ms": _stat([t * 1000 for t in self.forward_times]),
            "postprocess_ms": _stat([t * 1000 for t in self.postprocess_times]),
            "total_ms": _stat(total * 1000),
            "throughput": n / total.sum() if total.sum() > 0 else 0,
        }

    def reset(self):
        self.preprocess_times.clear()
        self.forward_times.clear()
        self.postprocess_times.clear()


class BaseInferencer(ABC):
    """Base inferencer class defining standard inference pipeline.

    Lifecycle: __init__ -> load_model -> predict_* -> (evaluate)

    Core methods (must implement):
        - load_model: Load model from checkpoint
        - preprocess: Preprocess single file
        - forward: Model forward pass
        - postprocess: Postprocess model outputs

    Optional methods:
        - compute_metrics: Compute evaluation metrics
    """

    def __init__(self, device: str = "cuda", enable_timing: bool = True):
        self.device = device
        self.enable_timing = enable_timing
        self.stats = InferenceStats()
        self.model = None

    @abstractmethod
    def load_model(self, checkpoint_path: str, **kwargs) -> None:
        """Load model from checkpoint.

        Args:
            checkpoint_path: Path to model weights
            **kwargs: Additional parameters (e.g., config_path)
        """
        pass

    @abstractmethod
    def preprocess(self, file_path: str) -> Any:
        """Preprocess single file.

        Args:
            file_path: Input file path

        Returns:
            Model input (e.g., torch.Tensor)
        """
        pass

    @abstractmethod
    def forward(self, inputs: Any) -> Dict[str, torch.Tensor]:
        """Model forward pass.

        Args:
            inputs: Preprocessed inputs

        Returns:
            Model outputs dict, e.g., {'logits': tensor, 'embeddings': tensor}
        """
        pass

    @abstractmethod
    def postprocess(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Postprocess model outputs.

        Args:
            outputs: Dict returned by forward

        Returns:
            Final predictions, e.g., {'label': 1, 'confidence': 0.95}
        """
        pass

    @torch.no_grad()
    def predict(
        self,
        inputs: Union[str, List[str]],
        ground_truths: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        batch_size: int = 32,
        return_raw: bool = False,
    ) -> Union[InferenceResult, List[InferenceResult]]:
        """Unified inference interface for single or batch inference.

        Args:
            inputs: Input file path(s) - single string or list
            ground_truths: Ground truth labels - single dict or list (optional)
            batch_size: Batch size (only for batch inference)
            return_raw: Whether to return raw model outputs

        Returns:
            Single InferenceResult or List[InferenceResult]
        """
        if isinstance(inputs, str):
            gt = (
                ground_truths
                if isinstance(ground_truths, dict) or ground_truths is None
                else ground_truths[0]
            )

            try:
                t0 = time.perf_counter()
                tensor_input = self.preprocess(inputs)
                t1 = time.perf_counter()

                if isinstance(tensor_input, torch.Tensor):
                    batch_input = tensor_input.unsqueeze(0)
                else:
                    batch_input = tensor_input

                outputs = self.forward(batch_input)
                self._sync()
                t2 = time.perf_counter()

                predictions = self.postprocess(outputs)
                t3 = time.perf_counter()

                timing = None
                if self.enable_timing:
                    pre, fwd, post = t1 - t0, t2 - t1, t3 - t2
                    self.stats.add(pre, fwd, post)
                    timing = {
                        "preprocess": pre * 1000,
                        "forward": fwd * 1000,
                        "postprocess": post * 1000,
                        "total": (t3 - t0) * 1000,
                    }

                return InferenceResult(
                    file_path=inputs,
                    predictions=predictions,
                    ground_truth=gt,
                    raw_outputs=(
                        {k: v.cpu().tolist() for k, v in outputs.items()}
                        if return_raw
                        else None
                    ),
                    timing_ms=timing,
                )
            except Exception as e:
                logger.error(f"Inference failed for {inputs}: {e}")
                return InferenceResult.failed(inputs, str(e))

        file_paths = inputs
        n = len(file_paths)

        if ground_truths is None:
            ground_truths = [None] * n
        elif isinstance(ground_truths, dict):
            ground_truths = [ground_truths] * n

        results = [None] * n

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_paths = file_paths[start:end]
            batch_gts = ground_truths[start:end]

            if n > batch_size:
                logger.info(f"Batch [{start+1}-{end}] / {n}")

            t0 = time.perf_counter()
            batch_tensors = []
            valid_indices = []

            for i, path in enumerate(batch_paths):
                try:
                    tensor = self.preprocess(path)
                    batch_tensors.append(tensor)
                    valid_indices.append(i)
                except Exception as e:
                    logger.warning(f"Preprocess failed for {path}: {e}")
                    results[start + i] = InferenceResult.failed(
                        path, f"Preprocess failed: {e}"
                    )

            if not batch_tensors:
                continue

            if isinstance(batch_tensors[0], torch.Tensor):
                max_len = max(t.shape[0] for t in batch_tensors)
                padded_tensors = []
                for tensor in batch_tensors:
                    if tensor.shape[0] < max_len:
                        pad_size = max_len - tensor.shape[0]
                        pad = torch.zeros(
                            pad_size,
                            *tensor.shape[1:],
                            dtype=tensor.dtype,
                            device=tensor.device,
                        )
                        tensor = torch.cat([tensor, pad], dim=0)
                    padded_tensors.append(tensor)
                batch_inputs = torch.stack(padded_tensors, dim=0)
            else:
                batch_inputs = batch_tensors

            t1 = time.perf_counter()

            outputs = self.forward(batch_inputs)
            self._sync()
            t2 = time.perf_counter()

            t3 = time.perf_counter()
            batch_results = []
            for i in range(len(valid_indices)):
                sample_outputs = {k: v[i : i + 1] for k, v in outputs.items()}
                predictions = self.postprocess(sample_outputs)
                batch_results.append(predictions)
            t4 = time.perf_counter()

            n_valid = len(valid_indices)
            pre_t = (t1 - t0) / n_valid
            fwd_t = (t2 - t1) / n_valid
            post_t = (t4 - t3) / n_valid

            for j, local_idx in enumerate(valid_indices):
                global_idx = start + local_idx

                timing = None
                if self.enable_timing:
                    self.stats.add(pre_t, fwd_t, post_t)
                    timing = {
                        "preprocess": pre_t * 1000,
                        "forward": fwd_t * 1000,
                        "postprocess": post_t * 1000,
                        "total": (pre_t + fwd_t + post_t) * 1000,
                    }

                raw = None
                if return_raw:
                    raw = {k: v[j].cpu().tolist() for k, v in outputs.items()}

                results[global_idx] = InferenceResult(
                    file_path=batch_paths[local_idx],
                    predictions=batch_results[j],
                    ground_truth=batch_gts[local_idx],
                    raw_outputs=raw,
                    timing_ms=timing,
                )

        return results

    def compute_metrics(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Compute evaluation metrics (optional implementation).

        Args:
            predictions: List of predictions
            ground_truths: List of ground truths

        Returns:
            Metrics dict, e.g., {'accuracy': 0.95, 'f1': 0.92}
        """
        raise NotImplementedError("Subclass should implement evaluate()")

    def evaluate(self, results: List[InferenceResult]) -> Dict[str, float]:
        """Run evaluation.

        Args:
            results: List of inference results

        Returns:
            Evaluation metrics dict
        """
        successful = [r for r in results if r.status == "success" and r.ground_truth]
        if not successful:
            logger.warning("No valid samples for evaluation")
            return {}

        predictions = [r.predictions for r in successful]
        ground_truths = [r.ground_truth for r in successful]

        return self.compute_metrics(predictions, ground_truths)

    def _sync(self):
        """Synchronize CUDA operations for accurate timing."""
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        return self.stats.summary()

    def reset_stats(self):
        """Reset statistics."""
        self.stats.reset()
