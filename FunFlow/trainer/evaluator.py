"""Evaluator module for model evaluation"""

from typing import Dict, Any, List, Optional
import torch
from abc import ABC, abstractmethod

from FunFlow.registry import EVALUATORS


class BaseEvaluator(ABC):
    """Base evaluator for collecting model outputs and computing metrics"""

    def __init__(self, metric_prefix: str = ""):
        """Args:
        metric_prefix: Prefix for metric names
        """
        self.metric_prefix = metric_prefix
        self.reset()

    def reset(self):
        """Reset evaluator state"""
        self._outputs: List[Dict[str, Any]] = []
        self._loss_accum: Dict[str, float] = {}
        self._num_batches = 0

    def process_batch(self, outputs: Dict[str, Any], batch: Dict[str, Any]):
        """Process single batch outputs

        Args:
            outputs: Model outputs including loss
            batch: Input batch including labels
        """
        batch_output = {}

        assert "loss" in outputs, "outputs must contain 'loss'"
        if isinstance(outputs["loss"], dict):
            loss_dict = outputs["loss"]
        else:
            loss_dict = {"loss": outputs["loss"]}

        self._num_batches += 1
        for key, value in loss_dict.items():
            loss_val = value.item() if torch.is_tensor(value) else value
            if key not in self._loss_accum:
                self._loss_accum[key] = 0.0
            self._loss_accum[key] += loss_val

        for key, value in outputs.items():
            if key == "loss_dict":
                continue
            if isinstance(value, torch.Tensor):
                batch_output[key] = value.detach().cpu()
            else:
                batch_output[key] = value

        self._process_batch_labels(batch_output, batch)
        self._outputs.append(batch_output)

    def _process_batch_labels(
        self, batch_output: Dict[str, Any], batch: Dict[str, Any]
    ):
        """Extract labels from batch"""
        if "labels" in batch:
            labels = batch["labels"]
            if isinstance(labels, torch.Tensor):
                batch_output["labels"] = labels.detach().cpu()
            else:
                batch_output["labels"] = labels

    def aggregate_outputs(self) -> Dict[str, Any]:
        """Aggregate outputs from all batches

        Returns:
            Dict of aggregated outputs
        """
        if not self._outputs:
            return {}

        aggregated = {}

        all_keys = set()
        for out in self._outputs:
            all_keys.update(out.keys())

        for key in all_keys:
            values = [out[key] for out in self._outputs if key in out]
            if not values:
                continue

            if isinstance(values[0], torch.Tensor):
                try:
                    aggregated[key] = torch.cat(values, dim=0)
                except:
                    aggregated[key] = values
            else:
                aggregated[key] = values

        if self._num_batches > 0:
            for loss_key, loss_sum in self._loss_accum.items():
                aggregated[loss_key] = loss_sum / self._num_batches

        return aggregated

    @abstractmethod
    def compute_metrics(self, aggregated_outputs: Dict[str, Any]) -> Dict[str, float]:
        """Compute evaluation metrics

        Args:
            aggregated_outputs: Aggregated outputs

        Returns:
            Dict of metrics
        """
        pass

    def evaluate(
        self, outputs_list: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, float]:
        """Run evaluation

        Args:
            outputs_list: Optional list of outputs

        Returns:
            Dict of metrics
        """
        if outputs_list is not None:
            self.reset()
            for item in outputs_list:
                if isinstance(item, tuple) and len(item) == 2:
                    outputs, batch = item
                    self.process_batch(outputs, batch)
                else:
                    self.process_batch(item, {})

        aggregated = self.aggregate_outputs()

        if not aggregated:
            return {}

        metrics = {}
        for loss_key in self._loss_accum.keys():
            if loss_key in aggregated:
                metrics[loss_key] = aggregated[loss_key]

        computed_metrics = self.compute_metrics(aggregated)
        metrics.update(computed_metrics)

        if self.metric_prefix:
            metrics = {f"{self.metric_prefix}{k}": v for k, v in metrics.items()}

        return metrics


@EVALUATORS.register("ClassificationEvaluator")
class ClassificationEvaluator(BaseEvaluator):
    """Classification evaluator"""

    def compute_metrics(self, aggregated_outputs: Dict[str, Any]) -> Dict[str, float]:
        preds = aggregated_outputs.get("preds")
        labels = aggregated_outputs.get("labels")

        if preds is None or labels is None:
            return {}

        if isinstance(preds, list):
            preds = torch.cat(preds)
        if isinstance(labels, list):
            labels = torch.cat(labels)

        preds = preds.view(-1)
        labels = labels.view(-1)

        correct = preds == labels
        accuracy = correct.float().mean().item()

        return {"accuracy": accuracy}


def build_evaluator(cfg: Dict[str, Any]) -> BaseEvaluator:
    """Build evaluator from config

    Args:
        cfg: Evaluator config

    Returns:
        Evaluator instance
    """
    return EVALUATORS.build(cfg)
