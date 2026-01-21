from typing import Dict, Any
import torch

from FunFlow.trainer.evaluator import BaseEvaluator
from FunFlow.registry import EVALUATORS


@EVALUATORS.register('AudioClassificationEvaluator')
class AudioClassificationEvaluator(BaseEvaluator):
    
    def compute_metrics(self, aggregated_outputs: Dict[str, Any]) -> Dict[str, float]:
        preds = aggregated_outputs.get('preds')
        labels = aggregated_outputs.get('labels')
        
        if preds is None or labels is None:
            return {}
        
        if isinstance(preds, list):
            preds = torch.cat(preds)
        if isinstance(labels, list):
            labels = torch.cat(labels)
        
        preds = preds.view(-1)
        labels = labels.view(-1)
        
        correct = (preds == labels)
        accuracy = correct.float().mean().item()
        
        tp = ((preds == 1) & (labels == 1)).sum().item()
        tn = ((preds == 0) & (labels == 0)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'recall': recall,
            'specificity': specificity,
        }
