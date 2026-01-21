from typing import Dict, Any, Union, Tuple

import torch
import torch.nn as nn

from FunFlow.registry import MODELS
from FunFlow.models.base import BaseModel
from FunFlow.models.preprocessing.cmvn import GlobalCMVN


@MODELS.register('AudioClassifier')
class AudioClassifier(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_weights()
    
    def _parse_inputs(self, 
        inputs: Union[Dict[str, torch.Tensor], torch.Tensor], 
        feat_lens: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """预处理音频输入"""
        if isinstance(inputs, dict):
            feats = inputs.get('feats', inputs.get('feat'))
            if feats is None:
                raise KeyError(
                    f"Cannot find 'feats' in inputs: {inputs.keys()}"
                )
            feat_lens = inputs.get('feat_lens', None)
        else:
            feats = inputs
            feat_lens = feat_lens

        if feats.dim() == 2:
            feats = feats.unsqueeze(0)
            if feat_lens is not None and feat_lens.dim() == 0:
                feat_lens = feat_lens.unsqueeze(0)
        
        return feats, feat_lens
    
    def forward(self, 
        inputs: Union[Dict[str, torch.Tensor], torch.Tensor], 
        feat_lens: torch.Tensor = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            inputs: 输入数据，可以是字典或张量
            
        Returns:
            包含 logits, probs, preds, features 的字典
        """
        x, x_lens = self._parse_inputs(inputs, feat_lens)

        x = self.preprocessing(x)

        x = self.quant(x)
        
        features, x_lens = self.backbone(x, x_lens)
        
        features = self.dequant(features)
        
        logits = self.head(features, x_lens)

        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        
        return {
            'features': features,
            'feat_lens': x_lens,
            'logits': logits,
            'probs': probs,
            'preds': preds,
        }


__all__ = ['AudioClassifier']


