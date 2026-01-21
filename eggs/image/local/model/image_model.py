"""
图像诊断模型
用于帕金森病图像分类任务

默认输入: 3 通道 RGB 图像，形状 (B, 3, H, W)，默认 H=W=224
默认Backbone: ResNet
默认Head: LinearHead

深度集成 Registry:
- 所有组件通过 Registry 构建
- 支持灵活的配置和扩展
"""
from typing import Dict, Any, Union

import torch
import torch.nn as nn

from FunFlow.registry import MODELS, HEADS
from FunFlow.models.base import BaseModel


@MODELS.register('ImageClassifier')
class ImageClassifier(BaseModel):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_weights()
    
    def _parse_inputs(self, inputs: Union[Dict[str, torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """预处理图像输入"""
        # 解析输入
        if isinstance(inputs, dict):
            x = inputs.get('images', inputs.get('image'))
            if x is None:
                raise KeyError(f"Cannot find 'images' in inputs: {inputs.keys()}")
        else:
            x = inputs
        
        # 确保输入形状正确 (B, C, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        return x


__all__ = ['ImageClassifier']
