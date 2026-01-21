"""
图像分类推理器
"""
import logging
from pathlib import Path
from typing import Dict, Any, List

import torch
from PIL import Image
import torchvision.transforms as T

from FunFlow.inference.base import BaseInferencer
from FunFlow.registry import INFERENCERS
from FunFlow.utils import load_config

logger = logging.getLogger(__name__)


@INFERENCERS.register('Inferencer')
class Inferencer(BaseInferencer):
    """图像分类推理器"""
    
    def __init__(
        self, 
        device: str = 'cuda', 
        enable_timing: bool = True, 
        num_threads: int = None, 
    ):
        super().__init__(device=device, enable_timing=enable_timing)
        self.config = None
        self.image_size = (224, 224)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.num_threads = num_threads
    
    def load_model(
        self,
        checkpoint_path: str = None,
        config_path: str = None,
        model: torch.nn.Module = None,
        **kwargs
    ) -> None:
        """加载模型，支持直接传入 model 或 checkpoint/config 路径，支持线程数"""
        from FunFlow.utils.model_loaders import load_pytorch_model
        num_threads = kwargs.get('num_threads', self.num_threads)
        
        if model is not None:
            self.model = model.to(self.device)
            self.model.eval()
            if num_threads is not None and num_threads >= 0:
                torch.set_num_threads(max(1, num_threads))
            logger.info("Model loaded from provided model object")
            return
        
        if checkpoint_path is None:
            raise ValueError("Either checkpoint_path or model must be provided")
        checkpoint_path = Path(checkpoint_path)
        checkpoint_dir = checkpoint_path.parent
        
        # 加载配置
        if config_path is None:
            config_path = checkpoint_dir / 'config.yaml'
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        self.config = load_config(config_path)
        
        # 使用 FunFlow 官方加载器
        self.model = load_pytorch_model(
            config=self.config,
            checkpoint_path=str(checkpoint_path),
            device=self.device,
            num_threads=num_threads,
        )
        
        # 提取数据配置
        val_conf = self.config['data']['val']['conf']
        for proc in val_conf['processors']:
            if proc['name'] == 'resize':
                self.image_size = tuple(proc.get('size', [224, 224]))
            elif proc['name'] == 'normalize':
                self.mean = proc.get('mean', [0.485, 0.456, 0.406])
                self.std = proc.get('std', [0.229, 0.224, 0.225])
        
        logger.info(f"Model loaded from {checkpoint_path}")
    
    def preprocess(self, file_path: str) -> torch.Tensor:
        """预处理图像文件"""
        # 加载图像
        image = Image.open(file_path).convert('RGB')
        
        # 构建预处理 pipeline
        transform = T.Compose([
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std),
        ])
        
        # 应用变换
        image_tensor = transform(image)  # (C, H, W)
        
        return image_tensor
    
    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """模型前向推理"""
        inputs = inputs.to(self.device)
        outputs = self.model(inputs)
        return outputs
    
    def postprocess(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """后处理模型输出"""
        probs = outputs['probs'].cpu()
        preds = outputs['preds'].cpu()
        
        # 批量处理，取第一个样本
        if probs.dim() > 1:
            probs = probs[0]
            preds = preds[0]
        
        return {
            'label': int(preds.item()),
            'confidence': float(probs[preds].item()),
            'probs': probs.tolist(),
        }
    
    def compute_metrics(
        self,
        predictions: List[Dict[str, Any]],
        ground_truths: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """计算评估指标：准确率、召回率、特异性"""
        preds = torch.tensor([p['label'] for p in predictions])
        labels = torch.tensor([0 if gt['label'] == 'healthy' else 1 
                               for gt in ground_truths])
        
        # 准确率
        correct = (preds == labels)
        accuracy = correct.float().mean().item()
        
        # 混淆矩阵元素
        tp = ((preds == 1) & (labels == 1)).sum().item()
        tn = ((preds == 0) & (labels == 0)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()
        
        # 召回率 (敏感性)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # 特异性
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'recall': recall,
            'specificity': specificity,
        }
