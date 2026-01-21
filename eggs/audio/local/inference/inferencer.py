"""
音频分类推理器
"""
import logging
from pathlib import Path
from typing import Dict, Any, List

import torch
import torchaudio
import librosa
import numpy as np

from FunFlow.inference.base import BaseInferencer
from FunFlow.registry import INFERENCERS
from FunFlow.utils import load_config

logger = logging.getLogger(__name__)


@INFERENCERS.register('Inferencer')
class Inferencer(BaseInferencer):
    """音频分类推理器"""
    
    def __init__(
        self, 
        device: str = 'cuda', 
        enable_timing: bool = True, 
        num_threads: int = None, 
    ):
        super().__init__(device=device, enable_timing=enable_timing)
        self.config = None
        self.sample_rate = 16000
        self.num_mel_bins = 80
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
            if proc['name'] == 'load_audio':
                self.sample_rate = proc.get('target_sr', 16000)
            elif proc['name'] == 'compute_fbank':
                self.num_mel_bins = proc.get('num_mel_bins', 80)
        
        logger.info(f"Model loaded from {checkpoint_path}")
    
    def preprocess(self, file_path: str) -> torch.Tensor:
        """预处理音频文件"""
        # 加载音频
        waveform, sr = torchaudio.load(file_path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)
        
        # 计算 FBank 特征（使用 librosa 保证训练推理一致）
        y = waveform.numpy()
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_fft=int(self.sample_rate * 0.025),
            hop_length=int(self.sample_rate * 0.010),
            n_mels=self.num_mel_bins,
            fmin=0,
            fmax=self.sample_rate // 2,
            norm='slaney',
            htk=True,
        )
        fbank = librosa.power_to_db(mel_spec, ref=np.max, top_db=80.0)
        fbank = torch.from_numpy(fbank.T).float()  # (T, num_mel_bins)
        
        return fbank
    
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
        labels = torch.tensor([0 if gt['label'] == 'Healthy' else 1 
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
