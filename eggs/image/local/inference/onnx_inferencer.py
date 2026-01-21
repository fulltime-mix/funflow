"""
图像分类 ONNX 推理器
"""
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import yaml
import numpy as np
import torch

from FunFlow.registry import INFERENCERS
from .inferencer import Inferencer

logger = logging.getLogger(__name__)


@INFERENCERS.register('ONNXInferencer', force=True)
class ONNXInferencer(Inferencer):
    """图像分类 ONNX 推理器"""
    
    def __init__(
        self,
        device: str = 'cpu',
        enable_timing: bool = True,
        num_threads: int = 1,
        providers: Optional[List[str]] = None,
    ):
        super().__init__(
            device=device,
            enable_timing=enable_timing,
            num_threads=num_threads,
        )
        
        # ONNX 特有属性
        self.session = None
        self.input_name = None
        self.output_names = None
        self.providers = providers
        self.model = None
    
    def load_model(
        self, 
        checkpoint_path: Optional[str] = None, 
        config_path: Optional[str] = None,
        session = None,
        **kwargs
    ) -> None:
        """加载 ONNX 模型"""
        from FunFlow.utils.model_loaders import load_onnx_model
        
        num_threads = kwargs.get('num_threads', self.num_threads)
        
        # 方式1: 直接传入已加载的 session
        if session is not None:
            self.session = session
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            # 加载配置（如果提供）
            if config_path is not None:
                config_path = Path(config_path)
                if config_path.exists():
                    with open(config_path, 'r', encoding='utf-8') as f:
                        self.config = yaml.safe_load(f)
                    
                    # 提取数据配置
                    val_conf = self.config['data']['val']['conf']
                    for proc in val_conf['processors']:
                        if proc['name'] == 'resize':
                            self.image_size = tuple(proc.get('size', [224, 224]))
                        elif proc['name'] == 'normalize':
                            self.mean = proc.get('mean', [0.485, 0.456, 0.406])
                            self.std = proc.get('std', [0.229, 0.224, 0.225])
                    
                    logger.info(f"Config loaded from {config_path}")
            
            logger.info(f"Session set from provided session object")
            logger.info(f"  Input: {self.input_name}")
            logger.info(f"  Outputs: {self.output_names}")
            logger.info(f"  Providers: {self.session.get_providers()}")
            return
        
        # 方式2: 通过 checkpoint_path 加载模型
        if checkpoint_path is None:
            raise ValueError("Either checkpoint_path or session must be provided")
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {checkpoint_path}")
        
        # 加载配置（如果提供）
        checkpoint_dir = checkpoint_path.parent
        if config_path is None:
            config_path = checkpoint_dir / 'config.yaml'
        config_path = Path(config_path)
        
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            # 提取数据配置
            val_conf = self.config['data']['val']['conf']
            for proc in val_conf['processors']:
                if proc['name'] == 'resize':
                    self.image_size = tuple(proc.get('size', [224, 224]))
                elif proc['name'] == 'normalize':
                    self.mean = proc.get('mean', [0.485, 0.456, 0.406])
                    self.std = proc.get('std', [0.229, 0.224, 0.225])
            
            logger.info(f"Config loaded from {config_path}")
        
        # 使用 model_loaders 加载模型
        self.session = load_onnx_model(
            config=self.config or {},
            checkpoint_path=str(checkpoint_path),
            device=self.device,
            num_threads=num_threads,
        )
        
        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        logger.info(f"ONNX model loaded from {checkpoint_path}")
        logger.info(f"  Input: {self.input_name}")
        logger.info(f"  Outputs: {self.output_names}")
        logger.info(f"  Providers: {self.session.get_providers()}")
    
    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """模型前向推理（使用 ONNX Runtime）"""
        # 确保输入是 4D: [batch_size, C, H, W]
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)  # [1, C, H, W]
        
        # 转换为 NumPy 进行 ONNX 推理
        inputs_np = inputs.cpu().numpy().astype(np.float32)
        
        # ONNX Runtime 推理
        onnx_inputs = {self.input_name: inputs_np}
        onnx_outputs = self.session.run(None, onnx_inputs)
        
        # 根据输出名称映射结果
        outputs_dict = {}
        for name, value in zip(self.output_names, onnx_outputs):
            outputs_dict[name] = value
        
        # 提取 probs 和 preds
        if 'probs' in outputs_dict:
            probs_np = outputs_dict['probs']
        elif len(onnx_outputs) >= 2:
            probs_np = onnx_outputs[1]
        else:
            raise ValueError(f"Cannot find 'probs' in ONNX outputs. Available: {self.output_names}")
        
        if 'preds' in outputs_dict:
            preds_np = outputs_dict['preds']
        elif len(onnx_outputs) >= 3:
            preds_np = onnx_outputs[2]
        else:
            preds_np = np.argmax(probs_np, axis=-1)
        
        # 转换回 torch.Tensor
        probs = torch.from_numpy(probs_np).to(self.device)
        preds = torch.from_numpy(preds_np).to(self.device)
        
        return {
            'probs': probs,
            'preds': preds,
        }
    
    def _sync(self):
        """ONNX 不需要同步"""
        pass
    
    def warmup(self, num_runs: int = 10) -> None:
        """预热 ONNX Runtime 推理引擎"""
        if self.session is None:
            logger.warning("Model not loaded, skipping warmup")
            return
        
        # 创建 dummy 输入
        dummy_input = np.random.randn(1, 3, self.image_size[0], self.image_size[1]).astype(np.float32)
        
        logger.info(f"Warming up ONNX Runtime with {num_runs} runs...")
        for _ in range(num_runs):
            self.session.run(None, {self.input_name: dummy_input})
        logger.info("Warmup completed")
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取 ONNX 模型信息"""
        if self.session is None:
            return {'error': 'Model not loaded'}
        
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        
        return {
            'inputs': [
                {'name': inp.name, 'shape': inp.shape, 'type': inp.type}
                for inp in inputs
            ],
            'outputs': [
                {'name': out.name, 'shape': out.shape, 'type': out.type}
                for out in outputs
            ],
            'providers': self.session.get_providers(),
        }
