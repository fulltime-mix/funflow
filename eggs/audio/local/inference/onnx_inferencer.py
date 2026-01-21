"""
Audio Classification ONNX Inferencer

基于 ONNX Runtime 的音频分类推理器，继承自 AudioClassificationInferencer

使用方式:
    from local.inference import AudioClassificationONNXInferencer
    
    # 创建推理器
    inferencer = AudioClassificationONNXInferencer(device='cpu')  # ONNX 推荐使用 CPU
    inferencer.load_model('exp/audio_classification/avg_5.onnx', config_path='exp/audio_classification/config.yaml')
    
    # 单样本推理
    result = inferencer.predict('path/to/audio.wav')
    print(result.predictions)
    # {
    #     'label': 0,
    #     'confidence': 0.95,
    #     'probs': [0.95, 0.05]
    # }
    
    # 批量推理
    results = inferencer.predict(file_paths)
    
CLI 调用:
    python -m fundiagnosis.inference.inference_cli \\
        --inferencer AudioClassificationONNXInferencer \\
        --checkpoint exp/audio_classification/avg_5.onnx \\
        --input data/test/data.jsonl \\
        --output results.jsonl \\
        --device cpu
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
    """
    音频分类 ONNX 推理器
    
    继承自 AudioClassificationInferencer，重写以下方法:
        - __init__: 添加 ONNX 相关属性
        - load_model: 加载 ONNX 模型
        - forward: 使用 ONNX Runtime 推理
    
    复用父类方法:
        - preprocess: 预处理（与 PyTorch 版本一致）
        - postprocess: 后处理（与 PyTorch 版本一致）
        - compute_metrics: 评估指标计算
        - predict: 推理接口
    
    Attributes:
        session: ONNX Runtime InferenceSession
        input_name: ONNX 模型输入名称
        output_names: ONNX 模型输出名称列表
    """
    
    def __init__(
        self,
        device: str = 'cpu',  # ONNX 推荐使用 CPU
        enable_timing: bool = True,
        num_threads: int = 1,
        providers: Optional[List[str]] = None,
    ):
        """
        初始化 ONNX 推理器
        
        Args:
            device: 推理设备 ('cpu' 或 'cuda')
            enable_timing: 是否启用计时统计
            num_threads: ONNX Runtime 使用的线程数
            providers: ONNX Runtime 执行提供者列表
        """
        # 调用父类构造函数初始化基本属性
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
        self.model = None  # ONNX 推理器不使用 PyTorch 模型
    
    def load_model(
        self, 
        checkpoint_path: Optional[str] = None, 
        config_path: Optional[str] = None,
        session = None,
        **kwargs
    ) -> None:
        """
        加载 ONNX 模型
        
        支持两种方式：
        1. 通过 checkpoint_path 加载模型
        2. 直接传入已加载的 ONNX Runtime InferenceSession (session 参数)
        
        Args:
            checkpoint_path: ONNX 模型路径 (.onnx)，与 session 二选一
            config_path: 配置文件路径 (.yaml)，用于获取预处理参数
            session: 已加载的 ONNX Runtime InferenceSession，与 checkpoint_path 二选一
            **kwargs: 其他参数
                - num_threads: 覆盖初始化时的线程数
                - providers: 覆盖初始化时的执行提供者
        
        Raises:
            FileNotFoundError: 如果模型文件不存在
            ImportError: 如果 onnxruntime 未安装
            ValueError: 如果 checkpoint_path 和 session 都未提供
        """
        from FunFlow.utils.model_loaders import load_onnx_model
        
        # 获取线程数配置
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
                        if proc['name'] == 'load_audio':
                            self.sample_rate = proc.get('target_sr', 16000)
                        elif proc['name'] == 'compute_fbank':
                            self.num_mel_bins = proc.get('num_mel_bins', 80)
                    
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
                if proc['name'] == 'load_audio':
                    self.sample_rate = proc.get('target_sr', 16000)
                elif proc['name'] == 'compute_fbank':
                    self.num_mel_bins = proc.get('num_mel_bins', 80)
            
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
        """
        模型前向推理（使用 ONNX Runtime）
        
        Args:
            inputs: 输入特征 [batch_size, time, num_mel_bins] 或 [time, num_mel_bins]
        
        Returns:
            模型输出字典（torch.Tensor，与父类保持一致）：
            - probs: (batch_size, num_classes) 或 (num_classes,) 类别概率
            - preds: (batch_size,) 或 scalar 预测类别
        """
        # 确保输入是 3D: [batch_size, time, num_mel_bins]
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(0)  # [1, time, num_mel_bins]
        
        # 转换为 NumPy 进行 ONNX 推理
        inputs_np = inputs.cpu().numpy().astype(np.float32)
        
        # ONNX Runtime 推理
        onnx_inputs = {self.input_name: inputs_np}
        onnx_outputs = self.session.run(None, onnx_inputs)
        
        # 根据输出名称映射结果
        outputs_dict = {}
        for name, value in zip(self.output_names, onnx_outputs):
            outputs_dict[name] = value
        
        # 提取 probs 和 preds（优先使用 output_names，回退到索引）
        if 'probs' in outputs_dict:
            probs_np = outputs_dict['probs']
        elif len(onnx_outputs) >= 2:
            probs_np = onnx_outputs[1]  # 假设 probs 是第二个输出
        else:
            raise ValueError(f"Cannot find 'probs' in ONNX outputs. Available: {self.output_names}")
        
        if 'preds' in outputs_dict:
            preds_np = outputs_dict['preds']
        elif len(onnx_outputs) >= 3:
            preds_np = onnx_outputs[2]  # 假设 preds 是第三个输出
        else:
            # 如果没有 preds，从 probs 计算
            preds_np = np.argmax(probs_np, axis=-1)
        
        # 转换回 torch.Tensor
        probs = torch.from_numpy(probs_np).to(self.device)
        preds = torch.from_numpy(preds_np).to(self.device)
        
        return {
            'probs': probs,  # (batch_size, num_classes)
            'preds': preds,  # (batch_size,)
        }
    
    def _sync(self):
        """ONNX 不需要同步"""
        pass
    
    def warmup(self, num_runs: int = 10) -> None:
        """
        预热 ONNX Runtime 推理引擎
        
        Args:
            num_runs: 预热运行次数
        """
        if self.session is None:
            logger.warning("Model not loaded, skipping warmup")
            return
        
        # 创建 dummy 输入
        dummy_input = np.random.randn(1, 100, self.num_mel_bins).astype(np.float32)
        
        logger.info(f"Warming up ONNX Runtime with {num_runs} runs...")
        for _ in range(num_runs):
            self.session.run(None, {self.input_name: dummy_input})
        logger.info("Warmup completed")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取 ONNX 模型信息
        
        Returns:
            模型信息字典
        """
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
