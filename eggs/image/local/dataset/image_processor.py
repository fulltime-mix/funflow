"""
图像处理器模块
用于图像分类任务的数据处理和增强

默认输出: 
- 图像: 3通道 RGB 图像，归一化后的 tensor，形状 (C, H, W)
- 支持的输入格式: PIL Image, numpy array
- 默认归一化: ImageNet 统计值 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

支持的处理器:
- load_image: 加载图像文件
- resize: 调整图像大小
- to_tensor: 转换为 PyTorch tensor
- normalize: 归一化
- random_flip: 随机水平翻转（数据增强）
- random_rotation: 随机旋转（数据增强）
- random_crop: 随机裁剪（数据增强）
- center_crop: 中心裁剪
- color_jitter: 颜色抖动（数据增强）
- random_grayscale: 随机灰度化（数据增强）
"""
import json
import random
from typing import Dict, Any, Iterator, List, Tuple

import torch

from FunFlow.logger import get_logger

# 模块级 logger，所有函数共享
logger = get_logger('FunDiagnosis')

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


# ============================================================
# 默认配置
# ============================================================

DEFAULT_LABEL_MAP = {'healthy': 0, 'parkinson': 1}
DEFAULT_IMAGE_SIZE = (224, 224)  # (H, W)
DEFAULT_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
DEFAULT_STD = [0.229, 0.224, 0.225]   # ImageNet std


# ============================================================
# 核心处理函数
# ============================================================

def load_image(data: Iterator, mode: str = 'RGB') -> Iterator:
    """加载图像文件"""
    if not PIL_AVAILABLE:
        raise RuntimeError("PIL required")
    
    for sample in data:
        image_path = sample.get('image_path')
        if not image_path:
            yield sample
            continue
        try:
            image = Image.open(image_path).convert(mode)
            sample['image'] = image
            sample['original_size'] = image.size
            yield sample
        except Exception as e:
            logger.warning(f"Failed to load {image_path}: {e}")


def resize(data: Iterator, size: Tuple[int, int] = (224, 224)) -> Iterator:
    """调整图像大小
    
    Args:
        data: 输入数据迭代器
        size: 目标尺寸 (H, W)
    
    Yields:
        调整大小后的样本
    """
    for sample in data:
        image = sample.get('image')
        if image is None:
            yield sample
            continue
        # PIL resize expects (W, H), so we reverse the tuple
        sample['image'] = image.resize((size[1], size[0]), Image.BILINEAR)
        yield sample


def to_tensor(data: Iterator) -> Iterator:
    """转换为 tensor
    
    将 PIL Image 或 numpy array 转换为 torch.Tensor
    输出范围 [0, 1]，形状 (C, H, W)
    
    Yields:
        包含 tensor 格式图像的样本
    """
    if not TORCHVISION_AVAILABLE:
        raise RuntimeError("torchvision required")
    transform = T.ToTensor()
    for sample in data:
        image = sample.get('image')
        if image is None:
            yield sample
            continue
        sample['image'] = transform(image)
        yield sample


def normalize(data: Iterator, mean: List[float] = None, std: List[float] = None) -> Iterator:
    """归一化
    
    使用 ImageNet 的均值和标准差进行归一化
    
    Args:
        data: 输入数据迭代器
        mean: RGB 通道均值，默认 ImageNet 统计值
        std: RGB 通道标准差，默认 ImageNet 统计值
    
    Yields:
        归一化后的样本
    """
    mean = mean or [0.485, 0.456, 0.406]
    std = std or [0.229, 0.224, 0.225]
    
    for sample in data:
        image = sample.get('image')
        if image is None or not isinstance(image, torch.Tensor):
            yield sample
            continue
        sample['image'] = TF.normalize(image, mean, std)
        yield sample


# ============================================================
# 数据增强
# ============================================================

def random_flip(data: Iterator, prob: float = 0.5) -> Iterator:
    """随机水平翻转
    
    Args:
        data: 输入数据迭代器
        prob: 翻转概率
    
    Yields:
        可能被翻转的样本
    """
    for sample in data:
        image = sample.get('image')
        if image is None or random.random() > prob:
            yield sample
            continue
        if isinstance(image, torch.Tensor):
            sample['image'] = torch.flip(image, dims=[-1])
        else:
            sample['image'] = image.transpose(Image.FLIP_LEFT_RIGHT)
        yield sample


def random_rotation(data: Iterator, degrees: float = 15, prob: float = 0.5) -> Iterator:
    """随机旋转
    
    Args:
        data: 输入数据迭代器
        degrees: 旋转角度范围 [-degrees, degrees]
        prob: 旋转概率
    
    Yields:
        可能被旋转的样本
    """
    for sample in data:
        image = sample.get('image')
        if image is None or random.random() > prob:
            yield sample
            continue
        angle = random.uniform(-degrees, degrees)
        if isinstance(image, torch.Tensor):
            sample['image'] = TF.rotate(image, angle)
        else:
            sample['image'] = image.rotate(angle, resample=Image.BILINEAR)
        yield sample


def color_jitter(data: Iterator, brightness: float = 0.2, contrast: float = 0.2,
                 saturation: float = 0.2, prob: float = 0.5) -> Iterator:
    """颜色抖动
    
    随机改变图像的亮度、对比度、饱和度
    
    Args:
        data: 输入数据迭代器
        brightness: 亮度抖动范围
        contrast: 对比度抖动范围
        saturation: 饱和度抖动范围
        prob: 应用概率
    
    Yields:
        可能被颜色抖动的样本
    """
    if not TORCHVISION_AVAILABLE:
        for sample in data:
            yield sample
        return
    
    jitter = T.ColorJitter(brightness, contrast, saturation)
    for sample in data:
        image = sample.get('image')
        if image is None or random.random() > prob:
            yield sample
            continue
        sample['image'] = jitter(image)
        yield sample


def random_crop(data: Iterator, size: Tuple[int, int] = (224, 224), prob: float = 1.0) -> Iterator:
    """随机裁剪
    
    从图像中随机裁剪指定大小的区域
    
    Args:
        data: 输入数据迭代器
        size: 裁剪大小 (H, W)
        prob: 应用概率
    
    Yields:
        可能被裁剪的样本
    """
    if not TORCHVISION_AVAILABLE:
        for sample in data:
            yield sample
        return
    
    for sample in data:
        image = sample.get('image')
        if image is None or random.random() > prob:
            yield sample
            continue
        
        if isinstance(image, torch.Tensor):
            # Tensor format: (C, H, W)
            _, h, w = image.shape
            if h >= size[0] and w >= size[1]:
                sample['image'] = T.RandomCrop(size)(image)
            else:
                # 如果图像太小，先resize再裁剪
                sample['image'] = T.Resize(size)(image)
        else:
            # PIL Image
            w, h = image.size
            if h >= size[0] and w >= size[1]:
                sample['image'] = T.RandomCrop(size)(image)
            else:
                sample['image'] = image.resize((size[1], size[0]), Image.BILINEAR)
        
        yield sample


def center_crop(data: Iterator, size: Tuple[int, int] = (224, 224)) -> Iterator:
    """中心裁剪
    
    从图像中心裁剪指定大小的区域
    
    Args:
        data: 输入数据迭代器
        size: 裁剪大小 (H, W)
    
    Yields:
        被中心裁剪的样本
    """
    if not TORCHVISION_AVAILABLE:
        for sample in data:
            yield sample
        return
    
    for sample in data:
        image = sample.get('image')
        if image is None:
            yield sample
            continue
        
        if isinstance(image, torch.Tensor):
            _, h, w = image.shape
            if h >= size[0] and w >= size[1]:
                sample['image'] = T.CenterCrop(size)(image)
            else:
                sample['image'] = T.Resize(size)(image)
        else:
            w, h = image.size
            if h >= size[0] and w >= size[1]:
                sample['image'] = T.CenterCrop(size)(image)
            else:
                sample['image'] = image.resize((size[1], size[0]), Image.BILINEAR)
        
        yield sample


def random_grayscale(data: Iterator, prob: float = 0.1) -> Iterator:
    """随机灰度化
    
    以一定概率将图像转换为灰度图（保持3通道）
    
    Args:
        data: 输入数据迭代器
        prob: 灰度化概率
    
    Yields:
        可能被灰度化的样本
    """
    if not TORCHVISION_AVAILABLE:
        for sample in data:
            yield sample
        return
    
    gray = T.RandomGrayscale(p=prob)
    for sample in data:
        image = sample.get('image')
        if image is None:
            yield sample
            continue
        sample['image'] = gray(image)
        yield sample


# ============================================================
# 数据解析和整理
# ============================================================

def parse_raw(data: Iterator, label_map: Dict[str, int] = None) -> Iterator:
    """解析 JSONL 原始数据
    
    从原始 JSON 数据中提取图像路径、标签等信息
    
    Args:
        data: 输入数据迭代器
        label_map: 标签名到数字的映射字典
    
    Yields:
        解析后的样本，包含 record_id, image_path, label_name, label
    """
    label_map = label_map or DEFAULT_LABEL_MAP
    
    for sample in data:
        raw = sample.get('raw')
        if raw is None:
            yield sample
            continue
        
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON")
                continue
        
        parsed = {
            'record_id': raw.get('image_id', raw.get('record_id', raw.get('id', ''))),
            'image_path': raw.get('image_path', raw.get('path', '')),
            'label_name': raw.get('label', ''),
        }
        
        label_name = parsed['label_name'].lower()
        parsed['label'] = label_map.get(label_name, -1)
        
        # 保留采样器信息
        for key in ['rank', 'world_size', 'worker_id', 'num_workers']:
            if key in sample:
                parsed[key] = sample[key]
        
        yield parsed


def collate_fn(data: Iterator) -> Iterator:
    """批数据整理
    
    将多个样本整理成一个批次
    输出: images (B, C, H, W), labels (B,)
    
    Args:
        data: 输入数据迭代器，每个元素是一个样本列表
    
    Yields:
        批数据字典，包含 record_ids, images, labels
    """
    for batch_list in data:
        if not batch_list:
            continue
        
        record_ids = [item.get('record_id', '') for item in batch_list]
        images = [item['image'] for item in batch_list if item.get('image') is not None]
        
        if images:
            images = torch.stack(images, dim=0)  # (B, C, H, W)
        else:
            images = None
        
        labels = torch.tensor([item.get('label', -1) for item in batch_list], dtype=torch.long)
        
        yield {
            'record_ids': record_ids,
            'images': images,
            'labels': labels,
        }
