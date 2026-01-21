"""
音频处理器模块
用于音频分类任务的数据处理和增强

默认输出: 80 维 FBank 特征，形状 (T, 80)
"""
import json
import random
from typing import Dict, Any, Iterator, List

import torch

from FunFlow.logger import get_logger

# 模块级 logger，所有函数共享
logger = get_logger('FunDiagnosis')

try:
    import torchaudio
    import torchaudio.compliance.kaldi as Kaldi
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    logger.warning("torchaudio not available")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa not available")


# ============================================================
# 默认配置
# ============================================================

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_NUM_MEL_BINS = 80

# ============================================================
# 核心处理函数
# ============================================================

def load_audio(data: Iterator, target_sr: int = DEFAULT_SAMPLE_RATE) -> Iterator:
    """加载音频文件"""
    if not TORCHAUDIO_AVAILABLE:
        logger.error("torchaudio not available")
        return
    
    for sample in data:
        audio_path = sample.get('audio_path')
        if not audio_path:
            yield sample
            continue
        
        try:
            waveform, sr = torchaudio.load(audio_path)
            if sr != target_sr:
                waveform = torchaudio.functional.resample(waveform, sr, target_sr)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)
            else:
                waveform = waveform.squeeze(0)
            
            sample['waveform'] = waveform
            sample['sample_rate'] = target_sr
            yield sample
        except Exception as e:
            logger.warning(f"Failed to load {audio_path}: {e}")


def filter_short_audio(
    data: Iterator,
    min_duration: float = 0.5,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> Iterator:
    """过滤掉太短的音频
    
    Args:
        data: 输入数据迭代器
        min_duration: 最小音频时长（秒），默认 0.5 秒
        sample_rate: 采样率，用于计算时长
    
    Yields:
        时长大于等于 min_duration 的样本
    """
    min_samples = int(min_duration * sample_rate)
    
    for sample in data:
        waveform = sample.get('waveform')
        if waveform is None:
            yield sample
            continue
        
        # 计算音频长度
        audio_len = waveform.shape[-1] if waveform.dim() > 0 else 0
        
        if audio_len >= min_samples:
            yield sample
        else:
            duration = audio_len / sample_rate
            logger.debug(f"Skipping short audio: {sample.get('audio_path', 'unknown')} "
                         f"(duration={duration:.3f}s < {min_duration}s)")


def compute_fbank(
    data: Iterator,
    num_mel_bins: int = DEFAULT_NUM_MEL_BINS,
    frame_length: float = 25.0,
    frame_shift: float = 10.0,
    dither: float = 0.0,
    style: str = 'librosa',
) -> Iterator:
    """计算 FBank 特征
    
    Args:
        data: 输入数据迭代器
        num_mel_bins: Mel 滤波器组数量
        frame_length: 帧长（毫秒）
        frame_shift: 帧移（毫秒）
        dither: dither 值（仅用于 kaldi style）
        style: 'librosa' 或 'kaldi'，默认 'librosa'
            - 'librosa': 使用纯 librosa 库（训练推理一致）
            - 'kaldi': 使用 kaldi 兼容实现
    
    Yields:
        包含 'feat' 字段的样本，形状 (T, num_mel_bins)
    """
    if not TORCHAUDIO_AVAILABLE:
        logger.error("torchaudio not available")
        return
    
    for sample in data:
        waveform = sample.get('waveform')
        if waveform is None:
            yield sample
            continue
        
        sr = sample.get('sample_rate', DEFAULT_SAMPLE_RATE)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        try:
            if style == 'kaldi':
                # Kaldi style: 使用 torchaudio.compliance.kaldi
                fbank = Kaldi.fbank(
                    waveform,
                    num_mel_bins=num_mel_bins,
                    frame_length=frame_length,
                    frame_shift=frame_shift,
                    dither=dither,
                    energy_floor=1.0,
                    sample_frequency=float(sr),
                    window_type='hamming',
                    htk_compat=True,
                    use_energy=False,
                    use_log_fbank=True,
                    use_power=True,
                )
            else:
                # Librosa style: 使用纯 librosa 库（训练推理一致）
                if not LIBROSA_AVAILABLE:
                    logger.error("librosa not available for librosa style")
                    yield sample
                    continue
                
                # 转换为 numpy
                waveform_np = waveform.squeeze().numpy() if isinstance(waveform, torch.Tensor) else waveform
                
                n_fft = int(sr * frame_length / 1000)
                hop_length = int(sr * frame_shift / 1000)
                win_length = n_fft
                
                # 使用 librosa 计算 mel spectrogram
                mel_spec = librosa.feature.melspectrogram(
                    y=waveform_np,
                    sr=sr,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    n_mels=num_mel_bins,
                    window='hamming',
                )
                # 转换为 log scale
                log_mel = librosa.power_to_db(mel_spec, ref=1.0)
                # 转换形状为 (T, n_mels)
                fbank = torch.from_numpy(log_mel.T).float()
            
            sample['feat'] = fbank  # (T, num_mel_bins)
            yield sample
        except Exception as e:
            logger.warning(f"Failed to compute fbank: {e}")


def compute_mel_spectrogram(
    data: Iterator,
    n_fft: int = 400,
    win_length: int = 400,
    hop_length: int = 160,
    n_mels: int = DEFAULT_NUM_MEL_BINS,
    normalized: bool = False,
) -> Iterator:
    """计算 Mel Spectrogram 特征
    
    Args:
        data: 输入数据迭代器
        n_fft: FFT 窗口大小
        win_length: 窗口长度
        hop_length: 帧移
        n_mels: Mel 滤波器组数量
        normalized: 是否归一化
    
    Yields:
        包含 'feat' 字段的样本，形状 (T, n_mels)
    """
    if not TORCHAUDIO_AVAILABLE:
        logger.error("torchaudio not available")
        return
    
    for sample in data:
        waveform = sample.get('waveform')
        if waveform is None:
            yield sample
            continue
        
        sr = sample.get('sample_rate', DEFAULT_SAMPLE_RATE)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        try:
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sr,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                n_mels=n_mels,
                normalized=normalized,
            )
            mel_spec = mel_transform(waveform)  # (1, n_mels, T)
            # 转换为 log scale
            mel_spec = torch.log(mel_spec + 1e-9)
            # 调整形状为 (T, n_mels)
            mel_spec = mel_spec.squeeze(0).transpose(0, 1)
            sample['feat'] = mel_spec
            yield sample
        except Exception as e:
            logger.warning(f"Failed to compute mel spectrogram: {e}")


def compute_mfcc(
    data: Iterator,
    n_mfcc: int = 13,
    n_fft: int = 400,
    win_length: int = 400,
    hop_length: int = 160,
    n_mels: int = 23,
    log_mels: bool = False,
) -> Iterator:
    """计算 MFCC 特征
    
    Args:
        data: 输入数据迭代器
        n_mfcc: MFCC 系数数量
        n_fft: FFT 窗口大小
        win_length: 窗口长度
        hop_length: 帧移
        n_mels: Mel 滤波器组数量
        log_mels: 是否使用 log-mel spectrogram
    
    Yields:
        包含 'feat' 字段的样本，形状 (T, n_mfcc)
    """
    if not TORCHAUDIO_AVAILABLE:
        logger.error("torchaudio not available")
        return
    
    for sample in data:
        waveform = sample.get('waveform')
        if waveform is None:
            yield sample
            continue
        
        sr = sample.get('sample_rate', DEFAULT_SAMPLE_RATE)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        try:
            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=sr,
                n_mfcc=n_mfcc,
                log_mels=log_mels,
                melkwargs={
                    'n_fft': n_fft,
                    'win_length': win_length,
                    'hop_length': hop_length,
                    'n_mels': n_mels,
                },
            )
            mfcc = mfcc_transform(waveform)  # (1, n_mfcc, T)
            # 调整形状为 (T, n_mfcc)
            mfcc = mfcc.squeeze(0).transpose(0, 1)
            sample['feat'] = mfcc
            yield sample
        except Exception as e:
            logger.warning(f"Failed to compute MFCC: {e}")


# ============================================================
# 数据增强
# ============================================================

def speed_perturb(data: Iterator, speeds: List[float] = None, prob: float = 0.5) -> Iterator:
    """速度扰动"""
    speeds = speeds or [0.9, 1.1]
    for sample in data:
        waveform = sample.get('waveform')
        if waveform is None or random.random() > prob:
            yield sample
            continue
        
        speed = random.uniform(speeds[0], speeds[1])
        if abs(speed - 1.0) < 1e-6:
            yield sample
            continue
        
        new_len = int(len(waveform) / speed)
        wav = waveform.unsqueeze(0).unsqueeze(0)
        resampled = torch.nn.functional.interpolate(wav, size=new_len, mode='linear', align_corners=False)
        sample['waveform'] = resampled.squeeze()
        yield sample


def add_noise(data: Iterator, noise_level: float = 0.005, prob: float = 0.5) -> Iterator:
    """添加高斯噪声"""
    for sample in data:
        waveform = sample.get('waveform')
        if waveform is None or random.random() > prob:
            yield sample
            continue
        sample['waveform'] = waveform + torch.randn_like(waveform) * noise_level
        yield sample


def spec_augment(data: Iterator, time_mask_num: int = 2, time_mask_width: int = 20,
                 freq_mask_num: int = 2, freq_mask_width: int = 10, prob: float = 0.5) -> Iterator:
    """SpecAugment 数据增强"""
    for sample in data:
        feat = sample.get('feat')
        if feat is None or random.random() > prob:
            yield sample
            continue
        
        feat = feat.clone()
        T, D = feat.shape
        
        for _ in range(time_mask_num):
            if T > 1:
                t = random.randint(0, T - 1)
                w = min(random.randint(1, time_mask_width), T - t)
                feat[t:t + w, :] = 0
        
        for _ in range(freq_mask_num):
            if D > 1:
                f = random.randint(0, D - 1)
                w = min(random.randint(1, freq_mask_width), D - f)
                feat[:, f:f + w] = 0
        
        sample['feat'] = feat
        yield sample


# ============================================================
# 数据解析和整理
# ============================================================

def parse_raw(data: Iterator, label_map: Dict[str, int] = None) -> Iterator:
    """解析 JSONL 原始数据"""
    label_map = label_map
    
    for sample in data:
        raw = sample.get('raw')
        if raw is None:
            yield sample
            continue
        
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                continue
        
        parsed = {
            'record_id': raw.get('record_id', raw.get('id', '')),
            'audio_path': raw.get('audio_path', raw.get('path', '')),
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
    """批数据整理: 输出 feats (B, T, D), feat_lens (B,), labels (B,)"""
    for batch_list in data:
        if not batch_list:
            continue
        
        record_ids = [item.get('record_id', '') for item in batch_list]
        feats, feat_lens = [], []
        
        for item in batch_list:
            feat = item.get('feat')
            if feat is not None:
                feats.append(feat)
                feat_lens.append(feat.shape[0])
        
        if feats:
            max_len = max(feat_lens)
            padded = []
            for feat in feats:
                if feat.shape[0] < max_len:
                    pad = torch.zeros(max_len - feat.shape[0], feat.shape[1], dtype=feat.dtype)
                    feat = torch.cat([feat, pad], dim=0)
                padded.append(feat)
            feats = torch.stack(padded, dim=0)
            feat_lens = torch.tensor(feat_lens, dtype=torch.long)
        else:
            feats, feat_lens = None, None
        
        labels = torch.tensor([item.get('label', -1) for item in batch_list], dtype=torch.long)
        
        yield {
            'record_ids': record_ids,
            'feats': feats,
            'feat_lens': feat_lens,
            'labels': labels,
        }
