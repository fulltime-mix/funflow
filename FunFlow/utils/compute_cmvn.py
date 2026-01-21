#!/usr/bin/env python
"""Compute CMVN (Cepstral Mean and Variance Normalization) statistics

Usage:
    python compute_cmvn.py --data_jsonl ./data/train/data.jsonl --output_file ./data/cmvn.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import numpy as np
from tqdm import tqdm

from FunFlow.logger import get_logger

logger = get_logger("FunFlow")

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


DEFAULT_SAMPLE_RATE = 16000
DEFAULT_NUM_MEL_BINS = 80


def load_audio(
    audio_path: str, target_sr: int = DEFAULT_SAMPLE_RATE
) -> Optional[torch.Tensor]:
    """Load audio file

    Args:
        audio_path: Audio file path
        target_sr: Target sample rate

    Returns:
        Audio waveform tensor, None if failed
    """
    try:
        waveform, sr = torchaudio.load(audio_path)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)
        return waveform
    except Exception as e:
        logger.warning(f"Failed to load {audio_path}: {e}")
        return None


def compute_fbank(
    waveform: torch.Tensor,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    num_mel_bins: int = DEFAULT_NUM_MEL_BINS,
    frame_length: float = 25.0,
    frame_shift: float = 10.0,
    dither: float = 0.0,
    style: str = "librosa",
) -> Optional[torch.Tensor]:
    """Compute FBank features

    Args:
        waveform: Audio waveform
        sample_rate: Sample rate
        num_mel_bins: Number of mel filter banks
        frame_length: Frame length (ms)
        frame_shift: Frame shift (ms)
        dither: Dither value (kaldi style only)
        style: 'librosa' or 'kaldi', default 'librosa'

    Returns:
        FBank features, shape (T, num_mel_bins)
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    try:
        if style == "kaldi":
            fbank = Kaldi.fbank(
                waveform,
                num_mel_bins=num_mel_bins,
                frame_length=frame_length,
                frame_shift=frame_shift,
                dither=dither,
                energy_floor=1.0,
                sample_frequency=float(sample_rate),
                window_type="hamming",
                htk_compat=True,
                use_energy=False,
                use_log_fbank=True,
                use_power=True,
            )
        else:
            if not LIBROSA_AVAILABLE:
                logger.error("librosa not available for librosa style")
                return None

            waveform_np = (
                waveform.squeeze().numpy()
                if isinstance(waveform, torch.Tensor)
                else waveform
            )

            n_fft = int(sample_rate * frame_length / 1000)
            hop_length = int(sample_rate * frame_shift / 1000)
            win_length = n_fft

            mel_spec = librosa.feature.melspectrogram(
                y=waveform_np,
                sr=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                n_mels=num_mel_bins,
                window="hamming",
            )
            log_mel = librosa.power_to_db(mel_spec, ref=1.0)
            fbank = torch.from_numpy(log_mel.T).float()

        return fbank  # (T, num_mel_bins)
    except Exception as e:
        logger.warning(f"Failed to compute fbank: {e}")
        return None


def compute_cmvn(
    data_jsonl: str,
    output_file: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    num_mel_bins: int = DEFAULT_NUM_MEL_BINS,
    frame_length: float = 25.0,
    frame_shift: float = 10.0,
    max_samples: int = -1,
    style: str = "librosa",
):
    """Compute CMVN statistics using Welford's online algorithm

    Args:
        data_jsonl: Training data jsonl file path
        output_file: CMVN output file path
        sample_rate: Sample rate
        num_mel_bins: Number of mel filter banks
        frame_length: Frame length (ms)
        frame_shift: Frame shift (ms)
        max_samples: Max samples to process, -1 for all
        style: 'librosa' or 'kaldi', default 'librosa'
    """
    if not TORCHAUDIO_AVAILABLE:
        raise RuntimeError("torchaudio is required but not available")

    print("=" * 60)
    print("CMVN Computation")
    print("=" * 60)
    print(f"Input: {data_jsonl}")
    print(f"Output: {output_file}")
    print(f"Sample rate: {sample_rate}")
    print(f"Num mel bins: {num_mel_bins}")
    print(f"Frame length: {frame_length}ms")
    print(f"Frame shift: {frame_shift}ms")
    print("=" * 60)

    data_jsonl = Path(data_jsonl).resolve()
    output_file = Path(output_file).resolve()

    samples = []
    with open(data_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line.strip()))

    if max_samples > 0:
        samples = samples[:max_samples]

    print(f"Total samples: {len(samples)}")

    n = 0
    mean = torch.zeros(num_mel_bins, dtype=torch.float64)
    M2 = torch.zeros(num_mel_bins, dtype=torch.float64)

    # 同时记录最大值和最小值（可选，用于调试）
    feat_min = torch.full((num_mel_bins,), float("inf"), dtype=torch.float64)
    feat_max = torch.full((num_mel_bins,), float("-inf"), dtype=torch.float64)

    failed_count = 0
    total_frames = 0

    for sample in tqdm(samples, desc="Processing"):
        audio_path = sample.get("audio_path")
        if not audio_path or not os.path.exists(audio_path):
            failed_count += 1
            continue

        waveform = load_audio(audio_path, sample_rate)
        if waveform is None:
            failed_count += 1
            continue

        fbank = compute_fbank(
            waveform,
            sample_rate=sample_rate,
            num_mel_bins=num_mel_bins,
            frame_length=frame_length,
            frame_shift=frame_shift,
            style=style,
        )
        if fbank is None:
            failed_count += 1
            continue

        fbank = fbank.to(torch.float64)

        for frame in fbank:
            n += 1
            delta = frame - mean
            mean += delta / n
            delta2 = frame - mean
            M2 += delta * delta2

            feat_min = torch.minimum(feat_min, frame)
            feat_max = torch.maximum(feat_max, frame)

        total_frames += fbank.shape[0]

    if n < 2:
        raise ValueError("Not enough data to compute CMVN statistics")

    variance = M2 / n
    std = torch.sqrt(variance)

    std = torch.clamp(std, min=1e-8)

    cmvn_stats = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "variance": variance.tolist(),
        "min": feat_min.tolist(),
        "max": feat_max.tolist(),
        "num_frames": n,
        "num_samples": len(samples) - failed_count,
        "num_failed": failed_count,
        "config": {
            "sample_rate": sample_rate,
            "num_mel_bins": num_mel_bins,
            "frame_length": frame_length,
            "frame_shift": frame_shift,
        },
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cmvn_stats, f, indent=2)

    print("\n" + "=" * 60)
    print("CMVN Statistics Summary")
    print("=" * 60)
    print(f"Total frames: {n}")
    print(f"Successful samples: {len(samples) - failed_count}")
    print(f"Failed samples: {failed_count}")
    print(f"\nMean (first 5 dims): {mean[:5].tolist()}")
    print(f"Std (first 5 dims): {std[:5].tolist()}")
    print(f"Min (first 5 dims): {feat_min[:5].tolist()}")
    print(f"Max (first 5 dims): {feat_max[:5].tolist()}")
    print(f"\nCMVN saved to: {output_file}")

    return cmvn_stats


def main():
    parser = argparse.ArgumentParser(
        description="Compute CMVN statistics from training data"
    )
    parser.add_argument(
        "--data_jsonl", type=str, required=True, help="Path to training data.jsonl file"
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Output CMVN JSON file path"
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help=f"Audio sample rate (default: {DEFAULT_SAMPLE_RATE})",
    )
    parser.add_argument(
        "--num_mel_bins",
        type=int,
        default=DEFAULT_NUM_MEL_BINS,
        help=f"Number of mel filter banks (default: {DEFAULT_NUM_MEL_BINS})",
    )
    parser.add_argument(
        "--frame_length",
        type=float,
        default=25.0,
        help="Frame length in milliseconds (default: 25.0)",
    )
    parser.add_argument(
        "--frame_shift",
        type=float,
        default=10.0,
        help="Frame shift in milliseconds (default: 10.0)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Maximum number of samples to process (-1 for all)",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="librosa",
        choices=["librosa", "kaldi"],
        help="FBank style: librosa or kaldi (default: librosa)",
    )

    args = parser.parse_args()

    compute_cmvn(
        data_jsonl=args.data_jsonl,
        output_file=args.output_file,
        sample_rate=args.sample_rate,
        num_mel_bins=args.num_mel_bins,
        frame_length=args.frame_length,
        frame_shift=args.frame_shift,
        max_samples=args.max_samples,
        style=args.style,
    )


if __name__ == "__main__":
    main()
