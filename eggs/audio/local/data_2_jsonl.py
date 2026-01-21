#!/usr/bin/env python
"""
数据准备脚本
将silero_sliced数据集转换为jsonl格式，用于训练、验证和测试
按患者划分数据集，避免数据泄露（同一患者的所有音频只能在训练集、验证集或测试集中）
"""
import os
import sys
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

import torchaudio


def get_audio_info(audio_path: str) -> dict:
    """
    获取音频文件的元信息
    
    Args:
        audio_path: 音频文件路径
        
    Returns:
        包含采样率、时长、通道数的字典
    """
    try:
        info = torchaudio.info(audio_path)
        return {
            'sampling_rate': info.sample_rate,
            'duration': info.num_frames / info.sample_rate,
            'channels': info.num_channels
        }
    except Exception as e:
        print(f"Warning: Failed to read audio info from {audio_path}: {e}")
        return {
            'sampling_rate': 16000,
            'duration': 0.0,
            'channels': 1
        }


def prepare_data(
    data_root: str,
    output_dir: str,
    eval_samples_per_class: int = 150,
    test_samples_per_class: int = 150,
    seed: int = 42
):
    """
    准备训练、验证和测试数据，生成jsonl文件
    按患者划分数据集，避免数据泄露
    
    Args:
        data_root: silero_sliced数据集根目录
        output_dir: 输出目录
        eval_samples_per_class: 每个类别验证集的目标样本数
        test_samples_per_class: 每个类别测试集的目标样本数
        seed: 随机种子
    """
    print("=" * 60)
    print("Data Preparation: silero_sliced -> JSONL")
    print("=" * 60)
    
    random.seed(seed)
    
    data_root = Path(data_root).resolve()  # 转换为绝对路径
    output_dir = Path(output_dir).resolve()  # 转换为绝对路径
    
    # 创建输出目录
    train_dir = output_dir / 'train'
    eval_dir = output_dir / 'eval'
    test_dir = output_dir / 'test'
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # 支持的音频格式
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    
    # 获取类别（Healthy, Parkinson）
    classes = sorted([d for d in os.listdir(data_root) if os.path.isdir(data_root / d)])
    
    print(f"Data root: {data_root}")
    print(f"Found {len(classes)} classes: {classes}")
    
    # 类别到标签的映射
    class_to_label = {cls: idx for idx, cls in enumerate(classes)}
    
    # 收集每个类别下的患者及其音频文件
    # 结构: {class_name: {patient_id: [audio_paths]}}
    class_patient_data = defaultdict(lambda: defaultdict(list))
    
    for class_name in classes:
        class_path = data_root / class_name
        # 遍历患者文件夹
        for patient_dir in os.listdir(class_path):
            patient_path = class_path / patient_dir
            if not os.path.isdir(patient_path):
                continue
            
            # 收集该患者的所有音频文件
            for filename in os.listdir(patient_path):
                if any(filename.lower().endswith(ext) for ext in audio_extensions):
                    audio_path = str(patient_path / filename)
                    class_patient_data[class_name][patient_dir].append(audio_path)
    
    # 打印统计信息
    print(f"\nPatient and sample statistics:")
    for class_name in classes:
        patients = class_patient_data[class_name]
        total_samples = sum(len(files) for files in patients.values())
        print(f"  {class_name}: {len(patients)} patients, {total_samples} audio samples")
    
    # 按患者划分数据集
    train_records = []
    eval_records = []
    test_records = []
    record_id = 0
    
    train_patients = defaultdict(list)
    eval_patients = defaultdict(list)
    test_patients = defaultdict(list)
    
    for class_name in classes:
        patients = class_patient_data[class_name]
        patient_ids = list(patients.keys())
        random.shuffle(patient_ids)
        
        # 选择测试集患者，直到音频数量接近目标
        test_audio_count = 0
        test_patient_set = []
        
        for patient_id in patient_ids:
            patient_audios = patients[patient_id]
            if test_audio_count < test_samples_per_class:
                test_patient_set.append(patient_id)
                test_audio_count += len(patient_audios)
            else:
                break
        
        # 从剩余患者中选择验证集患者
        remaining_patients = [p for p in patient_ids if p not in test_patient_set]
        eval_audio_count = 0
        eval_patient_set = []
        
        for patient_id in remaining_patients:
            patient_audios = patients[patient_id]
            if eval_audio_count < eval_samples_per_class:
                eval_patient_set.append(patient_id)
                eval_audio_count += len(patient_audios)
            else:
                break
        
        # 剩余患者作为训练集
        train_patient_set = [p for p in patient_ids if p not in test_patient_set and p not in eval_patient_set]
        
        train_patients[class_name] = train_patient_set
        eval_patients[class_name] = eval_patient_set
        test_patients[class_name] = test_patient_set
        
        print(f"\n{class_name}:")
        print(f"  Test patients: {len(test_patient_set)}, ~{test_audio_count} samples")
        print(f"  Eval patients: {len(eval_patient_set)}, ~{eval_audio_count} samples")
        print(f"  Train patients: {len(train_patient_set)}")
        
        # 生成训练集记录
        for patient_id in train_patient_set:
            for audio_path in patients[patient_id]:
                audio_info = get_audio_info(audio_path)
                
                record = {
                    'record_id': f"train_{record_id:06d}",
                    'patient_id': patient_id,
                    'audio_path': audio_path,
                    'modality': 'audio',
                    'audio_type': None,
                    'sampling_rate': audio_info['sampling_rate'],
                    'duration': audio_info['duration'],
                    'transcription': None,
                    'label': class_name,
                    'vad_segments': None,
                    'channels': audio_info['channels']
                }
                train_records.append(record)
                record_id += 1
        
        # 生成验证集记录
        for patient_id in eval_patient_set:
            for audio_path in patients[patient_id]:
                audio_info = get_audio_info(audio_path)
                
                record = {
                    'record_id': f"eval_{record_id:06d}",
                    'patient_id': patient_id,
                    'audio_path': audio_path,
                    'modality': 'audio',
                    'audio_type': None,
                    'sampling_rate': audio_info['sampling_rate'],
                    'duration': audio_info['duration'],
                    'transcription': None,
                    'label': class_name,
                    'vad_segments': None,
                    'channels': audio_info['channels']
                }
                eval_records.append(record)
                record_id += 1
        
        # 生成测试集记录
        for patient_id in test_patient_set:
            for audio_path in patients[patient_id]:
                audio_info = get_audio_info(audio_path)
                
                record = {
                    'record_id': f"test_{record_id:06d}",
                    'patient_id': patient_id,
                    'audio_path': audio_path,
                    'modality': 'audio',
                    'audio_type': None,
                    'sampling_rate': audio_info['sampling_rate'],
                    'duration': audio_info['duration'],
                    'transcription': None,
                    'label': class_name,
                    'vad_segments': None,
                    'channels': audio_info['channels']
                }
                test_records.append(record)
                record_id += 1
    
    # 打乱记录顺序
    random.shuffle(train_records)
    random.shuffle(eval_records)
    random.shuffle(test_records)
    
    # 保存jsonl文件
    train_jsonl_path = train_dir / 'data.jsonl'
    eval_jsonl_path = eval_dir / 'data.jsonl'
    test_jsonl_path = test_dir / 'data.jsonl'
    
    with open(train_jsonl_path, 'w', encoding='utf-8') as f:
        for record in train_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    with open(eval_jsonl_path, 'w', encoding='utf-8') as f:
        for record in eval_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    with open(test_jsonl_path, 'w', encoding='utf-8') as f:
        for record in test_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    # 统计每类的样本数
    train_class_count = defaultdict(int)
    eval_class_count = defaultdict(int)
    test_class_count = defaultdict(int)
    
    for r in train_records:
        train_class_count[r['label']] += 1
    
    for r in eval_records:
        eval_class_count[r['label']] += 1
    
    for r in test_records:
        test_class_count[r['label']] += 1
    
    # 保存元信息
    meta_info = {
        'classes': classes,
        'class_to_label': class_to_label,
        'num_classes': len(classes),
        'train_samples': len(train_records),
        'eval_samples': len(eval_records),
        'test_samples': len(test_records),
        'total_samples': len(train_records) + len(eval_records) + len(test_records),
        'train_class_samples': dict(train_class_count),
        'eval_class_samples': dict(eval_class_count),
        'test_class_samples': dict(test_class_count),
        'train_patients': {k: len(v) for k, v in train_patients.items()},
        'eval_patients': {k: len(v) for k, v in eval_patients.items()},
        'test_patients': {k: len(v) for k, v in test_patients.items()}
    }
    
    meta_path = output_dir / 'meta.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta_info, f, indent=2, ensure_ascii=False)
    
    print(f"\n" + "=" * 60)
    print("Output Summary")
    print("=" * 60)
    print(f"Train set: {len(train_records)} samples -> {train_jsonl_path}")
    print(f"Eval set: {len(eval_records)} samples -> {eval_jsonl_path}")
    print(f"Test set: {len(test_records)} samples -> {test_jsonl_path}")
    print(f"Meta info: {meta_path}")
    print(f"\nClass mapping: {class_to_label}")
    
    # 打印每类的训练/验证/测试分布
    print(f"\nPer-class distribution:")
    for cls in classes:
        print(f"  {cls}: train={train_class_count[cls]}, eval={eval_class_count[cls]}, test={test_class_count[cls]}")
    
    # 打印患者分布
    print(f"\nPatient distribution:")
    for cls in classes:
        print(f"  {cls}: train_patients={len(train_patients[cls])}, eval_patients={len(eval_patients[cls])}, test_patients={len(test_patients[cls])}")
    
    return meta_info


def main():
    parser = argparse.ArgumentParser(description='Convert silero_sliced dataset to JSONL format')
    parser.add_argument('--data_root', type=str, default='./download/silero_sliced',
                        help='silero_sliced dataset root directory')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Output directory for JSONL files')
    parser.add_argument('--eval_samples_per_class', type=int, default=150,
                        help='Target number of eval samples per class (default: 150)')
    parser.add_argument('--test_samples_per_class', type=int, default=150,
                        help='Target number of test samples per class (default: 150)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    prepare_data(
        data_root=args.data_root,
        output_dir=args.output_dir,
        eval_samples_per_class=args.eval_samples_per_class,
        test_samples_per_class=args.test_samples_per_class,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
