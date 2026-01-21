#!/usr/bin/env python
"""
图像数据准备脚本
将图像数据集转换为jsonl格式，用于训练、验证和测试
数据集结构：class_name/image_files (扁平结构)
"""
import os
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

from PIL import Image


def get_image_info(image_path: str) -> dict:
    """获取图像文件的元信息"""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            mode_to_channels = {'L': 1, 'RGB': 3, 'RGBA': 4, 'P': 1, 'CMYK': 4, '1': 1, 'I': 1, 'F': 1}
            channels = mode_to_channels.get(img.mode, 3)
            return {'width': width, 'height': height, 'channels': channels}
    except Exception as e:
        print(f"Warning: Failed to read {image_path}: {e}")
        return {'width': 0, 'height': 0, 'channels': 3}


def prepare_data_by_count(
    data_root: str,
    output_dir: str,
    eval_samples_per_class: int = 150,
    test_samples_per_class: int = 150,
    seed: int = 42
):
    """按样本数量划分数据集"""
    print("=" * 60)
    print("Image Data Preparation: Dataset -> JSONL (By Count)")
    print("=" * 60)
    
    random.seed(seed)
    data_root = Path(data_root).resolve()
    output_dir = Path(output_dir).resolve()
    
    # 创建输出目录
    for subdir in ['train', 'eval', 'test']:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # 支持的图像格式
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.webp']
    
    # 获取类别
    classes = sorted([d for d in os.listdir(data_root) if (data_root / d).is_dir()])
    print(f"Data root: {data_root}")
    print(f"Found {len(classes)} classes: {classes}\n")
    
    # 收集每个类别的图像
    class_images = defaultdict(list)
    for class_name in classes:
        class_path = data_root / class_name
        for f in os.listdir(class_path):
            if any(f.lower().endswith(ext) for ext in image_exts):
                class_images[class_name].append(str(class_path / f))
        print(f"{class_name}: {len(class_images[class_name])} images")
    
    # 划分数据集
    train_records, eval_records, test_records = [], [], []
    record_id = 0
    
    for class_name in classes:
        images = class_images[class_name]
        random.shuffle(images)
        
        n_test = min(test_samples_per_class, len(images))
        n_eval = min(eval_samples_per_class, len(images) - n_test)
        
        test_imgs = images[:n_test]
        eval_imgs = images[n_test:n_test + n_eval]
        train_imgs = images[n_test + n_eval:]
        
        print(f"\n{class_name}: train={len(train_imgs)}, eval={len(eval_imgs)}, test={len(test_imgs)}")
        
        # 生成记录
        for split, imgs in [('train', train_imgs), ('eval', eval_imgs), ('test', test_imgs)]:
            for img_path in imgs:
                info = get_image_info(img_path)
                record = {
                    'image_id': f"{split}_{record_id:06d}",
                    'image_path': img_path,
                    'modality': 'image',
                    'width': info['width'],
                    'height': info['height'],
                    'label': class_name,
                    'channels': info['channels']
                }
                if split == 'train':
                    train_records.append(record)
                elif split == 'eval':
                    eval_records.append(record)
                else:
                    test_records.append(record)
                record_id += 1
    
    # 打乱并保存
    random.shuffle(train_records)
    random.shuffle(eval_records)
    random.shuffle(test_records)
    
    _save_records(train_records, eval_records, test_records, output_dir, classes)


def prepare_data_by_ratio(
    data_root: str,
    output_dir: str,
    train_ratio: float = 0.7,
    eval_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
):
    """按比例划分数据集"""
    print("=" * 60)
    print("Image Data Preparation: Dataset -> JSONL (By Ratio)")
    print("=" * 60)
    
    if abs(train_ratio + eval_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0")
    
    random.seed(seed)
    data_root = Path(data_root).resolve()
    output_dir = Path(output_dir).resolve()
    
    # 创建输出目录
    for subdir in ['train', 'eval', 'test']:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # 支持的图像格式
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.webp']
    
    # 获取类别
    classes = sorted([d for d in os.listdir(data_root) if (data_root / d).is_dir()])
    print(f"Data root: {data_root}")
    print(f"Found {len(classes)} classes: {classes}")
    print(f"Ratios: train={train_ratio:.2f}, eval={eval_ratio:.2f}, test={test_ratio:.2f}\n")
    
    # 收集每个类别的图像
    class_images = defaultdict(list)
    for class_name in classes:
        class_path = data_root / class_name
        for f in os.listdir(class_path):
            if any(f.lower().endswith(ext) for ext in image_exts):
                class_images[class_name].append(str(class_path / f))
        print(f"{class_name}: {len(class_images[class_name])} images")
    
    # 划分数据集
    train_records, eval_records, test_records = [], [], []
    record_id = 0
    
    for class_name in classes:
        images = class_images[class_name]
        random.shuffle(images)
        
        n_total = len(images)
        n_test = int(n_total * test_ratio)
        n_eval = int(n_total * eval_ratio)
        
        test_imgs = images[:n_test]
        eval_imgs = images[n_test:n_test + n_eval]
        train_imgs = images[n_test + n_eval:]
        
        print(f"\n{class_name}: train={len(train_imgs)}, eval={len(eval_imgs)}, test={len(test_imgs)}")
        
        # 生成记录
        for split, imgs in [('train', train_imgs), ('eval', eval_imgs), ('test', test_imgs)]:
            for img_path in imgs:
                info = get_image_info(img_path)
                record = {
                    'image_id': f"{split}_{record_id:06d}",
                    'image_path': img_path,
                    'modality': 'image',
                    'width': info['width'],
                    'height': info['height'],
                    'label': class_name,
                    'channels': info['channels']
                }
                if split == 'train':
                    train_records.append(record)
                elif split == 'eval':
                    eval_records.append(record)
                else:
                    test_records.append(record)
                record_id += 1
    
    # 打乱并保存
    random.shuffle(train_records)
    random.shuffle(eval_records)
    random.shuffle(test_records)
    
    _save_records(train_records, eval_records, test_records, output_dir, classes)


def _save_records(train_records, eval_records, test_records, output_dir, classes):
    """保存记录到文件"""
    # 保存jsonl文件
    for split, records in [('train', train_records), ('eval', eval_records), ('test', test_records)]:
        jsonl_path = output_dir / split / 'data.jsonl'
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    # 保存元信息
    meta_info = {
        'classes': classes,
        'class_to_label': {cls: idx for idx, cls in enumerate(classes)},
        'num_classes': len(classes),
        'train_samples': len(train_records),
        'eval_samples': len(eval_records),
        'test_samples': len(test_records),
        'total_samples': len(train_records) + len(eval_records) + len(test_records)
    }
    
    with open(output_dir / 'meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta_info, f, indent=2, ensure_ascii=False)
    
    # 打印统计信息
    print(f"\n" + "=" * 60)
    print("Output Summary")
    print("=" * 60)
    print(f"Train: {len(train_records)} samples -> {output_dir}/train/data.jsonl")
    print(f"Eval:  {len(eval_records)} samples -> {output_dir}/eval/data.jsonl")
    print(f"Test:  {len(test_records)} samples -> {output_dir}/test/data.jsonl")
    print(f"Meta:  {output_dir}/meta.json")
    print(f"\nClass mapping: {meta_info['class_to_label']}")
    
    # 每类分布
    print(f"\nPer-class distribution:")
    for cls in classes:
        train_count = sum(1 for r in train_records if r['label'] == cls)
        eval_count = sum(1 for r in eval_records if r['label'] == cls)
        test_count = sum(1 for r in test_records if r['label'] == cls)
        print(f"  {cls}: train={train_count}, eval={eval_count}, test={test_count}")


def main():
    parser = argparse.ArgumentParser(description='Convert image dataset to JSONL format')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Image dataset root directory')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Output directory for JSONL files')
    parser.add_argument('--mode', type=str, choices=['count', 'ratio'], default='count',
                        help='Split mode: "count" (by sample count) or "ratio" (by percentage)')
    
    # 按数量划分的参数
    parser.add_argument('--eval_samples_per_class', type=int, default=150,
                        help='Target number of eval samples per class (for count mode, default: 150)')
    parser.add_argument('--test_samples_per_class', type=int, default=150,
                        help='Target number of test samples per class (for count mode, default: 150)')
    
    # 按比例划分的参数
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Train set ratio (for ratio mode, default: 0.7)')
    parser.add_argument('--eval_ratio', type=float, default=0.15,
                        help='Eval set ratio (for ratio mode, default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Test set ratio (for ratio mode, default: 0.15)')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    if args.mode == 'count':
        prepare_data_by_count(
            data_root=args.data_root,
            output_dir=args.output_dir,
            eval_samples_per_class=args.eval_samples_per_class,
            test_samples_per_class=args.test_samples_per_class,
            seed=args.seed
        )
    else:  # ratio mode
        prepare_data_by_ratio(
            data_root=args.data_root,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            eval_ratio=args.eval_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed
        )


if __name__ == '__main__':
    main()
