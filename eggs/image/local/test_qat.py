#!/usr/bin/env python3
"""
最终修正版：使用ImageClassifier直接加载和测试QAT模型
"""

import sys
sys.path.insert(0, '/data/project/FunDiagnosis/eggs/image')

import torch
import torch.nn as nn
import time
import numpy as np
import json
from pathlib import Path
import onnxruntime as ort
import yaml

import local.model.image_model


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model_from_config(config):
    """通过注册表构建模型"""
    from FunFlow.registry import MODELS
    
    # 从config构建模型，添加quantization=True参数
    model_cfg = config['model'].copy()
    model_cfg['quantization'] = True  # 启用量化支持
    
    model = MODELS.build(model_cfg)
    
    return model


def load_checkpoint(model, checkpoint_path):
    """加载checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # 直接加载state_dict（键名应该完全匹配）
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    
    if missing:
        print(f"   ⚠️  缺失键: {len(missing)} 个")
        if len(missing) <= 3:
            for k in missing:
                print(f"      - {k}")
    if unexpected:
        print(f"   ⚠️  未预期键: {len(unexpected)} 个")
        if len(unexpected) <= 3:
            for k in unexpected:
                print(f"      - {k}")
    
    return model


def prepare_fake_data(batch_size=32, num_batches=100):
    """准备假数据进行测试"""
    data_list = []
    for _ in range(num_batches):
        data = torch.randn(batch_size, 3, 224, 224)
        data_list.append(data)
    return data_list


def benchmark_inference(model, data_list, device='cpu', warmup=10):
    """基准测试推理性能"""
    model.eval()
    model.to(device)
    
    # 预热
    with torch.no_grad():
        for i, data in enumerate(data_list[:warmup]):
            data = data.to(device)
            inputs = {'images': data}
            _ = model(inputs)
    
    # 正式测试
    times = []
    with torch.no_grad():
        for data in data_list:
            data = data.to(device)
            inputs = {'images': data}
            start_time = time.time()
            _ = model(inputs)
            end_time = time.time()
            times.append(end_time - start_time)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'total_time': np.sum(times),
        'throughput': len(data_list) * data_list[0].shape[0] / np.sum(times)
    }


def benchmark_onnx_inference(onnx_path, data_list, warmup=10):
    """基准测试ONNX推理性能"""
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    # 预热
    for i, data in enumerate(data_list[:warmup]):
        data_numpy = data.numpy()
        _ = session.run(None, {input_name: data_numpy})
    
    # 正式测试
    times = []
    for data in data_list:
        data_numpy = data.numpy()
        start_time = time.time()
        _ = session.run(None, {input_name: data_numpy})
        end_time = time.time()
        times.append(end_time - start_time)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'total_time': np.sum(times),
        'throughput': len(data_list) * data_list[0].shape[0] / np.sum(times)
    }


def benchmark_jit_inference(jit_path, data_list, device='cpu', warmup=10):
    """基准测试TorchScript JIT推理性能"""
    model = torch.jit.load(jit_path, map_location=device)
    model.eval()
    
    # 预热
    with torch.no_grad():
        for i, data in enumerate(data_list[:warmup]):
            data = data.to(device)
            _ = model(data)
    
    # 正式测试
    times = []
    with torch.no_grad():
        for data in data_list:
            data = data.to(device)
            start_time = time.time()
            _ = model(data)
            end_time = time.time()
            times.append(end_time - start_time)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'total_time': np.sum(times),
        'throughput': len(data_list) * data_list[0].shape[0] / np.sum(times)
    }


def main():
    # 路径设置
    qat_dir = Path('/data/project/FunDiagnosis/eggs/image/exp/image_classification_test/exp_qat')
    checkpoint_path = qat_dir / 'avg_5.pth'
    config_path = qat_dir / 'config.yaml'
    output_dir = Path('/data/project/FunDiagnosis/eggs/image/local/qat_benchmark_output')
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("QAT模型推理性能测试（最终版 - 使用ImageClassifier）")
    print("=" * 80)
    
    # 加载配置和模型
    print("\n1. 加载配置和QAT模型...")
    config = load_config(config_path)
    model = create_model_from_config(config)

    from FunFlow.compression.QAT import QATQuantizer
    # prepare_qat会插入fake_quant层
    model = QATQuantizer.prepare_qat(
        model,
        backend='fbgemm',
        fuse_modules=True,
        fusion_strategy='resnet',
        inplace=True
    )

    model = load_checkpoint(model, checkpoint_path)
    model.eval()
    print("   ✓ 模型加载完成")
    
    # 检查模型结构
    print("\n   模型结构检查:")
    print(f"   - 模型类型: {type(model).__name__}")
    print(f"   - Backbone类型: {type(model.backbone).__name__}")
    print(f"   - Head类型: {type(model.head).__name__}")
    
    # 准备测试数据
    print("\n2. 准备测试数据...")
    batch_size = 32
    num_batches = 100
    data_list = prepare_fake_data(batch_size, num_batches)
    print(f"   生成 {num_batches} 批次数据，每批 {batch_size} 张图像")
    
    results = {}
    
    # 测试1: Convert之前的模型推理（QAT模式）
    print("\n3. 测试Convert之前的QAT模型推理...")
    print("   使用训练时的QAT模型（带伪量化层FakeQuantize）...")
    result_before = benchmark_inference(model, data_list, device='cpu', warmup=10)
    results['qat_before_convert'] = result_before
    print(f"   平均推理时间: {result_before['mean_time']*1000:.2f} ms")
    print(f"   吞吐量: {result_before['throughput']:.2f} images/sec")
    
    # 测试2: Convert后的PyTorch INT8量化模型
    print("\n4. Convert量化模型（转为真正的INT8）...")
    
    # 重新加载模型并prepare QAT
    model_to_quantize = create_model_from_config(config)
    
    # 关键步骤：使用QATQuantizer.prepare_qat重建QAT结构
    print("   准备QAT模型结构（插入FakeQuantize层）...")
    from FunFlow.compression.QAT import QATQuantizer
    
    # prepare_qat会插入fake_quant层
    model_to_quantize = QATQuantizer.prepare_qat(
        model_to_quantize,
        backend='fbgemm',
        fuse_modules=True,
        fusion_strategy='resnet',
        inplace=True
    )
    
    # 现在加载训练好的QAT权重（包括fake_quant的scale/zero_point）
    print("   加载QAT训练后的权重...")
    model_to_quantize = load_checkpoint(model_to_quantize, checkpoint_path)
    model_to_quantize.eval()
    
    # 检查量化准备
    print("   检查模型量化准备情况...")
    fake_quant_layers = []
    for name, module in model_to_quantize.named_modules():
        type_name = type(module).__name__
        if 'FakeQuantize' in type_name or 'FloatFunctional' in type_name:
            fake_quant_layers.append((name, type_name))
    
    print(f"   找到 {len(fake_quant_layers)} 个量化相关层")
    if len(fake_quant_layers) > 0:
        print("   示例:")
        for name, type_name in fake_quant_layers[:5]:
            print(f"     - {name}: {type_name}")
    
    # Convert
    print("\n   执行Convert（FakeQuant -> INT8）...")
    try:
        model_quantized = QATQuantizer.convert_qat(model_to_quantize, inplace=False)
        print("   ✓ Convert成功")
        
        # 检查量化后的层
        quantized_layers = []
        for name, module in model_quantized.named_modules():
            if isinstance(module, (torch.nn.quantized.Linear, torch.nn.quantized.Conv2d)):
                quantized_layers.append((name, type(module).__name__))
        
        print(f"   找到 {len(quantized_layers)} 个INT8量化层")
        if len(quantized_layers) > 0:
            print("   示例:")
            for name, type_name in quantized_layers[:5]:
                print(f"     - {name}: {type_name}")
        else:
            print("   ❌ 警告：Convert后仍然没有找到INT8量化层！")
        
    except Exception as e:
        print(f"   ❌ Convert失败: {e}")
        print(f"   错误类型: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        model_quantized = model_to_quantize
    
    # 保存模型
    quantized_model_path = output_dir / 'quantized_model_full.pth'
    torch.save(model_quantized, quantized_model_path)
    model_size_mb = quantized_model_path.stat().st_size / (1024 * 1024)
    print(f"\n   量化模型已保存到: {quantized_model_path}")
    print(f"   模型文件大小: {model_size_mb:.2f} MB")
    
    # 测试推理性能
    result_pytorch = benchmark_inference(model_quantized, data_list, device='cpu', warmup=10)
    results['pytorch_quantized'] = result_pytorch
    print(f"   平均推理时间: {result_pytorch['mean_time']*1000:.2f} ms")
    print(f"   吞吐量: {result_pytorch['throughput']:.2f} images/sec")
    print(f"   vs QAT前加速: {result_before['mean_time']/result_pytorch['mean_time']:.2f}x")

    # 测试2.5：保存convert后的模型权重，重新创建模型实例，然后prepare_qat + convert，然后再去加载刚刚保存的convert以后的权重，进行验证。
    print("\n4.5. 测试Convert后权重的重新加载能力...")
    print("   保存convert后的模型权重...")
    
    # 保存convert后的state_dict
    quantized_state_dict_path = output_dir / 'quantized_state_dict.pth'
    torch.save(model_quantized.state_dict(), quantized_state_dict_path)
    state_dict_size_mb = quantized_state_dict_path.stat().st_size / (1024 * 1024)
    print(f"   权重已保存到: {quantized_state_dict_path}")
    print(f"   权重文件大小: {state_dict_size_mb:.2f} MB")
    
    # 重新创建模型并加载权重
    print("\n   重新创建模型实例...")
    model_reload = create_model_from_config(config)
    
    print("   执行prepare_qat...")
    model_reload = QATQuantizer.prepare_qat(
        model_reload,
        backend='fbgemm',
        fuse_modules=True,
        fusion_strategy='resnet',
        inplace=True
    )
    
    print("   执行convert...")
    model_reload = QATQuantizer.convert_qat(model_reload, inplace=True)
    
    print("   加载convert后的权重...")
    try:
        missing, unexpected = model_reload.load_state_dict(
            torch.load(quantized_state_dict_path, map_location='cpu', weights_only=True),
            strict=False
        )
        
        if missing:
            print(f"   ⚠️  缺失键: {len(missing)} 个")
            if len(missing) <= 3:
                for k in missing:
                    print(f"      - {k}")
        if unexpected:
            print(f"   ⚠️  未预期键: {len(unexpected)} 个")
            if len(unexpected) <= 3:
                for k in unexpected:
                    print(f"      - {k}")
        
        if not missing and not unexpected:
            print("   ✓ 权重加载完美匹配！")
        
        model_reload.eval()
        
        # 测试重新加载的模型推理性能
        print("\n   测试重新加载模型的推理性能...")
        result_reload = benchmark_inference(model_reload, data_list, device='cpu', warmup=10)
        results['pytorch_quantized_reload'] = result_reload
        print(f"   平均推理时间: {result_reload['mean_time']*1000:.2f} ms")
        print(f"   吞吐量: {result_reload['throughput']:.2f} images/sec")
        
        # 对比原始convert模型和重新加载的模型
        time_diff = abs(result_reload['mean_time'] - result_pytorch['mean_time'])
        time_diff_percent = (time_diff / result_pytorch['mean_time']) * 100
        print(f"\n   与原始INT8模型对比:")
        print(f"   - 耗时差异: {time_diff*1000:.2f} ms ({time_diff_percent:.2f}%)")
        
        if time_diff_percent < 5:
            print("   ✓ 性能一致！权重重新加载验证成功")
        else:
            print("   ⚠️  性能差异较大，可能存在问题")
        
        # 验证输出一致性
        print("\n   验证输出一致性...")
        model_quantized.eval()
        model_reload.eval()
        
        max_abs_diff_list = []
        mean_abs_diff_list = []
        max_rel_diff_list = []
        
        with torch.no_grad():
            # 使用前5批数据进行验证
            for i, data in enumerate(data_list[:5]):
                inputs = {'images': data}
                
                # 原始INT8模型输出
                output1 = model_quantized(inputs)
                # 重新加载的模型输出
                output2 = model_reload(inputs)
                
                # 提取实际的输出张量（输出可能是字典或直接是张量）
                if isinstance(output1, dict):
                    output1 = output1.get('logits', output1.get('output', list(output1.values())[0]))
                if isinstance(output2, dict):
                    output2 = output2.get('logits', output2.get('output', list(output2.values())[0]))
                
                # 计算差异
                abs_diff = torch.abs(output1 - output2)
                max_abs_diff = abs_diff.max().item()
                mean_abs_diff = abs_diff.mean().item()
                
                # 计算相对差异
                rel_diff = abs_diff / (torch.abs(output1) + 1e-8)
                max_rel_diff = rel_diff.max().item()
                
                max_abs_diff_list.append(max_abs_diff)
                mean_abs_diff_list.append(mean_abs_diff)
                max_rel_diff_list.append(max_rel_diff)
        
        avg_max_abs_diff = np.mean(max_abs_diff_list)
        avg_mean_abs_diff = np.mean(mean_abs_diff_list)
        avg_max_rel_diff = np.mean(max_rel_diff_list)
        
        print(f"   - 平均最大绝对差异: {avg_max_abs_diff:.6e}")
        print(f"   - 平均平均绝对差异: {avg_mean_abs_diff:.6e}")
        print(f"   - 平均最大相对差异: {avg_max_rel_diff:.6e} ({avg_max_rel_diff*100:.4f}%)")
        
        # 判断一致性
        if avg_max_abs_diff < 1e-5 and avg_max_rel_diff < 1e-5:
            print("   ✓ 输出完全一致！")
        elif avg_max_abs_diff < 1e-3 and avg_max_rel_diff < 0.01:
            print("   ✓ 输出基本一致（在可接受误差范围内）")
        else:
            print(f"   ⚠️  输出存在较大差异")
        
        results['output_consistency'] = {
            'avg_max_abs_diff': float(avg_max_abs_diff),
            'avg_mean_abs_diff': float(avg_mean_abs_diff),
            'avg_max_rel_diff': float(avg_max_rel_diff)
        }
            
    except Exception as e:
        print(f"   ❌ 权重加载失败: {e}")
        print(f"   错误类型: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        results['pytorch_quantized_reload'] = {'error': str(e)}
    
    
    # 保存结果
    results_path = output_dir / 'benchmark_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 打印汇总表格
    print("\n" + "=" * 80)
    print("性能对比汇总表")
    print("=" * 80)
    print(f"{'模型格式':<30} {'平均耗时(ms)':<15} {'标准差(ms)':<15} {'吞吐量(img/s)':<15} {'相对加速':<10}")
    print("-" * 80)
    
    base_time = results['qat_before_convert']['mean_time']
    
    formats = [
        ('QAT模型(FakeQuant)', 'qat_before_convert'),
        ('PyTorch INT8量化', 'pytorch_quantized'),
        ('PyTorch INT8(重加载)', 'pytorch_quantized_reload'),
    ]
    
    for name, key in formats:
        if key in results and 'error' not in results[key]:
            r = results[key]
            speedup = base_time / r['mean_time']
            print(f"{name:<30} {r['mean_time']*1000:<15.2f} {r['std_time']*1000:<15.2f} "
                  f"{r['throughput']:<15.2f} {speedup:<10.2f}x")
        else:
            print(f"{name:<30} {'FAILED':<15} {'-':<15} {'-':<15} {'-':<10}")
    
    print("=" * 80)
    
    # 打印文件大小对比
    print("\n文件大小对比:")
    print("-" * 40)
    original_size = checkpoint_path.stat().st_size / (1024 * 1024)
    print(f"原始QAT checkpoint: {original_size:.2f} MB")
    if quantized_model_path.exists():
        print(f"INT8量化模型:       {model_size_mb:.2f} MB ({original_size/model_size_mb:.2f}x压缩)")
    
    print(f"\n测试完成！共测试了 {len(data_list)} 批次，总计 {len(data_list) * batch_size} 张图像")
    print(f"所有输出文件保存在: {output_dir}")
    print(f"详细结果: {results_path}")


if __name__ == '__main__':
    main()
