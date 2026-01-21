#!/bin/bash
# ============================================================
# 图像分类实验脚本
#
# 使用方法:
#   bash run.sh --stage 0 --stop_stage 1
#
# Stage 说明:
#   -1: 环境检查
#    0: 数据准备（生成 JSONL）
#    1: 训练模型
#    2: 平均模型 checkpoints
#    3: 推理/评估
#    4: 导出 ONNX
#    5: ONNX 推理
#    6: 动态量化
#    7: 动态量化推理
#    8: 静态量化
#    9: 静态量化推理
#   10: QAT 量化感知训练
#   11: QAT 模型转换
#   12: 量化模型推理
#   13: 结构化剪枝
#   14: 知识蒸馏
#   15: 剪枝+QAT联合优化
#   16: 剪枝模型推理
# ============================================================

set -e
set -o pipefail

. ./path.sh

# ============================================================
# 参数配置
# ============================================================

stage=$1
stop_stage=$2
gpu=0

raw_data=download
data_dir=data

config=conf/config.yaml
exp_dir=exp/new_test
checkpoint=
seed=42

num_average=5
avg_min_epoch=50

include_eval=true  # 是否在推理时包含 eval 数据集

qat_exp_dir=${exp_dir}/exp_qat
qat_config=conf/config_qat.yaml
qat_backend=fbgemm  # 量化后端，可选 fbgemm 或 qnnpack

# 剪枝配置
pruning_exp_dir=${exp_dir}/pruned
pruning_ratio=0.5  # 剪枝比例
pruning_finetune_epochs=10  # 剪枝后微调轮数
pruning_finetune_lr=1e-4  # 微调学习率

# 蒸馏配置
distill_exp_dir=${exp_dir}/distilled
distill_temperature=4.0  # 蒸馏温度
distill_alpha=0.7  # 软标签权重
distill_epochs=100  # 蒸馏训练轮数
student_backbone=ResNet18  # 学生模型backbone

. ./parse_options.sh || exit 1


# ============================================================
# Stage 0: 数据准备
# ============================================================
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "Stage 0: Data preparation"

    [ ! -d "${raw_data}" ] && {
        echo "Error: Raw data not found at ${raw_data}"
        exit 1
    }
    mkdir -p ${data_dir}/{train,eval,test} ${exp_dir}

    python local/data_2_jsonl.py \
        --data_root ${raw_data} \
        --output_dir ${data_dir} \
        --mode ratio \
        --train_ratio 0.8 \
        --eval_ratio 0.1 \
        --test_ratio 0.1 \
        --seed 42                           

    echo "Generated JSONL files:"
    wc -l ${data_dir}/{train,eval,test}/data.jsonl
fi

# ============================================================
# Stage 1: 训练模型
# ============================================================
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Stage 1: Training"

    [ ! -f "${data_dir}/train/data.jsonl" ] && {
        echo "Error: Training data not found. Run stage 0 first."
        exit 1
    }

    [ ! -f "${config}" ] && {
        echo "Error: Config file not found at ${config}"
        exit 1
    }

    python FunFlow/bin/train.py \
        --config ${config} \
        --work_dir ${exp_dir} \
        --seed ${seed} \
        ${checkpoint:+--checkpoint $checkpoint}

    echo "Training completed! Model saved to: ${exp_dir}"
fi

# ============================================================
# Stage 2: 平均模型 checkpoints
# ============================================================
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Stage 2: Averaging checkpoints"

    model_dir=${exp_dir}
    avg_model=${exp_dir}/avg_${num_average}.pth

    python FunFlow/utils/average_model.py \
        --src_path ${model_dir} \
        --dst_model ${avg_model} \
        --metric cv_loss \
        --num ${num_average} \
        --min_epoch ${avg_min_epoch}

    echo "Averaged model saved to: ${avg_model}"
fi

# ============================================================
# Stage 3: 推理/评估
# ============================================================
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Stage 3: Inference"

    model_path=${exp_dir}/avg_${num_average}.pth
    [ ! -f "${model_path}" ] && {
        echo "Error: Model not found at ${model_path}. Run stage 2 first."
        exit 1
    }

    model_name=$(basename ${model_path} .pth)
    output_file=${exp_dir}/eval/predictions_${model_name}.json

    # 构建输入文件列表
    input_files="${data_dir}/test/data.jsonl"
    
    if [ "${include_eval}" = "true" ]; then
        if [ -f "${data_dir}/eval/data.jsonl" ]; then
            echo "Including eval dataset in inference..."
            # 临时合并两个jsonl文件
            temp_jsonl=/tmp/inference_combined_$$.jsonl
            cat ${data_dir}/test/data.jsonl ${data_dir}/eval/data.jsonl > ${temp_jsonl}
            input_files=${temp_jsonl}
        else
            echo "Warning: include_eval=true but eval dataset not found at ${data_dir}/eval/data.jsonl"
        fi
    else
        echo "Only testing on test set (include_eval=false)"
    fi

    python -m FunFlow.inference.inference_cli \
        --inferencer Inferencer \
        --checkpoint ${model_path} \
        --input ${input_files} \
        --config ${config} \
        --output ${output_file} \
        --batch_size 1 \
        --device cpu \
        --file_field image_path \
        --gt_fields label \
        --num_threads 1 

    # 清理临时文件
    [ -f "${temp_jsonl}" ] && rm -f ${temp_jsonl}

    echo "Inference completed! Results saved to: ${output_file}"
fi

# ============================================================
# Stage 4: 导出 ONNX
# ============================================================
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Stage 4: Export ONNX"

    model_path=${exp_dir}/avg_${num_average}.pth
    [ ! -f "${model_path}" ] && {
        echo "Error: Model not found at ${model_path}. Run stage 2 first."
        exit 1
    }
    model_name=$(basename ${model_path} .pth)

    python -m FunFlow.export.export_cli \
        --checkpoint ${model_path} \
        --config ${config} \
        --output ${exp_dir}/${model_name}.onnx \
        --format onnx \
        --opset_version 14 \
        --input_shape 1,3,224,224 \
        --output_keys logits,probs,preds \
        --dynamic_axes 'input:0=batch_size;logits:0=batch_size;probs:0=batch_size;preds:0=batch_size' \
        --simplify

    echo "ONNX model saved to: ${exp_dir}/${model_name}.onnx"
fi

# ============================================================
# Stage 5: ONNX 推理/评估
# ============================================================
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "Stage 5: ONNX Inference"

    model_path=${exp_dir}/avg_${num_average}.onnx
    [ ! -f "${model_path}" ] && {
        echo "Error: Model not found at ${model_path}. Run stage 5 first."
        exit 1
    }

    model_name=$(basename ${model_path} .onnx)
    output_file=${exp_dir}/eval/predictions_${model_name}_onnx.json

    # 构建输入文件列表
    input_files="${data_dir}/test/data.jsonl"
    
    if [ "${include_eval}" = "true" ]; then
        if [ -f "${data_dir}/eval/data.jsonl" ]; then
            echo "Including eval dataset in inference..."
            # 临时合并两个jsonl文件
            temp_jsonl=/tmp/inference_onnx_combined_$$.jsonl
            cat ${data_dir}/test/data.jsonl ${data_dir}/eval/data.jsonl > ${temp_jsonl}
            input_files=${temp_jsonl}
        else
            echo "Warning: include_eval=true but eval dataset not found at ${data_dir}/eval/data.jsonl"
        fi
    else
        echo "Only testing on test set (include_eval=false)"
    fi

    python -m FunFlow.inference.inference_cli \
        --inferencer ONNXInferencer \
        --checkpoint ${model_path} \
        --input ${input_files} \
        --config ${config} \
        --output ${output_file} \
        --batch_size 1 \
        --device cpu \
        --file_field image_path \
        --gt_fields label \
        --num_threads 1

    # 清理临时文件
    [ -f "${temp_jsonl}" ] && rm -f ${temp_jsonl}

    echo "Inference completed! Results saved to: ${output_file}"
fi

# ============================================================
# Stage 6: 动态量化
# ============================================================
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "Stage 6: Dynamic Quantization (PTQ)"

    # 确保先导出ONNX模型
    onnx_model=${exp_dir}/avg_${num_average}.onnx
    [ ! -f "${onnx_model}" ] && {
        echo "Error: ONNX model not found at ${onnx_model}. Run stage 4 first."
        exit 1
    }
    
    model_name=$(basename ${onnx_model} .onnx)
    output_model=${exp_dir}/${model_name}_dynamic_quantized_QInt8.onnx
    quant_info_path=${exp_dir}/eval/${model_name}_dynamic_quant_info_QInt8

    # 使用PTQ CLI工具进行动态量化
    python -m FunFlow.compression.ptq_cli \
        --method dynamic \
        --model_path ${onnx_model} \
        --output_path ${output_model} \
        --weight_type QInt8 \
        --save_quant_info \
        --quant_info_path ${quant_info_path}

    echo "Dynamic quantized model saved to: ${output_model}"
    echo "Quantization info saved to: ${quant_info_path}.{json,txt}"
fi

# ============================================================
# Stage 7: 动态量化推理/评估
# ============================================================
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "Stage 7: Dynamic Quantized Inference"

    model_path=${exp_dir}/avg_${num_average}_dynamic_quantized_QInt8.onnx
    [ ! -f "${model_path}" ] && {
        echo "Error: Model not found at ${model_path}. Run stage 7 first."
        exit 1
    }

    model_name=$(basename ${model_path} .onnx)
    output_file=${exp_dir}/eval/predictions_${model_name}.json

    # 构建输入文件列表
    input_files="${data_dir}/test/data.jsonl"
    
    if [ "${include_eval}" = "true" ]; then
        if [ -f "${data_dir}/eval/data.jsonl" ]; then
            echo "Including eval dataset in inference..."
            # 临时合并两个jsonl文件
            temp_jsonl=/tmp/inference_dynamic_quant_$$.jsonl
            cat ${data_dir}/test/data.jsonl ${data_dir}/eval/data.jsonl > ${temp_jsonl}
            input_files=${temp_jsonl}
        else
            echo "Warning: include_eval=true but eval dataset not found at ${data_dir}/eval/data.jsonl"
        fi
    else
        echo "Only testing on test set (include_eval=false)"
    fi

    python -m FunFlow.inference.inference_cli \
        --inferencer ONNXInferencer \
        --checkpoint ${model_path} \
        --input ${input_files} \
        --config ${config} \
        --output ${output_file} \
        --batch_size 1 \
        --device cpu \
        --file_field image_path \
        --gt_fields label \
        --num_threads 1

    # 清理临时文件
    [ -f "${temp_jsonl}" ] && rm -f ${temp_jsonl}

    echo "Inference completed! Results saved to: ${output_file}"
fi

# ============================================================
# Stage 8: 静态量化
# ============================================================
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    echo "Stage 8: Static Quantization (PTQ)"

    # 确保先导出ONNX模型
    onnx_model=${exp_dir}/avg_${num_average}.onnx
    [ ! -f "${onnx_model}" ] && {
        echo "Error: ONNX model not found at ${onnx_model}. Run stage 4 first."
        exit 1
    }

    calibration_data=${data_dir}/train/data.jsonl
    [ ! -f "${calibration_data}" ] && {
        echo "Error: Calibration data not found at ${calibration_data}. Run stage 0 first."
        exit 1
    }

    model_name=$(basename ${onnx_model} .onnx)
    output_model=${exp_dir}/${model_name}_static_quantized_per_channel.onnx
    quant_info_path=${exp_dir}/eval/${model_name}_static_quant_info_per_channel

    # 使用PTQ CLI工具进行静态量化
    python -m FunFlow.compression.ptq_cli \
        --method static \
        --model_path ${onnx_model} \
        --output_path ${output_model} \
        --weight_type QInt8 \
        --activation_type QUInt8 \
        --calibration_data ${calibration_data} \
        --config ${config} \
        --num_calibration_batches 100 \
        --calibrate_method MinMax \
        --save_quant_info \
        --quant_info_path ${quant_info_path} \
        --per_channel

    echo "Static quantized model saved to: ${output_model}"
    echo "Quantization info saved to: ${quant_info_path}.{json,txt}"
fi

# ============================================================
# Stage 9: 静态量化推理/评估
# ============================================================
if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    echo "Stage 9: Static Quantized Inference"

    model_path=${exp_dir}/avg_${num_average}_static_quantized_per_channel.onnx
    [ ! -f "${model_path}" ] && {
        echo "Error: Model not found at ${model_path}. Run stage 9 first."
        exit 1
    }

    model_name=$(basename ${model_path} .onnx)
    output_file=${exp_dir}/eval/predictions_${model_name}.json

    # 构建输入文件列表
    input_files="${data_dir}/test/data.jsonl"
    
    if [ "${include_eval}" = "true" ]; then
        if [ -f "${data_dir}/eval/data.jsonl" ]; then
            echo "Including eval dataset in inference..."
            # 临时合并两个jsonl文件
            temp_jsonl=/tmp/inference_static_quant_$$.jsonl
            cat ${data_dir}/test/data.jsonl ${data_dir}/eval/data.jsonl > ${temp_jsonl}
            input_files=${temp_jsonl}
        else
            echo "Warning: include_eval=true but eval dataset not found at ${data_dir}/eval/data.jsonl"
        fi
    else
        echo "Only testing on test set (include_eval=false)"
    fi

    python -m FunFlow.inference.inference_cli \
        --inferencer ONNXInferencer \
        --checkpoint ${model_path} \
        --input ${input_files} \
        --config ${config} \
        --output ${output_file} \
        --batch_size 1 \
        --device cpu \
        --file_field image_path \
        --gt_fields label \
        --num_threads 1

    # 清理临时文件
    [ -f "${temp_jsonl}" ] && rm -f ${temp_jsonl}

    echo "Inference completed! Results saved to: ${output_file}"
fi


























# ============================================================
# Stage 10: QAT 量化感知训练
# ============================================================
if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    echo "Stage 10: Quantization-Aware Training (QAT)"
    echo "=============================================="

    # QAT 需要预训练模型作为起点
    pretrained_model=${exp_dir}/avg_${num_average}.pth
    if [ -f "${pretrained_model}" ]; then
        echo "Using pretrained model: ${pretrained_model}"
    else
        echo "Error: Pretrained model not found at ${pretrained_model}"
        echo "Please run stage 1-2 first to train and average the model."
        exit 1
    fi

    # 检查 QAT 配置文件
    if [ ! -f "${qat_config}" ]; then
        echo "Warning: QAT config not found at ${qat_config}"
        echo "Using default config: ${config}"
        qat_config=${config}
    fi

    echo "QAT Config: ${qat_config}"
    echo "Work dir: ${qat_exp_dir}"
    echo "Backend: ${qat_backend}"
    echo ""

    # 使用 train_qat.py 进行 QAT 训练
    python fundiagnosis/bin/train_qat.py \
        --config ${qat_config} \
        --work_dir ${qat_exp_dir} \
        --checkpoint ${pretrained_model} \
        --gpu ${gpu} \
        --backend ${qat_backend}
fi

# ============================================================
# Stage 11: 平均模型 checkpoints
# ============================================================
if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
    echo "Stage 11: Averaging checkpoints"

    model_dir=${qat_exp_dir}
    avg_model=${qat_exp_dir}/avg_${num_average}.pth

    python fundiagnosis/utils/average_model.py \
        --src_path ${model_dir} \
        --dst_model ${avg_model} \
        --metric cv_loss \
        --num ${num_average} \

    echo "Averaged model saved to: ${avg_model}"
fi

# ============================================================
# Stage 12: QAT 模型转换和保存为量化模型
# ============================================================
if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
    echo "Stage 12: Convert QAT Model to Quantized Model"

    qat_model=${qat_exp_dir}/avg_${num_average}.pth
    [ ! -f "${qat_model}" ] && {
        echo "Error: QAT model not found at ${qat_model}. Run stage 10 first."
        exit 1
    }
    model_name=$(basename ${qat_model} .pth)
    output_file=${qat_exp_dir}/${model_name}_qat_quantized_state_dict.pth

    qat_config=${qat_exp_dir}/config.yaml
    if [ ! -f "${qat_config}" ]; then
        echo "Warning: QAT config not found at ${qat_config}"
        echo "Using default config: ${config}"
        qat_config=conf/config_qat.yaml
    fi

    # 使用简化后的参数：--format quantized --quant_method qat
    python -m fundiagnosis.compression.qat_cli \
        --checkpoint ${qat_model} \
        --config ${qat_config} \
        --output ${output_file} \
        --backend ${qat_backend}

    echo "QAT model conversion completed!"
fi

# ============================================================
# Stage 13: 量化模型推理
# ============================================================
if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ]; then
    echo "Stage 13: Inference with Quantized Model"
    quantized_model=${qat_exp_dir}/avg_${num_average}_qat_quantized_state_dict.pth
    [ ! -f "${quantized_model}" ] && {
        echo "Error: Quantized model not found at ${quantized_model}. Run stage 11 first."
        exit 1
    }
    qat_config=${qat_exp_dir}/config.yaml
    if [ ! -f "${qat_config}" ]; then
        echo "Warning: QAT config not found at ${qat_config}"
        echo "Using default config: ${config}"
        qat_config=conf/config_qat.yaml
    fi
    model_name=$(basename ${quantized_model} .pth)
    output_file=${exp_dir}/eval/predictions_${model_name}.json

    # 推理
    echo "Running inference with quantized model..."

    # 构建 JSONL 文件列表
    jsonl_files="${data_dir}/test/data.jsonl"

    if [ "${include_eval}" = "true" ]; then
        if [ -f "${data_dir}/eval/data.jsonl" ]; then
            echo "Including eval dataset in inference..."
            jsonl_files="${jsonl_files} ${data_dir}/eval/data.jsonl"
        fi
    fi

    python -m fundiagnosis.inference.cli \
        --inferencer_type image \
        --processor_type image \
        --model_loader qat \
        --config ${qat_config} \
        --checkpoint ${quantized_model} \
        --jsonl ${jsonl_files} \
        --output ${output_file} \
        --gpu -1 \
        --warmup 100 \
        --timing \
        --batch_size 1 \
        --num_threads 1

    echo "Quantized model inference completed! Results saved to: ${output_file}"
fi

# ============================================================
# Stage 14: 结构化剪枝 (使用 torch-pruning 实现真正的模型压缩)
# ============================================================
if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ]; then
    echo "Stage 14: Structured Pruning (torch-pruning)"
    echo ""
    echo "This stage uses torch-pruning library for REAL structured pruning."
    echo "This actually removes channels/filters, resulting in a smaller model"
    echo "with faster inference."
    echo ""

    model_path=${exp_dir}/avg_${num_average}.pth
    [ ! -f "${model_path}" ] && {
        echo "Error: Model not found at ${model_path}. Run stage 2 first."
        exit 1
    }

    mkdir -p ${pruning_exp_dir}

    echo "Applying structured (channel) pruning with ratio ${pruning_ratio}..."
    echo "Using iterative pruning with finetuning between iterations..."

    # 使用 torch-pruning 进行真正的结构化剪枝
    # --iterative: 迭代剪枝，每次剪枝后进行微调，然后再剪枝
    # --importance: 重要性评估方法 (l1, l2, taylor, bn_scale, random)
    # --round_to: 通道数对齐到指定倍数，有利于硬件加速
    # --export_onnx: 同时导出 ONNX 模型
    # 注意：代码会自动检测并排除有 attention_biases 的 Transformer 层
    python -m fundiagnosis.bin.prune \
        --config ${config} \
        --checkpoint ${model_path} \
        --output ${pruning_exp_dir}/pruned_model.pth \
        --ratio ${pruning_ratio} \
        --importance l1 \
        --round_to 8 \
        --exclude_layers classifier head fc \
        --iterative \
        --num_iterations 5 \
        --finetune \
        --finetune_epochs ${pruning_finetune_epochs} \
        --finetune_lr ${pruning_finetune_lr} \
        --export_onnx \
        --gpu ${gpu}

    echo ""
    echo "Structured pruning completed!"
    echo "Pruned model saved to: ${pruning_exp_dir}/pruned_model.pth"
    echo "ONNX model saved to: ${pruning_exp_dir}/pruned_model.onnx"
    echo ""
    echo "Note: The pruned model has a different architecture than the original."
    echo "      When loading, use the saved 'model' key which contains the full model."
fi

# ============================================================
# Stage 15: 知识蒸馏
# ============================================================
if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ]; then
    echo "Stage 15: Knowledge Distillation"

    teacher_model=${exp_dir}/avg_${num_average}.pth
    [ ! -f "${teacher_model}" ] && {
        echo "Error: Teacher model not found at ${teacher_model}. Run stage 2 first."
        exit 1
    }

    mkdir -p ${distill_exp_dir}

    echo "Training student model with knowledge distillation..."
    echo "Teacher: ${teacher_model}"
    echo "Student backbone: ${student_backbone}"
    echo "Temperature: ${distill_temperature}"
    echo "Alpha: ${distill_alpha}"

    python -m fundiagnosis.bin.distill \
        --config ${config} \
        --teacher_checkpoint ${teacher_model} \
        --output ${distill_exp_dir} \
        --student_backbone ${student_backbone} \
        --temperature ${distill_temperature} \
        --alpha ${distill_alpha} \
        --epochs ${distill_epochs} \
        --gpu ${gpu}

    echo "Knowledge distillation completed!"
    echo "Student model saved to: ${distill_exp_dir}/best_student.pth"
fi

# ============================================================
# Stage 16: 剪枝+QAT联合优化
# ============================================================
if [ ${stage} -le 16 ] && [ ${stop_stage} -ge 16 ]; then
    echo "Stage 16: Pruning + QAT Combined Optimization"

    # 检查是否有结构化剪枝模型
    pruned_model=${pruning_exp_dir}/pruned_model.pth

    [ ! -f "${pruned_model}" ] && {
        echo "Error: Pruned model not found at ${pruned_model}. Run stage 13 first."
        exit 1
    }

    combined_exp_dir=${exp_dir}/pruned_qat
    mkdir -p ${combined_exp_dir}

    echo "Starting QAT on pruned model: ${pruned_model}"

    # Step 1: 在剪枝模型上进行QAT训练
    python fundiagnosis/bin/train.py \
        --config ${config} \
        --work_dir ${combined_exp_dir} \
        --gpu ${gpu} \
        --qat \
        --checkpoint ${pruned_model}

    # Step 2: 转换为量化模型
    qat_model=${combined_exp_dir}/best.pth
    if [ -f "${qat_model}" ]; then
        echo "Converting QAT model..."
        python -m fundiagnosis.export.export_cli \
            --config ${config} \
            --checkpoint ${qat_model} \
            --output ${combined_exp_dir}/pruned_quantized.pth \
            --format quantized \
            --quant_method qat \
            --quant_backend ${qat_backend}
    fi

    echo "Pruning + QAT combined optimization completed!"
    echo "Final model saved to: ${combined_exp_dir}/"
fi

# ============================================================
# Stage 17: 剪枝模型推理/评估
# ============================================================
if [ ${stage} -le 17 ] && [ ${stop_stage} -ge 17 ]; then
    echo "Stage 17: Pruned Model Inference"

    # 查找可用的剪枝模型
    model_path=""
    model_name=""

    # 优先级：联合优化 > 结构化剪枝 > 蒸馏
    if [ -f "${exp_dir}/pruned_qat/pruned_quantized.pth" ]; then
        model_path="${exp_dir}/pruned_qat/pruned_quantized.pth"
        model_name="pruned_quantized"
    elif [ -f "${pruning_exp_dir}/pruned_model.pth" ]; then
        model_path="${pruning_exp_dir}/pruned_model.pth"
        model_name="pruned_structured"
    elif [ -f "${distill_exp_dir}/best_student.pth" ]; then
        model_path="${distill_exp_dir}/best_student.pth"
        model_name="distilled_student"
    fi

    [ -z "${model_path}" ] && {
        echo "Error: No compressed model found. Run stage 13, 14, or 15 first."
        exit 1
    }

    echo "Running inference with: ${model_path}"
    
    output_file=${exp_dir}/eval/predictions_${model_name}.json

    jsonl_files="${data_dir}/test/data.jsonl"
    
    if [ "${include_eval}" = "true" ]; then
        if [ -f "${data_dir}/eval/data.jsonl" ]; then
            echo "Including eval dataset in inference..."
            jsonl_files="${jsonl_files} ${data_dir}/eval/data.jsonl"
        fi
    fi

    # 根据模型类型选择推理方式
    if [[ "${model_name}" == *"quantized"* ]]; then
        python fundiagnosis/bin/image_inference.py \
            --config ${config} \
            --checkpoint ${model_path} \
            --quantized \
            --backend ${qat_backend} \
            --jsonl ${jsonl_files} \
            --output ${output_file} \
            --gpu -1
    else
        python fundiagnosis/bin/image_inference.py \
            --config ${config} \
            --checkpoint ${model_path} \
            --jsonl ${jsonl_files} \
            --output ${output_file} \
            --gpu ${gpu}
    fi

    echo "Inference completed! Results saved to: ${output_file}"
fi

echo ""
echo "=============================================="
echo "Pipeline completed!"
echo "==============================================" 



