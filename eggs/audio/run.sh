#!/bin/bash
# ============================================================
# 音频分类实验脚本
#
# 使用方法:
#   bash run.sh --stage 0 --stop_stage 3
#
# Stage 说明:
#   -1: 环境检查
#    0: 数据准备（生成 JSONL）
#    1: 计算 CMVN
#    2: 训练模型
#    3: 平均模型 checkpoints
#    4: 推理/评估
#    5: 导出 ONNX
#    6: ONNX 推理
#    7: 动态量化
#    8: 动态量化推理
#    9: 静态量化
#   10: 静态量化推理
#   11: QAT 量化感知训练
#   12: 平均 QAT 模型 checkpoints
#   13: QAT 模型转换（转换为量化模型）
#   14: 量化模型推理
# ============================================================

set -e
set -o pipefail

. ./path.sh

# ============================================================
# 参数配置（仅保留stage、gpu和路径配置）
# ============================================================

stage=$1
stop_stage=$2
gpu=0  # 单卡训练: gpu=0  多卡训练: gpu=0,1,2,3
export CUDA_VISIBLE_DEVICES=${gpu} 

raw_data=download/silero_sliced
data_dir=data

config=conf/config.yaml
exp_dir=exp/audio_test
checkpoint=
seed=42

num_average=5
avg_min_epoch=0

include_eval=true  # 是否在推理时包含 eval 数据集

qat_exp_dir=${exp_dir}/exp_qat
qat_config=conf/config_qat.yaml
qat_backend=fbgemm  # 量化后端，可选 fbgemm 或 qnnpack

. ./parse_options.sh || exit 1

# 创建目录
mkdir -p ${data_dir}/{train,eval,test} ${exp_dir}


# ============================================================
# Stage 0: 数据准备
# ============================================================
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "Stage 0: Data preparation"

    [ ! -d "${raw_data}" ] && {
        echo "Error: Raw data not found at ${raw_data}"
        exit 1
    }

    python local/data_2_jsonl.py \
        --data_root ${raw_data} \
        --output_dir ${data_dir} \
        --eval_samples_per_class 150 \
        --test_samples_per_class 150 \
        --seed 42

    echo "Generated JSONL files:"
    wc -l ${data_dir}/{train,eval,test}/data.jsonl
fi

# ============================================================
# Stage 1: 计算 CMVN
# ============================================================
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Stage 1: Computing CMVN"

    [ ! -f "${data_dir}/train/data.jsonl" ] && {
        echo "Error: Training data not found. Run stage 0 first."
        exit 1
    }

    python FunFlow/utils/compute_cmvn.py \
        --data_jsonl ${data_dir}/train/data.jsonl \
        --output_file ${data_dir}/cmvn.json \
        --sample_rate 16000 \
        --num_mel_bins 80 \
        --frame_length 25.0 \
        --frame_shift 10.0 \
        --max_samples -1 \
        --style librosa

    echo "CMVN saved to: ${data_dir}/cmvn.json"
fi

# ============================================================
# Stage 2: 训练模型
# ============================================================
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Stage 2: Training"

    [ ! -f "${data_dir}/train/data.jsonl" ] && {
        echo "Error: Training data not found. Run stage 0 first."
        exit 1
    }

    [ ! -f "${config}" ] && {
        echo "Error: Config file not found at ${config}"
        exit 1
    }

    echo "Training on GPU: ${gpu}"
    
    python FunFlow/bin/train.py \
        --config ${config} \
        --work_dir ${exp_dir} \
        --seed ${seed} \
        ${checkpoint:+--checkpoint $checkpoint}

    echo "Training completed! Model saved to: ${exp_dir}"
fi

# ============================================================
# Stage 3: 平均模型 checkpoints
# ============================================================
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Stage 3: Averaging checkpoints"

    model_dir=${exp_dir}
    avg_model=${exp_dir}/avg_${num_average}.pth

    python FunFlow/utils/average_model.py \
        --src_path ${model_dir} \
        --dst_model ${avg_model} \
        --metric cv_recall \
        --num ${num_average} \
        --min_epoch ${avg_min_epoch}

    echo "Averaged model saved to: ${avg_model}"
fi

# ============================================================
# Stage 4: 推理/评估
# ============================================================
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Stage 4: Inference"

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
        --file_field audio_path \
        --gt_fields label \
        --num_threads 1 

    # 清理临时文件
    [ -f "${temp_jsonl}" ] && rm -f ${temp_jsonl}

    echo "Inference completed! Results saved to: ${output_file}"
fi

# ============================================================
# Stage 5: 导出 ONNX
# ============================================================
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "Stage 5: Export ONNX"

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
        --input_shape 1,300,80 \
        --output_keys logits,probs,preds \
        --dynamic_axes 'input:0=batch_size,1=seq_len;logits:0=batch_size;probs:0=batch_size;preds:0=batch_size' \
        --simplify

    echo "ONNX model saved to: ${exp_dir}/${model_name}.onnx"
fi

# ============================================================
# Stage 6: ONNX 推理/评估
# ============================================================
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "Stage 6: ONNX Inference"

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
        --file_field audio_path \
        --gt_fields label \
        --num_threads 1

    # 清理临时文件
    [ -f "${temp_jsonl}" ] && rm -f ${temp_jsonl}

    echo "Inference completed! Results saved to: ${output_file}"
fi

# ============================================================
# Stage 7: 动态量化
# ============================================================
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "Stage 7: Dynamic Quantization (PTQ)"

    # 确保先导出ONNX模型
    onnx_model=${exp_dir}/avg_${num_average}.onnx
    [ ! -f "${onnx_model}" ] && {
        echo "Error: ONNX model not found at ${onnx_model}. Run stage 5 first."
        exit 1
    }
    
    model_name=$(basename ${onnx_model} .onnx)
    output_model=${exp_dir}/${model_name}_dynamic_quantized_QUInt8.onnx
    quant_info_path=${exp_dir}/eval/${model_name}_dynamic_quant_info_QUInt8

    # 使用PTQ CLI工具进行动态量化
    python -m FunFlow.compression.ptq_cli \
        --method dynamic \
        --model_path ${onnx_model} \
        --output_path ${output_model} \
        --weight_type QUInt8 \
        --save_quant_info \
        --quant_info_path ${quant_info_path}

    echo "Dynamic quantized model saved to: ${output_model}"
    echo "Quantization info saved to: ${quant_info_path}.{json,txt}"
fi

# ============================================================
# Stage 8: 动态量化推理/评估
# ============================================================
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    echo "Stage 8: Dynamic Quantized Inference"

    model_path=${exp_dir}/avg_${num_average}_dynamic_quantized_QUInt8.onnx
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
        --file_field audio_path \
        --gt_fields label \
        --num_threads 1

    # 清理临时文件
    [ -f "${temp_jsonl}" ] && rm -f ${temp_jsonl}

    echo "Inference completed! Results saved to: ${output_file}"
fi

# ============================================================
# Stage 9: 静态量化
# ============================================================
if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    echo "Stage 9: Static Quantization (PTQ)"

    # 确保先导出ONNX模型
    onnx_model=${exp_dir}/avg_${num_average}.onnx
    [ ! -f "${onnx_model}" ] && {
        echo "Error: ONNX model not found at ${onnx_model}. Run stage 5 first."
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
# Stage 10: 静态量化推理/评估
# ============================================================
if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    echo "Stage 10: Static Quantized Inference"

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
        --file_field audio_path \
        --gt_fields label \
        --num_threads 1

    # 清理临时文件
    [ -f "${temp_jsonl}" ] && rm -f ${temp_jsonl}

    echo "Inference completed! Results saved to: ${output_file}"
fi




















# ============================================================
# Stage 11: QAT 量化感知训练
# ============================================================
if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
    echo "Stage 11: Quantization-Aware Training (QAT)"
    echo "=============================================="

    # QAT 需要预训练模型作为起点
    pretrained_model=${exp_dir}/avg_${num_average}.pth
    if [ -f "${pretrained_model}" ]; then
        echo "Using pretrained model: ${pretrained_model}"
    else
        echo "Error: Pretrained model not found at ${pretrained_model}"
        echo "Please run stage 1-3 first to train and average the model."
        exit 1
    fi

    # 检查 QAT 配置文件
    qat_config=conf/config_qat.yaml
    if [ ! -f "${qat_config}" ]; then
        echo "Warning: QAT config not found at ${qat_config}"
        echo "Using default config: ${config}"
        qat_config=${config}
    fi

    echo "QAT Config: ${qat_config}"
    echo "Work dir: ${qat_exp_dir}"
    echo "Backend: ${qat_backend}"
    echo ""

    echo "QAT training on GPU: ${gpu}"
    
    CUDA_VISIBLE_DEVICES=${gpu} python FunFlow/bin/train_qat.py \
        --config ${qat_config} \
        --work_dir ${qat_exp_dir} \
        --checkpoint ${pretrained_model} \
        --seed ${seed} \
        --backend ${qat_backend}

    echo "QAT training completed! Model saved to: ${qat_exp_dir}/"
fi

# ============================================================
# Stage 12: 平均QAT模型 checkpoints
# ============================================================
if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
    echo "Stage 12: Averaging QAT checkpoints"

    model_dir=${qat_exp_dir}
    avg_model=${qat_exp_dir}/avg_${num_average}.pth

    python FunFlow/utils/average_model.py \
        --src_path ${model_dir} \
        --dst_model ${avg_model} \
        --metric cv_recall \
        --num ${num_average}

    echo "Averaged QAT model saved to: ${avg_model}"
fi

# ============================================================
# Stage 13: QAT 模型转换和保存为量化模型
# ============================================================
if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ]; then
    echo "Stage 13: Convert QAT Model to Quantized Model"

    qat_model=${qat_exp_dir}/avg_${num_average}.pth
    [ ! -f "${qat_model}" ] && {
        echo "Error: QAT model not found at ${qat_model}. Run stage 11-12 first."
        exit 1
    }
    model_name=$(basename ${qat_model} .pth)
    output_file=${qat_exp_dir}/${model_name}_qat_quantized_state_dict.pth

    qat_config=${qat_exp_dir}/config.yaml
    if [ ! -f "${qat_config}" ]; then
        echo "Warning: QAT config not found at ${qat_config}"
        echo "Using default config: conf/config_qat.yaml"
        qat_config=conf/config_qat.yaml
    fi

    # 使用 qat_cli 转换 QAT 模型为量化模型
    python -m FunFlow.compression.qat_cli \
        --checkpoint ${qat_model} \
        --config ${qat_config} \
        --output ${output_file} \
        --backend ${qat_backend}

    echo "QAT model conversion completed!"
fi

# ============================================================
# Stage 14: 量化模型推理
# ============================================================
if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ]; then
    echo "Stage 14: Inference with Quantized Model"
    
    quantized_model=${qat_exp_dir}/avg_${num_average}_qat_quantized_state_dict.pth
    [ ! -f "${quantized_model}" ] && {
        echo "Error: Quantized model not found at ${quantized_model}. Run stage 13 first."
        exit 1
    }
    
    qat_config=${qat_exp_dir}/config.yaml
    if [ ! -f "${qat_config}" ]; then
        echo "Warning: QAT config not found at ${qat_config}"
        echo "Using default config: conf/config_qat.yaml"
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

    python -m FunFlow.inference.cli \
        --inferencer_type audio \
        --processor_type audio \
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

echo ""
echo "=============================================="
echo "Pipeline completed!"
echo "=============================================="
