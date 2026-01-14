#!/bin/bash
# 训练 + 自动评估脚本
# 用法: ./train_and_eval.sh config/wedetect_tiny_tct_exp2.py exp2_frozen_backbone [GPU_ID]

CONFIG=$1
EXP_NAME=$2
GPU_ID=${3:-0}

if [ -z "$CONFIG" ] || [ -z "$EXP_NAME" ]; then
    echo "用法: ./train_and_eval.sh <config_file> <exp_name> [gpu_id]"
    echo "示例: ./train_and_eval.sh config/wedetect_tiny_tct_exp2.py exp2_frozen 0"
    exit 1
fi

# 从config文件提取work_dir
WORK_DIR=$(grep "work_dir" $CONFIG | head -1 | sed "s/.*'\(.*\)'.*/\1/")

echo "=============================================="
echo "实验: $EXP_NAME"
echo "配置: $CONFIG"
echo "输出: $WORK_DIR"
echo "GPU: $GPU_ID"
echo "=============================================="

# 1. 训练
echo "[1/2] 开始训练..."
CUDA_VISIBLE_DEVICES=$GPU_ID python tools/train.py $CONFIG

# 2. 自动评估并生成报告
echo "[2/2] 生成评估报告..."
CHECKPOINT=$(ls -t ${WORK_DIR}/best_*.pth 2>/dev/null | head -1)

if [ -z "$CHECKPOINT" ]; then
    echo "警告: 未找到best checkpoint，跳过评估"
    exit 0
fi

CUDA_VISIBLE_DEVICES=$GPU_ID python eval_summary.py \
    --exp_name "$EXP_NAME" \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT"

echo "=============================================="
echo "完成! 报告保存在 work_dirs/summaries/"
echo "=============================================="
