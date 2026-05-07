#!/usr/bin/env bash
set -euo pipefail

cd /home/25_liwenjie/code/WeDetect
source /home/25_liwenjie/anaconda3/etc/profile.d/conda.sh
conda activate wedetect

export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-wedetect}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

CFG=config/wedetect_tiny_tct_ngc_dev32_cache640_fullnames_4gpu.py
WORK=work_dirs/wedetect_tiny_tct_ngc_dev32_cache640_fullnames_4gpu
CKPT="$WORK/epoch_6.pth"
LOG="$WORK/train_tmux.log"
PORT=${PORT:-29644}

mkdir -p "$WORK" "$MPLCONFIGDIR"

{
  echo
  echo "==== RESTART_2GPU_ACCUM2 $(date '+%F %T') ===="
  echo "cfg=$CFG"
  echo "resume=$CKPT"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "gpus=2"
  echo "optim_wrapper.accumulative_counts=2"
} | tee -a "$LOG"

PORT="$PORT" bash dist_train.sh "$CFG" 2 \
  --resume "$CKPT" \
  --cfg-options optim_wrapper.accumulative_counts=2 \
  2>&1 | tee -a "$LOG"
