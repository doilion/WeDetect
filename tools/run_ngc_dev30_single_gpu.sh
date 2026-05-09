#!/usr/bin/env bash
# Fallback single-GPU launcher for dev30. Used after GPU 1 fell off the bus
# (Xid 79) corrupted the multi-GPU NVML state on this host. Resumes from
# the latest dev30 checkpoint (ep8 best). No torchrun / no DDP — avoids the
# whole NCCL surface area.
set -euo pipefail

cd /home/25_liwenjie/code/WeDetect

source /home/25_liwenjie/anaconda3/etc/profile.d/conda.sh
conda activate wedetect

export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export MPLCONFIGDIR=/tmp/matplotlib-wedetect
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

CFG="config/wedetect_tiny_tct_ngc_dev30_cache640_fullnames_disjoint_2gpu.py"
WORK="work_dirs/wedetect_tiny_tct_ngc_dev30_cache640_fullnames_disjoint_2gpu"
LOG="${WORK}/train_tmux.log"

mkdir -p "${WORK}"

{
  printf "\n==== NGC_DEV30_SINGLE_GPU_START %s ====\n" "$(date "+%F %T")"
  printf "CFG=%s\nWORK=%s\nCUDA_VISIBLE_DEVICES=%s\n" \
    "${CFG}" "${WORK}" "${CUDA_VISIBLE_DEVICES}"
  python train.py "${CFG}" --amp --resume auto
  printf "==== NGC_DEV30_SINGLE_GPU_DONE %s ====\n" "$(date "+%F %T")"
} 2>&1 | tee -a "${LOG}"
