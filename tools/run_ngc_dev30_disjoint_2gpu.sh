#!/usr/bin/env bash
set -euo pipefail

cd /home/25_liwenjie/code/WeDetect

source /home/25_liwenjie/anaconda3/etc/profile.d/conda.sh
conda activate wedetect

export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export MPLCONFIGDIR=/tmp/matplotlib-wedetect
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

CFG="config/wedetect_tiny_tct_ngc_dev30_cache640_fullnames_disjoint_2gpu.py"
WORK="work_dirs/wedetect_tiny_tct_ngc_dev30_cache640_fullnames_disjoint_2gpu"
LOG="${WORK}/train_tmux.log"

mkdir -p "${WORK}"

{
  printf "\n==== NGC_DEV30_DISJOINT_2GPU_START %s ====\n" "$(date "+%F %T")"
  printf "CFG=%s\nWORK=%s\nCUDA_VISIBLE_DEVICES=%s\nPORT=%s\n" \
    "${CFG}" "${WORK}" "${CUDA_VISIBLE_DEVICES}" "${PORT:-29646}"
  PORT="${PORT:-29646}" bash dist_train.sh "${CFG}" 2 --amp ${EXTRA_ARGS:-}
  printf "==== NGC_DEV30_DISJOINT_2GPU_DONE %s ====\n" "$(date "+%F %T")"
} 2>&1 | tee -a "${LOG}"
