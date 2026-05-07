#!/usr/bin/env bash
set -euo pipefail

cd /home/25_liwenjie/code/WeDetect

source /home/25_liwenjie/anaconda3/etc/profile.d/conda.sh
conda activate wedetect

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

LOG="work_dirs/tct_ngc_cache640_refresh.log"
mkdir -p work_dirs

{
  printf "\n==== CACHE_REFRESH_START %s ====\n" "$(date "+%F %T")"
  python tools/cache_tct_ngc_640.py \
    --source-root /home1/liwenjie/TCT_NGC \
    --out-root /home1/liwenjie/TCT_NGC_640 \
    --splits train_dev val_dev \
    --size 640 \
    --workers "${CACHE_WORKERS:-32}" \
    --quality 95 \
    --overwrite
  printf "==== CACHE_REFRESH_DONE %s ====\n" "$(date "+%F %T")"
} 2>&1 | tee -a "${LOG}"
