#!/usr/bin/env bash
set -u

cd /home/25_liwenjie/code/WeDetect || exit 1

LOG=${LOG:-work_dirs/wedetect_tiny_tct_ngc_dev32_cache640_fullnames_4gpu/train_tmux.log}
OUT=${OUT:-work_dirs/ngc_monitor_dev32_fullnames_4gpu.log}
INTERVAL=${INTERVAL:-300}

mkdir -p "$(dirname "$OUT")"

while true; do
  {
    echo "==== $(date '+%F %T') ===="
    stat -c "log_mtime=%y size=%s" "$LOG" 2>/dev/null || true
    rg "Epoch\\(train\\)|Epoch\\(val\\)|coco/bbox_mAP|Saving checkpoint" "$LOG" 2>/dev/null | tail -n 5 || true

    errors=$(tail -n 500 "$LOG" 2>/dev/null | rg "Traceback|NCCL.*timeout|DistBackendError|CUDA out of memory|RuntimeError" | tail -n 5 || true)
    if [[ -n "$errors" ]]; then
      echo "ALERT_ERRORS"
      echo "$errors"
    fi

    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true
  } 2>&1 | tee -a "$OUT"

  sleep "$INTERVAL"
done
