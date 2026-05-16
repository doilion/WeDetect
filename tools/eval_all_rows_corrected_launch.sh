#!/usr/bin/env bash
# Launch corrected-val re-eval for Row 3 (m1) on GPU 0 and Row 3.5 (m1_5attrmean) on GPU 1,
# in parallel. Row 4d (m2) is run after its training completes; do not launch here.
#
# Logs: work_dirs/<run>/corrected_val/{epoch_NN.{log,json},summary.csv}
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs/corrected_val
GPU0_LOG="logs/corrected_val/m1_gpu0.log"
GPU1_LOG="logs/corrected_val/m1_5attrmean_gpu1.log"

nohup bash tools/eval_all_ckpts_corrected_val.sh m1            0 > "$GPU0_LOG" 2>&1 &
PID0=$!
nohup bash tools/eval_all_ckpts_corrected_val.sh m1_5attrmean  1 > "$GPU1_LOG" 2>&1 &
PID1=$!

echo "Row 3   (m1)            GPU 0  PID=$PID0  log=$GPU0_LOG"
echo "Row 3.5 (m1_5attrmean)  GPU 1  PID=$PID1  log=$GPU1_LOG"
echo "Each row: 12 epochs × ~3 min ≈ 35 min wall (parallel)."
