#!/usr/bin/env bash
set -euo pipefail

cd /home/25_liwenjie/code/WeDetect

SESSION=${SESSION:-ngc_dev32_fullnames_2gpu}

while tmux has-session -t "$SESSION" 2>/dev/null; do
  echo "waiting for $SESSION to finish"
  sleep 120
done

bash tools/run_ngc_fullnames_postprocess_then_randomemb.sh
