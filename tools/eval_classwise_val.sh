#!/usr/bin/env bash
# Run val with classwise=True on each row's current best_*.pth to get per-class AP.
# Output: work_dirs/<run>/classwise_val/perclass.json + perclass.csv
#
# Usage:
#   bash tools/eval_classwise_val.sh <row_tag> <gpu_id> <ckpt_path>
set -euo pipefail

ROW_TAG="${1:-}"
GPU="${2:-3}"
CKPT="${3:-}"
if [[ -z "$ROW_TAG" || -z "$CKPT" ]]; then
    echo "usage: bash $0 <m1|m1_5attrmean|m2> <gpu_id> <ckpt_path>"; exit 1
fi

CONFIG="config/wedetect_tiny_tct_ngc_dev30_ochmta_${ROW_TAG}_biomedclip_2gpu.py"
WORK_DIR="work_dirs/wedetect_tiny_tct_ngc_dev30_ochmta_${ROW_TAG}_biomedclip_2gpu/classwise_val"
mkdir -p "$WORK_DIR"
LOG="${WORK_DIR}/perclass.log"

echo "[classwise] row=${ROW_TAG} gpu=${GPU} ckpt=${CKPT}"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. ~/anaconda3/envs/wedetect/bin/python test.py \
    "$CONFIG" "$CKPT" \
    --work-dir "${WORK_DIR}/runtime" \
    --cfg-options test_evaluator.classwise=True \
    > "$LOG" 2>&1

# Extract per-class AP from log
python - "$LOG" "${WORK_DIR}/perclass.csv" <<'PYEOF'
import re, sys, csv
log, csv_path = sys.argv[1], sys.argv[2]
text = open(log).read()

# OrganRestrictedCocoMetric's classwise table is a markdown-style block.
# Find header then walk rows until blank/separator.
rows = []
# Format from CocoMetric: lines like "| <class>      | <AP> |  | ..."
# Look for the per-class section.
mm = re.search(r'\+\-+\+[^|]*?per[-_ ]?class[^\n]*\n([\s\S]+?)(?:\n\n|\Z)', text, re.I)
if not mm:
    # Try the standard mmdet classwise table
    mm = re.search(r'category\s*\|\s*AP\s*\|.*?\n([\s\S]+?)\n\n', text, re.I)
# Fallback: grep lines like "| <class_name> | <num> |"
candidates = re.findall(r'\|\s*([A-Za-z_][\w\-\(\) ]*?)\s*\|\s*(\d+\.\d+)\s*\|', text)
seen = set()
for name, ap in candidates:
    name = name.strip()
    if name in {'category', 'AP', 'mAP', 'classwise', 'domain', 'overall macro', 'all-class flat', 'instance-weighted'}:
        continue
    if name in seen: continue
    seen.add(name)
    try:
        rows.append((name, float(ap)))
    except ValueError:
        pass

with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['class', 'AP'])
    for n, ap in rows:
        w.writerow([n, ap])
print(f"wrote {len(rows)} per-class rows to {csv_path}")
PYEOF

echo "[classwise] done. CSV: ${WORK_DIR}/perclass.csv  log: ${LOG}"
