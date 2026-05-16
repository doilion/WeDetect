#!/usr/bin/env bash
# Re-evaluate every saved epoch ckpt under the corrected val protocol
# (OrganRestrictedCocoMetric, 5 negatives excluded, organ-macro).
#
# Why: Row 3 / Row 3.5 / Row 4d were trained before commit 863e7ae fixed
# val_evaluator. Their "best_*.pth" was picked under the OLD 30-class flat
# COCO mAP (with negatives). We need to find the truly-best ckpt per row
# under the paper-protocol val metric.
#
# Usage:
#   bash tools/eval_all_ckpts_corrected_val.sh <row_tag> <gpu_id>
#   row_tag in {m1, m1_5attrmean, m2}
#
# Output:
#   work_dirs/<run>/corrected_val/epoch_NN.json   (per-epoch full metrics)
#   work_dirs/<run>/corrected_val/summary.csv     (epoch, organ_macro, all_flat)
set -euo pipefail

ROW_TAG="${1:-}"
GPU="${2:-0}"
if [[ -z "$ROW_TAG" ]]; then
    echo "usage: bash $0 <m1|m1_5attrmean|m2> <gpu_id>"; exit 1
fi

CONFIG="config/wedetect_tiny_tct_ngc_dev30_ochmta_${ROW_TAG}_biomedclip_2gpu.py"
WORK_DIR="work_dirs/wedetect_tiny_tct_ngc_dev30_ochmta_${ROW_TAG}_biomedclip_2gpu"
OUT_DIR="${WORK_DIR}/corrected_val"
mkdir -p "$OUT_DIR"

if [[ ! -f "$CONFIG" ]]; then
    echo "config not found: $CONFIG"; exit 1
fi

SUMMARY="${OUT_DIR}/summary.csv"
echo "epoch,organ_macro_mAP,all_class_mAP,instance_weighted_mAP" > "$SUMMARY"

for CKPT in "${WORK_DIR}"/epoch_*.pth; do
    [[ -f "$CKPT" ]] || continue
    EP=$(basename "$CKPT" .pth | sed 's/epoch_//')
    LOG="${OUT_DIR}/epoch_${EP}.log"
    JSON="${OUT_DIR}/epoch_${EP}.json"

    if [[ -f "$JSON" ]]; then
        echo "[skip] epoch $EP already evaluated -> $JSON"
    else
        echo "[run]  row=${ROW_TAG} epoch=${EP} gpu=${GPU}"
        CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. ~/anaconda3/envs/wedetect/bin/python test.py \
            "$CONFIG" "$CKPT" \
            --work-dir "${OUT_DIR}/epoch_${EP}_workdir" \
            > "$LOG" 2>&1 || { echo "FAILED epoch=$EP, see $LOG"; continue; }
        # Extract metric keys from log; tail bracket guards against multi-print
        python - "$LOG" "$JSON" <<'PYEOF'
import json, re, sys
log, out = sys.argv[1], sys.argv[2]
text = open(log).read()
# Grab last occurrence of each metric key
def grab(key):
    pat = re.compile(rf'{re.escape(key)}\s*[:=]\s*(-?\d+(?:\.\d+)?)')
    m = list(pat.finditer(text))
    return float(m[-1].group(1)) if m else float('nan')
metrics = {
    'overall/macro_mAP': grab('overall/macro_mAP'),
    'all_class/mAP': grab('all_class/mAP'),
    'overall/instance_weighted_mAP': grab('overall/instance_weighted_mAP'),
    'all_class/mAP_50': grab('all_class/mAP_50'),
    'all_class/mAP_75': grab('all_class/mAP_75'),
}
json.dump(metrics, open(out, 'w'), indent=2)
print(metrics)
PYEOF
    fi

    # Append to CSV
    python - "$JSON" "$EP" "$SUMMARY" <<'PYEOF'
import json, sys
js, ep, csv = sys.argv[1], sys.argv[2], sys.argv[3]
m = json.load(open(js))
with open(csv, 'a') as f:
    f.write(f'{ep},{m["overall/macro_mAP"]:.4f},{m["all_class/mAP"]:.4f},{m["overall/instance_weighted_mAP"]:.4f}\n')
PYEOF
done

echo ""
echo "===== ${ROW_TAG} corrected val summary ====="
column -t -s, "$SUMMARY"
echo ""
echo "best by organ_macro_mAP:"
sort -t, -k2 -gr "$SUMMARY" | grep -v '^epoch,' | head -1
