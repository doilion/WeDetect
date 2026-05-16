#!/usr/bin/env bash
# Run full eval pipeline for PCW (Per-Class Weights, Option 3a) ckpt:
#   1. Base 25-cls (excluding 5 dev30 negatives) — to compare against noTHAF 0.321
#   2. Novel 4 splits (main_3 / pseudo_2 / hard_4 / full_5) — text path via PCW
#      backbone's per-class softmax weights (novel classes use fallback row)
#
# Adapted from tools/eval_thaf_all_splits.sh — PCW uses the same 5-attr text
# format as THAF, but the text_model is PseudoPerClassWeightedBiomedCLIPLanguageBackbone
# instead of THAF's PseudoHierarchicalBiomedCLIPLanguageBackbone. eval_novel_thaf.py
# preserves the trained text_model (doesn't replace with PseudoLanguageBackbone),
# so it works for PCW too.
#
# Output:
#   work_dirs/wedetect_tiny_tct_ngc_dev30_pcw_biomedclip_2gpu/pcw_eval_summary.txt
#
# Usage:
#   nohup bash tools/eval_pcw_all_splits.sh > /tmp/pcw_eval.log 2>&1 &

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

source /home/25_liwenjie/anaconda3/etc/profile.d/conda.sh
conda activate wedetect

# Hard pin GPU 0 (single GPU eval is enough)
export CUDA_VISIBLE_DEVICES=0

CONFIG="config/wedetect_tiny_tct_ngc_dev30_pcw_biomedclip_2gpu.py"
WORK_DIR="work_dirs/wedetect_tiny_tct_ngc_dev30_pcw_biomedclip_2gpu"
SUMMARY="${WORK_DIR}/pcw_eval_summary.txt"

DEV30_NEG_EXCLUDE='respiratory tract-Impurity,Serous effusion-Negative samples,Thyroid gland-Negative samples,Urine-NHGUC,TCT_CCD-normal'
DATA_ROOT="/home1/liwenjie/TCT_NGC/"
BASE_ANN="annotations/instances_test_base_clean_dev30.json"

BEST_CKPT=$(ls -t "${WORK_DIR}"/best_coco_bbox_mAP_epoch_*.pth 2>/dev/null | head -1 || true)
if [[ -z "$BEST_CKPT" ]]; then
    echo "ERROR: no best ckpt in $WORK_DIR"
    exit 1
fi
echo "[ckpt] $BEST_CKPT"

echo "=== PCW (Option 3a) eval $(date) ===" > "$SUMMARY"
echo "ckpt: $BEST_CKPT" >> "$SUMMARY"
echo >> "$SUMMARY"

echo "[1/2] base 25-class eval (excluding 5 dev30 negatives)"
echo "## Base eval (25 classes excl. dev30 5 negatives)" >> "$SUMMARY"
PYTHONPATH="$REPO_ROOT" python test_exclude_negative.py \
    --config "$CONFIG" \
    --checkpoint "$BEST_CKPT" \
    --data-root "$DATA_ROOT" \
    --ann-file "$BASE_ANN" \
    --exclude-class-names "$DEV30_NEG_EXCLUDE" \
    --work-dir "${WORK_DIR}/eval_base_25cls" 2>&1 | tee -a "$SUMMARY" | tail -3
echo >> "$SUMMARY"

echo "[2/2] novel splits eval (main_3, pseudo_2, hard_4, full_5)"
echo "## Novel zero-shot eval (PCW text path: novel classes use fallback row uniform softmax = mean pool)" >> "$SUMMARY"

declare -A SPLIT_ANN=(
    [main_3]="annotations/instances_test_main_novel.json"
    [pseudo_2]="annotations/instances_test_pseudo_novel.json"
    [hard_4]="annotations/instances_hard_test.json"
    [full_5]="annotations/instances_test_novel.json"
)

for SPLIT in main_3 pseudo_2 hard_4 full_5; do
    EVAL_DIR="${WORK_DIR}/eval_novel_${SPLIT}_pcw"
    echo "  [$SPLIT] $(date)" | tee -a "$SUMMARY"
    PYTHONPATH="$REPO_ROOT" python tools/eval_novel_thaf.py \
        --config "$CONFIG" \
        --checkpoint "$BEST_CKPT" \
        --data-root "$DATA_ROOT" \
        --ann-file "${SPLIT_ANN[$SPLIT]}" \
        --text-json "data/texts/tct_ngc_attr_${SPLIT}_eval.json" \
        --outfile-prefix "${EVAL_DIR}/preds_pcw" \
        --work-dir "$EVAL_DIR" 2>&1 | tee -a "$SUMMARY" | grep -E 'bbox_mAP:|_precision' | tail -1
    echo "  [$SPLIT] done" >> "$SUMMARY"
    echo >> "$SUMMARY"
done

echo
echo "=== SUMMARY ==="
echo "  Best ckpt: $BEST_CKPT"
echo "  Predictions: ${WORK_DIR}/eval_novel_*_pcw/preds_pcw.bbox.json"
echo
grep -E 'bbox_mAP:' "$SUMMARY" | tail -8
