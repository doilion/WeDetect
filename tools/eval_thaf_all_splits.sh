#!/usr/bin/env bash
# Run the full Phase 3a/3b eval pipeline for one trained THAF variant.
#
# Usage:
#   bash tools/eval_thaf_all_splits.sh <encoder>
#   <encoder> = xlmr | biomedclip
#
# Steps:
#   1. Run base 25-class eval (excluding 5 dev30 negatives).
#   2. Run novel zero-shot eval on 4 splits (main_3 / pseudo_2 / hard_4 / full_5)
#      using tools/eval_novel_thaf.py (preserves trained fusion module).
#      Predictions persisted via --outfile-prefix for later score fusion.
#   3. Print a summary table of (encoder, split, mAP).
#
# Pre-conditions:
#   - work_dirs/wedetect_tiny_tct_ngc_dev30_thaf_<encoder>_2gpu/best_*.pth exists
#   - data/texts/tct_ngc_attr_<split>_eval.json (4 splits) exists in 5-attr format
#   - data/texts/tct_ngc_attr_<encoder>_per_attr.pth exists (referenced by config)
#
# Notes:
#   - The legacy step "build post-fused class cache via build_hier_class_embeddings.py"
#     was removed; eval_novel_thaf.py uses the trained backbone's fusion module
#     directly, so the cache is dead code (would be unused). See plan Fix-4.
#   - test_exclude_negative.py default exclude list is dev32-flavored and lacks
#     Urine-NHGUC; we override via --exclude-class-names to get the dev30 25-cls
#     headline. See plan Fix-2.

set -euo pipefail

ENCODER="${1:?usage: bash tools/eval_thaf_all_splits.sh <xlmr|biomedclip>}"
case "$ENCODER" in
    xlmr|biomedclip) ;;
    *) echo "ERROR: encoder must be xlmr or biomedclip"; exit 1 ;;
esac

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="config/wedetect_tiny_tct_ngc_dev30_thaf_${ENCODER}_2gpu.py"
WORK_DIR="work_dirs/wedetect_tiny_tct_ngc_dev30_thaf_${ENCODER}_2gpu"
SUMMARY="${WORK_DIR}/thaf_eval_summary.txt"

# dev30 5 negative classes to exclude from the 25-class headline:
DEV30_NEG_EXCLUDE='respiratory tract-Impurity,Serous effusion-Negative samples,Thyroid gland-Negative samples,Urine-NHGUC,TCT_CCD-normal'
DATA_ROOT="/home1/liwenjie/TCT_NGC/"
BASE_ANN="annotations/instances_test_base_clean_dev30.json"

BEST_CKPT=$(ls -t "${WORK_DIR}"/best_coco_bbox_mAP_epoch_*.pth 2>/dev/null | head -1 || true)
if [[ -z "$BEST_CKPT" ]]; then
    echo "ERROR: no best ckpt in $WORK_DIR"
    exit 1
fi
echo "[ckpt] $BEST_CKPT"

echo "[1/2] base 25-class eval (excluding 5 dev30 negatives)"
echo "=== THAF + ${ENCODER} eval $(date) ===" > "$SUMMARY"
echo >> "$SUMMARY"
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
echo "## Novel zero-shot eval" >> "$SUMMARY"

declare -A SPLIT_ANN=(
    [main_3]="annotations/instances_test_main_novel.json"
    [pseudo_2]="annotations/instances_test_pseudo_novel.json"
    [hard_4]="annotations/instances_hard_test.json"
    [full_5]="annotations/instances_test_novel.json"
)

for SPLIT in main_3 pseudo_2 hard_4 full_5; do
    EVAL_DIR="${WORK_DIR}/eval_novel_${SPLIT}_thaf"
    echo "  [$SPLIT] $(date)"
    PYTHONPATH="$REPO_ROOT" python tools/eval_novel_thaf.py \
        --config "$CONFIG" \
        --checkpoint "$BEST_CKPT" \
        --data-root "$DATA_ROOT" \
        --ann-file "${SPLIT_ANN[$SPLIT]}" \
        --text-json "data/texts/tct_ngc_attr_${SPLIT}_eval.json" \
        --outfile-prefix "${EVAL_DIR}/preds_thaf" \
        --work-dir "$EVAL_DIR" 2>&1 | tee -a "$SUMMARY" | grep -E 'bbox_mAP|_precision' | tail -1
    echo "  [$SPLIT] done" >> "$SUMMARY"
    echo >> "$SUMMARY"
done

echo
echo "=== SUMMARY ==="
echo "  Encoder: $ENCODER"
echo "  Best ckpt: $BEST_CKPT"
echo "  Predictions: ${WORK_DIR}/eval_novel_*_thaf/preds_thaf.bbox.json (for score fusion)"
echo
grep -E 'bbox_mAP:' "$SUMMARY" | tail -8
