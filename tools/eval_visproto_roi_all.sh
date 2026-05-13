#!/usr/bin/env bash
# Phase A: build + eval 5-shot visual prototypes using the new ROI path
# (KeepRatioResize+LetterResize + ROIAlign at FPN level) on noTHAF ckpt × 4 novel splits.
#
# Compares against the legacy path (cv2.resize + spatial global mean) which
# produced novel mean (9 unique) = 0.122 leakage / 0.123 strict.
#
# Output: work_dirs/wedetect_tiny_tct_ngc_dev30_biomedclip_noTHAF_2gpu/eval_visproto_roi_summary.txt
#
# Usage:
#   bash tools/eval_visproto_roi_all.sh              # full 4 novel splits
#   SPLITS='main_3' bash tools/eval_visproto_roi_all.sh   # smoke test on 1 split

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

source /home/25_liwenjie/anaconda3/etc/profile.d/conda.sh
conda activate wedetect

# Single GPU is enough for Phase A inference-only. Use GPU 0.
export CUDA_VISIBLE_DEVICES=0

BASE_CONFIG="config/wedetect_tiny_tct_ngc_dev30_biomedclip_noTHAF_2gpu.py"
BASE_CKPT="work_dirs/wedetect_tiny_tct_ngc_dev30_biomedclip_noTHAF_2gpu/best_coco_bbox_mAP_epoch_11.pth"
WORK_DIR="work_dirs/wedetect_tiny_tct_ngc_dev30_biomedclip_noTHAF_2gpu"

DATA_ROOT="/home1/liwenjie/TCT_NGC/"
TEXT_DIR="data/texts"

# Phase A hyperparameters
ROI_SIZE="${ROI_SIZE:-7}"
ROI_EXPAND="${ROI_EXPAND:-1.0}"  # expand bbox before ROIAlign (1.5 = legacy-like context)
BG_LAMBDA="${BG_LAMBDA:-0.0}"    # 0.0 = pure foreground ROI (Phase A baseline)
N_SHOT="${N_SHOT:-5}"
SEED="${SEED:-20260509}"         # match legacy seed for apples-to-apples
TAG_SUFFIX="${TAG_SUFFIX:-}"     # optional suffix (e.g. _exp15 for roi_expand=1.5)
TAG="visproto_roi${TAG_SUFFIX}"
SUMMARY="${WORK_DIR}/eval_${TAG}_summary.txt"

# Optional: override splits list for smoke test
SPLITS="${SPLITS:-main_3 pseudo_2 hard_4 full_5}"

declare -A SPLIT_ANN=(
    [main_3]="annotations/instances_test_main_novel.json"
    [pseudo_2]="annotations/instances_test_pseudo_novel.json"
    [hard_4]="annotations/instances_hard_test.json"
    [full_5]="annotations/instances_test_novel.json"
)

if [[ ! -f "$BASE_CKPT" ]]; then
    echo "ERROR: missing $BASE_CKPT"
    exit 1
fi

echo "=== Phase A: ROI visproto eval ($(date)) ===" | tee "$SUMMARY"
echo "config: $BASE_CONFIG" | tee -a "$SUMMARY"
echo "ckpt:   $BASE_CKPT" | tee -a "$SUMMARY"
echo "tag=$TAG  roi_size=$ROI_SIZE  roi_expand=$ROI_EXPAND  bg_lambda=$BG_LAMBDA  n_shot=$N_SHOT  seed=$SEED" | tee -a "$SUMMARY"
echo "splits: $SPLITS" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

# ────────────────────────────────────────────────────────────────────
# Step 1: build novel visproto × splits (new ROI path)
# ────────────────────────────────────────────────────────────────────
echo "[1/2] build novel visproto (ROI path)" | tee -a "$SUMMARY"
echo "## Step 1 — build novel visproto" >> "$SUMMARY"
for SPLIT in $SPLITS; do
    OUT="${TEXT_DIR}/tct_ngc_novel_${SPLIT}_${TAG}.pth"
    if [[ -f "$OUT" && "${FORCE_REBUILD:-0}" != "1" ]]; then
        echo "  [$SPLIT] cache exists, skip: $OUT  (set FORCE_REBUILD=1 to overwrite)" | tee -a "$SUMMARY"
        continue
    fi
    echo "  [$SPLIT-build] → $OUT" | tee -a "$SUMMARY"
    PYTHONPATH="$REPO_ROOT" python tools/build_visual_prototype.py \
        --config "$BASE_CONFIG" \
        --checkpoint "$BASE_CKPT" \
        --ann-file "${SPLIT_ANN[$SPLIT]}" \
        --data-root "$DATA_ROOT" \
        --img-prefix "images/" \
        --text-json "${TEXT_DIR}/tct_ngc_novel_${SPLIT}.json" \
        --out "$OUT" \
        --n-per-class "$N_SHOT" \
        --seed "$SEED" \
        --roi-size "$ROI_SIZE" \
        --roi-expand "$ROI_EXPAND" \
        --bg-lambda "$BG_LAMBDA" \
        --save-diag \
        2>&1 | tee -a "$SUMMARY" | grep -E '\[(ok|skip|warn|mode)\]'
done
echo "" | tee -a "$SUMMARY"

# ────────────────────────────────────────────────────────────────────
# Step 2: eval × splits (strict held-out: exclude exemplar images via holdout JSON)
# ────────────────────────────────────────────────────────────────────
echo "[2/2] novel eval (strict held-out)" | tee -a "$SUMMARY"
echo "## Step 2 — novel eval (strict held-out)" >> "$SUMMARY"
for SPLIT in $SPLITS; do
    VISPROTO="${TEXT_DIR}/tct_ngc_novel_${SPLIT}_${TAG}.pth"
    HOLDOUT="${VISPROTO%.pth}.holdout_anns.json"
    EVAL_DIR="${WORK_DIR}/eval_novel_${SPLIT}_${TAG}_strict"

    # Build strict ann (excludes images containing exemplar bboxes)
    STRICT_ANN="/tmp/strict_${SPLIT}_${TAG}.json"
    PYTHONPATH="$REPO_ROOT" python tools/build_strict_zeroshot_ann.py \
        --ann "${DATA_ROOT}${SPLIT_ANN[$SPLIT]}" \
        --holdout "$HOLDOUT" \
        --out "$STRICT_ANN" 2>&1 | tail -2 | sed "s|^|  $SPLIT strict-ann: |"

    echo "  [$SPLIT-eval] → $EVAL_DIR" | tee -a "$SUMMARY"
    PYTHONPATH="$REPO_ROOT" python tools/eval_novel_split.py \
        --config "$BASE_CONFIG" \
        --checkpoint "$BASE_CKPT" \
        --data-root "$DATA_ROOT" \
        --ann-file "$STRICT_ANN" \
        --text-json "${TEXT_DIR}/tct_ngc_novel_${SPLIT}.json" \
        --text-emb "$VISPROTO" \
        --outfile-prefix "${EVAL_DIR}/preds_roi" \
        --work-dir "$EVAL_DIR" \
        2>&1 | tee -a "$SUMMARY" | grep -E 'bbox_mAP_copypaste' | tail -1 | sed "s|^|  $SPLIT mAP: |"
done

echo "" | tee -a "$SUMMARY"
echo "=== DONE $(date) ===" | tee -a "$SUMMARY"
echo ""
echo "=== summary mAPs ==="
grep -E 'bbox_mAP_copypaste' "$SUMMARY" | tail -16
