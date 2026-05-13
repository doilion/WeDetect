#!/usr/bin/env bash
# Full eval of SAVPE-trained visproto on TCT_NGC dev30 4 novel splits.
#
# Run after `tools/train_savpe_cell_contrastive.py` finishes:
#   1. Build SAVPE-visproto for each of 4 novel splits (same seed as
#      inference-only visproto for apples-to-apples)
#   2. Eval each split with leakage version (matches existing visproto eval)
#   3. Eval each split strict zero-shot (reuses /tmp/strict_*.json built earlier)
#
# Writes summary to work_dirs/savpe_cellctr_v1/eval_summary.txt
#
# Usage:
#   nohup bash tools/eval_savpe_visproto_all.sh > /tmp/savpe_eval.log 2>&1 &

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

source /home/25_liwenjie/anaconda3/etc/profile.d/conda.sh
conda activate wedetect

BASE_CONFIG="config/wedetect_tiny_tct_ngc_dev30_biomedclip_noTHAF_2gpu.py"
BASE_CKPT="work_dirs/wedetect_tiny_tct_ngc_dev30_biomedclip_noTHAF_2gpu/best_coco_bbox_mAP_epoch_11.pth"
SAVPE_CKPT="work_dirs/savpe_cellctr_v1/savpe_final.pth"

if [[ ! -f "$SAVPE_CKPT" ]]; then
    echo "ERROR: missing $SAVPE_CKPT — SAVPE training not yet finished"
    exit 1
fi

DATA_ROOT="/home1/liwenjie/TCT_NGC/"
TEXT_DIR="data/texts"
SUMMARY="work_dirs/savpe_cellctr_v1/eval_summary.txt"

declare -A SPLIT_ANN=(
    [main_3]="annotations/instances_test_main_novel.json"
    [pseudo_2]="annotations/instances_test_pseudo_novel.json"
    [hard_4]="annotations/instances_hard_test.json"
    [full_5]="annotations/instances_test_novel.json"
)

declare -A STRICT_ANN=(
    [main_3]="/tmp/strict_main_3.json"
    [pseudo_2]="/tmp/strict_pseudo_2.json"
    [hard_4]="/tmp/strict_hard_4.json"
    [full_5]="/tmp/strict_full_5.json"
)

GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU

echo "=== SAVPE-visproto eval $(date) ===" > "$SUMMARY"
echo "SAVPE ckpt: $SAVPE_CKPT" >> "$SUMMARY"
echo "Base ckpt: $BASE_CKPT" >> "$SUMMARY"
echo "GPU: $GPU" >> "$SUMMARY"
echo "" >> "$SUMMARY"

# ────────────────────────────────────────────────────────────────────
# Step 1: build SAVPE-visproto for 4 novel splits (using SAVPE module)
# ────────────────────────────────────────────────────────────────────
echo
echo "[1/3] build SAVPE-visproto × 4 splits"
echo "## Step 1 — SAVPE-visproto build" >> "$SUMMARY"
for SPLIT in main_3 pseudo_2 hard_4 full_5; do
    OUT="${TEXT_DIR}/tct_ngc_novel_${SPLIT}_savpe_visproto.pth"
    if [[ -f "$OUT" ]]; then
        echo "  [$SPLIT] exists, skip"
        continue
    fi
    echo "  [$SPLIT] → $OUT" | tee -a "$SUMMARY"
    PYTHONPATH="$REPO_ROOT" python tools/build_savpe_visproto.py \
        --base-config "$BASE_CONFIG" \
        --base-ckpt "$BASE_CKPT" \
        --savpe-ckpt "$SAVPE_CKPT" \
        --ann-file "${SPLIT_ANN[$SPLIT]}" \
        --data-root "$DATA_ROOT" \
        --img-prefix "images/" \
        --text-json "${TEXT_DIR}/tct_ngc_novel_${SPLIT}.json" \
        --out "$OUT" \
        --n-per-class 5 2>&1 | tee -a "$SUMMARY" | tail -3
done
echo >> "$SUMMARY"

# ────────────────────────────────────────────────────────────────────
# Step 2: eval each split with leakage version (matches current baseline)
# ────────────────────────────────────────────────────────────────────
echo
echo "[2/3] eval × 4 splits — leakage version (matches 0.105 baseline)"
echo "## Step 2 — SAVPE-visproto eval (leakage)" >> "$SUMMARY"
for SPLIT in main_3 pseudo_2 hard_4 full_5; do
    EVAL_DIR="work_dirs/savpe_cellctr_v1/eval_${SPLIT}_leakage"
    VISPROTO="${TEXT_DIR}/tct_ngc_novel_${SPLIT}_savpe_visproto.pth"
    echo "  [$SPLIT-leakage]" | tee -a "$SUMMARY"
    PYTHONPATH="$REPO_ROOT" python tools/eval_novel_split.py \
        --config "$BASE_CONFIG" \
        --checkpoint "$BASE_CKPT" \
        --data-root "$DATA_ROOT" \
        --ann-file "${SPLIT_ANN[$SPLIT]}" \
        --text-json "${TEXT_DIR}/tct_ngc_novel_${SPLIT}.json" \
        --text-emb "$VISPROTO" \
        --outfile-prefix "${EVAL_DIR}/preds" \
        --work-dir "$EVAL_DIR" 2>&1 | tee -a "$SUMMARY" \
        | grep bbox_mAP_copypaste | tail -1 | sed "s|^|  $SPLIT leakage: |"
done
echo >> "$SUMMARY"

# ────────────────────────────────────────────────────────────────────
# Step 3: strict zero-shot eval (reuse /tmp/strict_*.json)
# ────────────────────────────────────────────────────────────────────
echo
echo "[3/3] eval × 4 splits — strict zero-shot"
echo "## Step 3 — SAVPE-visproto eval (strict zero-shot)" >> "$SUMMARY"
for SPLIT in main_3 pseudo_2 hard_4 full_5; do
    EVAL_DIR="work_dirs/savpe_cellctr_v1/eval_${SPLIT}_strict"
    VISPROTO="${TEXT_DIR}/tct_ngc_novel_${SPLIT}_savpe_visproto.pth"
    STRICT_FILE="${STRICT_ANN[$SPLIT]}"
    if [[ ! -f "$STRICT_FILE" ]]; then
        echo "  [$SPLIT-strict] WARN: missing $STRICT_FILE, run tools/build_strict_zeroshot_ann.py first" | tee -a "$SUMMARY"
        continue
    fi
    echo "  [$SPLIT-strict]" | tee -a "$SUMMARY"
    PYTHONPATH="$REPO_ROOT" python tools/eval_novel_split.py \
        --config "$BASE_CONFIG" \
        --checkpoint "$BASE_CKPT" \
        --data-root "$DATA_ROOT" \
        --ann-file "$STRICT_FILE" \
        --text-json "${TEXT_DIR}/tct_ngc_novel_${SPLIT}.json" \
        --text-emb "$VISPROTO" \
        --outfile-prefix "${EVAL_DIR}/preds" \
        --work-dir "$EVAL_DIR" 2>&1 | tee -a "$SUMMARY" \
        | grep bbox_mAP_copypaste | tail -1 | sed "s|^|  $SPLIT strict: |"
done
echo >> "$SUMMARY"

echo "=== DONE $(date) ===" >> "$SUMMARY"
echo "[done] $(date) → see $SUMMARY"

echo ""
echo "=== Final summary ==="
grep -E 'leakage:|strict:' "$SUMMARY" | head -16
