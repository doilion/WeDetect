#!/usr/bin/env bash
# THAF BiomedCLIP + 5-shot visproto-only eval × 4 novel splits.
#
# Compares to noTHAF + visproto baseline (avg 0.105). Tests whether the THAF
# checkpoint's image_encoder + head BN-running-stats give better/worse
# response to visproto class vectors than the noTHAF checkpoint.
#
# Eval-time text_model is overridden to PseudoLanguageBackbone (config:
# config/wedetect_tiny_tct_ngc_dev30_thaf_biomedclip_eval_visproto.py)
# so the visproto-only path bypasses THAF's text fusion module (which is
# moot here — visproto comes from image FPN, not text). State_dict
# strict=False silently ignores the fusion params at load time.
#
# Usage:
#   nohup bash tools/eval_thaf_biomedclip_visproto.sh > /tmp/thaf_visproto_eval.log 2>&1 &

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

source /home/25_liwenjie/anaconda3/etc/profile.d/conda.sh
conda activate wedetect

# Hard pin to GPU 0+1 only (yoloe owns GPU 2+3)
export CUDA_VISIBLE_DEVICES=0
echo "[launch] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

CONFIG_TRAIN="config/wedetect_tiny_tct_ngc_dev30_thaf_biomedclip_2gpu.py"
# Eval-time architecture container: noTHAF config (PseudoLanguageBackbone for
# text + same image_encoder + neck + bbox_head). Loading the THAF ckpt into
# this config gives us the THAF-trained image_encoder + head BN/scale/bias
# combined with visproto class vectors at inference. THAF's fusion params
# (backbone.text_model.fusion_query etc) are silently ignored via strict=False.
# (The original thaf_biomedclip_eval_visproto.py inherited THAF's test_pipeline
# which wraps texts in attr-nested format → unhashable cache_key crash.)
CONFIG_EVAL="config/wedetect_tiny_tct_ngc_dev30_biomedclip_noTHAF_2gpu.py"
CKPT="work_dirs/wedetect_tiny_tct_ngc_dev30_thaf_biomedclip_2gpu/best_coco_bbox_mAP_epoch_10.pth"
WORK_DIR="work_dirs/wedetect_tiny_tct_ngc_dev30_thaf_biomedclip_2gpu"
TEXT_DIR="data/texts"
TAG="thaf_biomedclip"
SUMMARY="${WORK_DIR}/thaf_visproto_eval_summary.txt"

DATA_ROOT_TEST="/home1/liwenjie/TCT_NGC/"

declare -A SPLIT_ANN=(
    [main_3]="annotations/instances_test_main_novel.json"
    [pseudo_2]="annotations/instances_test_pseudo_novel.json"
    [hard_4]="annotations/instances_hard_test.json"
    [full_5]="annotations/instances_test_novel.json"
)

echo "=== THAF + visproto eval $(date) ===" | tee "$SUMMARY"
echo "ckpt: $CKPT" | tee -a "$SUMMARY"
echo "train config: $CONFIG_TRAIN  (used for visproto build, image_encoder forward)" | tee -a "$SUMMARY"
echo "eval config:  $CONFIG_EVAL   (text_model → PseudoLanguageBackbone, no fusion)" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

# ────────────────────────────────────────────────────────────────────
# Step 1: build novel visproto × 4 (uses train config, image encoder only)
# ────────────────────────────────────────────────────────────────────
echo
echo "[1/2] build novel visproto × 4 splits — THAF image encoder"
echo "## Step 1 — build visproto × 4 (THAF image encoder)" >> "$SUMMARY"
for SPLIT in main_3 pseudo_2 hard_4 full_5; do
    OUT="${TEXT_DIR}/tct_ngc_novel_${SPLIT}_visproto_emb_${TAG}.pth"
    if [[ -f "$OUT" ]]; then
        echo "  [$SPLIT] cache exists, skip: $OUT" | tee -a "$SUMMARY"
        continue
    fi
    echo "  [$SPLIT-build] → $OUT" | tee -a "$SUMMARY"
    PYTHONPATH="$REPO_ROOT" python tools/build_visual_prototype.py \
        --config "$CONFIG_TRAIN" \
        --checkpoint "$CKPT" \
        --ann-file "${SPLIT_ANN[$SPLIT]}" \
        --data-root "$DATA_ROOT_TEST" \
        --img-prefix "images/" \
        --text-json "${TEXT_DIR}/tct_ngc_novel_${SPLIT}.json" \
        --out "$OUT" \
        --n-per-class 5 \
        --seed 20260509 2>&1 | tee -a "$SUMMARY" | tail -2
done
echo "" | tee -a "$SUMMARY"

# ────────────────────────────────────────────────────────────────────
# Step 2: visproto-only eval × 4 splits (eval config bypasses THAF text)
# ────────────────────────────────────────────────────────────────────
echo
echo "[2/2] visproto-only eval × 4 splits"
echo "## Step 2 — visproto-only eval × 4 splits" >> "$SUMMARY"
for SPLIT in main_3 pseudo_2 hard_4 full_5; do
    EVAL_DIR="${WORK_DIR}/eval_novel_${SPLIT}_visproto_${TAG}"
    VISPROTO="${TEXT_DIR}/tct_ngc_novel_${SPLIT}_visproto_emb_${TAG}.pth"
    echo "  [$SPLIT-eval]" | tee -a "$SUMMARY"
    PYTHONPATH="$REPO_ROOT" python tools/eval_novel_split.py \
        --config "$CONFIG_EVAL" \
        --checkpoint "$CKPT" \
        --data-root "$DATA_ROOT_TEST" \
        --ann-file "${SPLIT_ANN[$SPLIT]}" \
        --text-json "${TEXT_DIR}/tct_ngc_novel_${SPLIT}.json" \
        --text-emb "$VISPROTO" \
        --outfile-prefix "${EVAL_DIR}/preds_visproto" \
        --work-dir "$EVAL_DIR" 2>&1 | tee -a "$SUMMARY" | grep -E 'bbox_mAP:' | tail -1 | sed "s|^|  $SPLIT thaf visproto: |"
done
echo "" | tee -a "$SUMMARY"

echo "=== DONE $(date) ===" | tee -a "$SUMMARY"
echo
echo "=== summary mAP lines ==="
grep -E 'thaf visproto:' "$SUMMARY" | head -10
