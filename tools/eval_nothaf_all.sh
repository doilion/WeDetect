#!/usr/bin/env bash
# Comprehensive noTHAF (BiomedCLIP + 1 PSC) eval suite — mirrors
# `tools/eval_baseline_all.sh` for the noTHAF ckpt to give an
# apples-to-apples comparison with clean dev30 baseline.
#
# Step layout:
#   1. base 25-cls                                    (SKIP if already done)
#   2. v1 text baseline × 4 novel splits              (BiomedCLIP novel embs)
#   3. base-30 visproto from train (noTHAF ckpt)
#   4. novel visproto × 4 splits (noTHAF ckpt, 5-shot from test, with leakage)
#   5. visproto-only eval × 4 splits
#   6. (SKIP — Procrustes refit, DEAD-5)
#   7. (SKIP — Procrustes calfused, DEAD-5)
#   8. score fusion × 4 splits (text preds + visproto preds, per-class merge)
#
# All results tee'd to a single summary file for ablation table.
#
# Uses 1 GPU (auto-picks first idle GPU with <2GB used).
# Designed for nohup-style background launch:
#   nohup bash tools/eval_nothaf_all.sh > /tmp/nothaf_eval_all.log 2>&1 &

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

source /home/25_liwenjie/anaconda3/etc/profile.d/conda.sh
conda activate wedetect

CONFIG="config/wedetect_tiny_tct_ngc_dev30_biomedclip_noTHAF_2gpu.py"
WORK_DIR="work_dirs/wedetect_tiny_tct_ngc_dev30_biomedclip_noTHAF_2gpu"
CKPT="${WORK_DIR}/best_coco_bbox_mAP_epoch_11.pth"

if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: missing $CKPT"
    exit 1
fi

DATA_ROOT_640="/home1/liwenjie/TCT_NGC_640/"
DATA_ROOT_TEST="/home1/liwenjie/TCT_NGC/"
TRAIN_ANN="annotations/instances_train_dev_disjoint_dev30.json"
BASE_TEST_ANN="annotations/instances_test_base_clean_dev30.json"
DEV30_NEG_EXCLUDE='respiratory tract-Impurity,Serous effusion-Negative samples,Thyroid gland-Negative samples,Urine-NHGUC,TCT_CCD-normal'

TEXT_DIR="data/texts"
V1_TEXT_BASE="${TEXT_DIR}/tct_ngc_fullnames_30_embeddings_biomedclip.pth"  # base 30 cache (used by training)
V1_TEXT_JSON_BASE="${TEXT_DIR}/tct_ngc_fullnames_30.json"
TAG="biomedclip_noTHAF"
SUMMARY="${WORK_DIR}/baseline_eval_summary.txt"

declare -A SPLIT_ANN=(
    [main_3]="annotations/instances_test_main_novel.json"
    [pseudo_2]="annotations/instances_test_pseudo_novel.json"
    [hard_4]="annotations/instances_hard_test.json"
    [full_5]="annotations/instances_test_novel.json"
)

# Wait for an idle GPU (< 2GB used). Returns its index.
wait_for_free_gpu() {
    while true; do
        local free_gpu
        free_gpu=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
                  | awk -F',' '{gsub(/ /,""); if ($2 < 2000) {print $1; exit 0}}')
        if [[ -n "$free_gpu" ]]; then
            echo "$free_gpu"
            return 0
        fi
        sleep 60
    done
}

GPU=$(wait_for_free_gpu)
export CUDA_VISIBLE_DEVICES=$GPU

echo "=== noTHAF BiomedCLIP comprehensive eval $(date) ===" > "$SUMMARY"
echo "ckpt: $CKPT" >> "$SUMMARY"
echo "GPU: $GPU" >> "$SUMMARY"
echo >> "$SUMMARY"

# ────────────────────────────────────────────────────────────────────
# Step 1: test_base 25-cls — skip if already done
# ────────────────────────────────────────────────────────────────────
echo
echo "[1/6] test_base 25-cls"
echo "## Step 1 — test_base 25-cls" >> "$SUMMARY"
if [[ -f "${WORK_DIR}/eval_base_25cls.log" ]] && grep -q 'bbox_mAP_copypaste' "${WORK_DIR}/eval_base_25cls.log"; then
    echo "  (already done, copying result)"
    echo "  (re-using ${WORK_DIR}/eval_base_25cls.log)" >> "$SUMMARY"
    grep 'bbox_mAP_copypaste' "${WORK_DIR}/eval_base_25cls.log" | tail -1 >> "$SUMMARY"
else
    PYTHONPATH="$REPO_ROOT" python test_exclude_negative.py \
        --config "$CONFIG" \
        --checkpoint "$CKPT" \
        --data-root "$DATA_ROOT_TEST" \
        --ann-file "$BASE_TEST_ANN" \
        --exclude-class-names "$DEV30_NEG_EXCLUDE" \
        --work-dir "${WORK_DIR}/eval_base_25cls" 2>&1 | tee -a "$SUMMARY" | grep -E 'bbox_mAP_copypaste' | tail -1
fi
echo >> "$SUMMARY"

# ────────────────────────────────────────────────────────────────────
# Step 2: v1 text baseline × 4 splits — skip splits already done
# ────────────────────────────────────────────────────────────────────
echo
echo "[2/6] v1 text baseline × 4 splits"
echo "## Step 2 — v1 text baseline (BiomedCLIP novel embs) × 4 splits" >> "$SUMMARY"
for SPLIT in main_3 pseudo_2 hard_4 full_5; do
    EVAL_DIR="${WORK_DIR}/eval_novel_${SPLIT}"
    if [[ -f "${EVAL_DIR}/preds_v1text.bbox.json" ]]; then
        echo "  [$SPLIT-v1text] already done, parsing log" | tee -a "$SUMMARY"
        LOG=$(find "$EVAL_DIR" -name '*.log' 2>/dev/null | head -1)
        [[ -n "$LOG" ]] && grep 'coco/bbox_mAP:' "$LOG" | tail -1 >> "$SUMMARY"
        continue
    fi
    echo "  [$SPLIT-v1text]" | tee -a "$SUMMARY"
    PYTHONPATH="$REPO_ROOT" python tools/eval_novel_split.py \
        --config "$CONFIG" \
        --checkpoint "$CKPT" \
        --data-root "$DATA_ROOT_TEST" \
        --ann-file "${SPLIT_ANN[$SPLIT]}" \
        --text-json "${TEXT_DIR}/tct_ngc_novel_${SPLIT}.json" \
        --text-emb "${TEXT_DIR}/tct_ngc_novel_${SPLIT}_emb_biomedclip.pth" \
        --outfile-prefix "${EVAL_DIR}/preds_v1text" \
        --work-dir "$EVAL_DIR" 2>&1 | tee -a "$SUMMARY" | grep -E 'bbox_mAP:' | tail -1
done
echo >> "$SUMMARY"

# ────────────────────────────────────────────────────────────────────
# Step 3: base-30 visproto from train (noTHAF image encoder)
# ────────────────────────────────────────────────────────────────────
BASE_VISPROTO="${TEXT_DIR}/tct_ngc_base30_visproto_train_${TAG}.pth"
echo
echo "[3/6] build base-30 visproto from train → ${BASE_VISPROTO}"
echo "## Step 3 — base-30 visproto from train (noTHAF ckpt)" >> "$SUMMARY"
if [[ -f "$BASE_VISPROTO" ]]; then
    echo "  (already exists, skip)" | tee -a "$SUMMARY"
else
    PYTHONPATH="$REPO_ROOT" python tools/build_visual_prototype.py \
        --config "$CONFIG" \
        --checkpoint "$CKPT" \
        --ann-file "$TRAIN_ANN" \
        --data-root "$DATA_ROOT_640" \
        --img-prefix "images/" \
        --text-json "$V1_TEXT_JSON_BASE" \
        --out "$BASE_VISPROTO" \
        --n-per-class 5 2>&1 | tee -a "$SUMMARY" | tail -3
fi
echo >> "$SUMMARY"

# ────────────────────────────────────────────────────────────────────
# Step 4: novel visproto × 4 splits (5-shot from test, with leakage)
# ────────────────────────────────────────────────────────────────────
echo
echo "[4/6] build novel visproto × 4 splits"
echo "## Step 4 — novel visproto × 4 splits (noTHAF ckpt)" >> "$SUMMARY"
for SPLIT in main_3 pseudo_2 hard_4 full_5; do
    NOVEL_VISPROTO="${TEXT_DIR}/tct_ngc_novel_${SPLIT}_visproto_emb_${TAG}.pth"
    if [[ -f "$NOVEL_VISPROTO" ]]; then
        echo "  [$SPLIT-visproto] exists, skip" | tee -a "$SUMMARY"
        continue
    fi
    echo "  [$SPLIT-visproto-build] → $NOVEL_VISPROTO" | tee -a "$SUMMARY"
    PYTHONPATH="$REPO_ROOT" python tools/build_visual_prototype.py \
        --config "$CONFIG" \
        --checkpoint "$CKPT" \
        --ann-file "${SPLIT_ANN[$SPLIT]}" \
        --data-root "$DATA_ROOT_TEST" \
        --img-prefix "images/" \
        --text-json "${TEXT_DIR}/tct_ngc_novel_${SPLIT}.json" \
        --out "$NOVEL_VISPROTO" \
        --n-per-class 5 2>&1 | tee -a "$SUMMARY" | tail -2
done
echo >> "$SUMMARY"

# ────────────────────────────────────────────────────────────────────
# Step 5: visproto-only eval × 4 splits
# ────────────────────────────────────────────────────────────────────
echo
echo "[5/6] visproto-only eval × 4 splits"
echo "## Step 5 — visproto-only eval × 4 splits (noTHAF ckpt)" >> "$SUMMARY"
for SPLIT in main_3 pseudo_2 hard_4 full_5; do
    EVAL_DIR="${WORK_DIR}/eval_novel_${SPLIT}_visproto"
    NOVEL_VISPROTO="${TEXT_DIR}/tct_ngc_novel_${SPLIT}_visproto_emb_${TAG}.pth"
    if [[ -f "${EVAL_DIR}/preds_visproto.bbox.json" ]]; then
        echo "  [$SPLIT-visproto-eval] already done" | tee -a "$SUMMARY"
        continue
    fi
    if [[ ! -f "$NOVEL_VISPROTO" ]]; then
        echo "  [$SPLIT-visproto-eval] MISSING $NOVEL_VISPROTO, skip" | tee -a "$SUMMARY"
        continue
    fi
    echo "  [$SPLIT-visproto-eval]" | tee -a "$SUMMARY"
    PYTHONPATH="$REPO_ROOT" python tools/eval_novel_split.py \
        --config "$CONFIG" \
        --checkpoint "$CKPT" \
        --data-root "$DATA_ROOT_TEST" \
        --ann-file "${SPLIT_ANN[$SPLIT]}" \
        --text-json "${TEXT_DIR}/tct_ngc_novel_${SPLIT}.json" \
        --text-emb "$NOVEL_VISPROTO" \
        --outfile-prefix "${EVAL_DIR}/preds_visproto" \
        --work-dir "$EVAL_DIR" 2>&1 | tee -a "$SUMMARY" | grep -E 'bbox_mAP:' | tail -1
done
echo >> "$SUMMARY"

# ────────────────────────────────────────────────────────────────────
# Step 8: score fusion × 4 splits (text v1 + visproto per-class merge)
# ────────────────────────────────────────────────────────────────────
echo
echo "[6/6] score fusion × 4 splits"
echo "## Step 8 — score fusion × 4 splits (text v1 + visproto)" >> "$SUMMARY"
for SPLIT in main_3 pseudo_2 hard_4 full_5; do
    TEXT_PREDS="${WORK_DIR}/eval_novel_${SPLIT}/preds_v1text.bbox.json"
    VIS_PREDS="${WORK_DIR}/eval_novel_${SPLIT}_visproto/preds_visproto.bbox.json"
    GT_ANN="${DATA_ROOT_TEST}${SPLIT_ANN[$SPLIT]}"
    OUT_MERGED="${WORK_DIR}/eval_novel_${SPLIT}_scorefuse/preds_scorefuse.bbox.json"
    if [[ ! -f "$TEXT_PREDS" ]] || [[ ! -f "$VIS_PREDS" ]]; then
        echo "  [$SPLIT-scorefuse] MISSING preds (text=$TEXT_PREDS vis=$VIS_PREDS), skip" | tee -a "$SUMMARY"
        continue
    fi
    mkdir -p "$(dirname "$OUT_MERGED")"
    echo "  [$SPLIT-scorefuse] → $OUT_MERGED" | tee -a "$SUMMARY"
    PYTHONPATH="$REPO_ROOT" python tools/fuse_novel_predictions.py \
        --text-preds "$TEXT_PREDS" \
        --vis-preds "$VIS_PREDS" \
        --gt-ann "$GT_ANN" \
        --out "$OUT_MERGED" 2>&1 | tee -a "$SUMMARY" \
        | grep -E 'AP @\[ IoU=0.50:0.95.*all.*100' | head -1
done
echo >> "$SUMMARY"

echo "=== DONE $(date) ===" >> "$SUMMARY"
echo "[done] $(date) → $SUMMARY"
