#!/usr/bin/env bash
# Full baseline novel zero-shot re-eval suite for a clean dev30 ckpt.
#
# Reproduces the pre-existing landscape on a fresh checkpoint:
#   1. test_base 25-cls (excl. 5 dev30 negatives)
#   2. v2 PSC/MAL-S/Bethesda text baseline × 4 novel splits  [--outfile-prefix]
#   3. base-30 visual prototype from training set (no test leakage)
#   4. novel visual prototype × 4 splits (5-shot from test set, with leakage)
#   5. visproto-only eval × 4 splits                          [--outfile-prefix]
#   6. Procrustes R refit (base text ↔ base visproto)
#   7. Procrustes-calfused emb × 4 splits + eval (DEAD-5 verify on new ckpt)
#   8. Score fusion × 4 splits (merge v2-text preds + visproto preds per class)
#
# All 4 strategy-eval results are tee'd to a single summary file for ablation.
#
# Usage:
#   bash tools/eval_baseline_all.sh [<CKPT>]
#
# If CKPT is omitted, picks `best_coco_bbox_mAP_epoch_*.pth` from the clean
# dev30 work_dir. Outputs go under data/texts/clean_dev30_*.pth (separate
# from the old throttled-GPU-1 ckpt's caches) so old numbers stay reproducible.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="config/wedetect_tiny_tct_ngc_dev30_cache640_fullnames_disjoint_clean_2gpu.py"
WORK_DIR="work_dirs/wedetect_tiny_tct_ngc_dev30_cache640_fullnames_disjoint_clean_2gpu"

# Safety guard 1: refuse to run while training process is still active —
# best_*.pth keeps updating mid-training, and stealing a training GPU for
# eval can OOM the training run. Match by config-name substring; will catch
# both `dist_train.sh ... disjoint_clean_2gpu.py` and the torchrun children.
ACTIVE_PIDS=$(pgrep -f 'disjoint_clean_2gpu' | grep -v "^$$\$" || true)
if [[ -n "$ACTIVE_PIDS" ]]; then
    echo "ERROR: clean dev30 training is still running. Wait for it to finish."
    echo "  (active PIDs: $(echo $ACTIVE_PIDS | tr '\n' ' '))"
    exit 1
fi

# Safety guard 2: require ep12 ckpt (final epoch) to confirm training is
# complete. Training writes best_*.pth incrementally so its presence alone
# isn't sufficient.
EP12_CKPT="${WORK_DIR}/epoch_12.pth"
if [[ ! -f "$EP12_CKPT" ]]; then
    echo "ERROR: training not complete; no $EP12_CKPT found."
    echo "  ckpts present: $(ls "${WORK_DIR}"/epoch_*.pth 2>/dev/null | wc -l) of 12 expected"
    exit 1
fi

CKPT="${1:-}"
if [[ -z "$CKPT" ]]; then
    CKPT=$(ls -t "${WORK_DIR}"/best_coco_bbox_mAP_epoch_*.pth 2>/dev/null | head -1 || true)
fi
if [[ -z "$CKPT" ]] || [[ ! -f "$CKPT" ]]; then
    echo "ERROR: no clean dev30 ckpt at $WORK_DIR (looking for best_coco_bbox_mAP_epoch_*.pth)"
    exit 1
fi
echo "[ckpt] $CKPT"

# Output filenames are tagged _clean to not overwrite old throttled-ckpt caches
TAG="clean"
TEXT_DIR="data/texts"
SUMMARY="${WORK_DIR}/baseline_eval_summary.txt"

DATA_ROOT_640="/home1/liwenjie/TCT_NGC_640/"
DATA_ROOT_TEST="/home1/liwenjie/TCT_NGC/"
TRAIN_ANN="annotations/instances_train_dev_disjoint_dev30.json"
BASE_TEST_ANN="annotations/instances_test_base_clean_dev30.json"
DEV30_NEG_EXCLUDE='respiratory tract-Impurity,Serous effusion-Negative samples,Thyroid gland-Negative samples,Urine-NHGUC,TCT_CCD-normal'

declare -A SPLIT_ANN=(
    [main_3]="annotations/instances_test_main_novel.json"
    [pseudo_2]="annotations/instances_test_pseudo_novel.json"
    [hard_4]="annotations/instances_hard_test.json"
    [full_5]="annotations/instances_test_novel.json"
)

V2_TEXT_BASE="${TEXT_DIR}/tct_ngc_fullnames_30.json"

echo "=== clean dev30 baseline re-eval $(date) ===" > "$SUMMARY"
echo "ckpt: $CKPT" >> "$SUMMARY"
echo >> "$SUMMARY"

# ────────────────────────────────────────────────────────────────────
# Step 1: test_base 25-cls
# ────────────────────────────────────────────────────────────────────
echo
echo "[1/8] test_base 25-cls (excluding 5 dev30 negatives)"
echo "## Step 1 — test_base 25-cls" >> "$SUMMARY"
PYTHONPATH="$REPO_ROOT" python test_exclude_negative.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --data-root "$DATA_ROOT_TEST" \
    --ann-file "$BASE_TEST_ANN" \
    --exclude-class-names "$DEV30_NEG_EXCLUDE" \
    --work-dir "${WORK_DIR}/eval_base_25cls_${TAG}" 2>&1 | tee -a "$SUMMARY" | grep -E 'bbox_mAP_copypaste|coco/bbox_mAP:' | tail -2
echo >> "$SUMMARY"

# ────────────────────────────────────────────────────────────────────
# Step 2: v2 text baseline × 4 splits
# ────────────────────────────────────────────────────────────────────
echo
echo "[2/8] v2 text baseline × 4 splits  (--outfile-prefix saved for score fusion)"
echo "## Step 2 — v2 text baseline novel × 4 splits" >> "$SUMMARY"
for SPLIT in main_3 pseudo_2 hard_4 full_5; do
    EVAL_DIR="${WORK_DIR}/eval_novel_${SPLIT}_v2text_${TAG}"
    echo "  [$SPLIT-v2text]"
    PYTHONPATH="$REPO_ROOT" python tools/eval_novel_split.py \
        --config "$CONFIG" \
        --checkpoint "$CKPT" \
        --data-root "$DATA_ROOT_TEST" \
        --ann-file "${SPLIT_ANN[$SPLIT]}" \
        --text-json "${TEXT_DIR}/tct_ngc_novel_${SPLIT}.json" \
        --text-emb "${TEXT_DIR}/tct_ngc_novel_${SPLIT}_emb.pth" \
        --outfile-prefix "${EVAL_DIR}/preds_v2text" \
        --work-dir "$EVAL_DIR" 2>&1 | tee -a "$SUMMARY" | grep -E 'bbox_mAP:' | tail -1
done
echo >> "$SUMMARY"

# ────────────────────────────────────────────────────────────────────
# Step 3: base-30 visproto from training set
# ────────────────────────────────────────────────────────────────────
BASE_VISPROTO="${TEXT_DIR}/tct_ngc_base30_visproto_train_${TAG}.pth"
echo
echo "[3/8] build base-30 visproto from training set → ${BASE_VISPROTO}"
echo "## Step 3 — base-30 visproto from train" >> "$SUMMARY"
PYTHONPATH="$REPO_ROOT" python tools/build_visual_prototype.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --ann-file "$TRAIN_ANN" \
    --data-root "$DATA_ROOT_640" \
    --img-prefix "images/" \
    --text-json "$V2_TEXT_BASE" \
    --out "$BASE_VISPROTO" \
    --n-per-class 5 2>&1 | tee -a "$SUMMARY" | tail -3
echo >> "$SUMMARY"

# ────────────────────────────────────────────────────────────────────
# Step 4: novel visproto × 4 splits (5-shot from test, with leakage)
# ────────────────────────────────────────────────────────────────────
echo
echo "[4/8] build novel visproto × 4 splits"
echo "## Step 4 — novel visproto × 4 splits" >> "$SUMMARY"
for SPLIT in main_3 pseudo_2 hard_4 full_5; do
    NOVEL_VISPROTO="${TEXT_DIR}/tct_ngc_novel_${SPLIT}_visproto_emb_${TAG}.pth"
    echo "  [$SPLIT-visproto-build] → $NOVEL_VISPROTO"
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
echo "[5/8] visproto-only eval × 4 splits  (--outfile-prefix saved for score fusion)"
echo "## Step 5 — visproto-only eval × 4 splits" >> "$SUMMARY"
for SPLIT in main_3 pseudo_2 hard_4 full_5; do
    EVAL_DIR="${WORK_DIR}/eval_novel_${SPLIT}_visproto_${TAG}"
    NOVEL_VISPROTO="${TEXT_DIR}/tct_ngc_novel_${SPLIT}_visproto_emb_${TAG}.pth"
    echo "  [$SPLIT-visproto-eval]"
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
# Step 6: Procrustes R refit (base text ↔ base visproto)
# ────────────────────────────────────────────────────────────────────
PROCRUSTES_R="${TEXT_DIR}/procrustes_R_${TAG}.pth"
echo
echo "[6/8] Procrustes R refit → ${PROCRUSTES_R}"
echo "## Step 6 — Procrustes R refit" >> "$SUMMARY"
NOVEL_VIS_LIST=()
for SPLIT in main_3 pseudo_2 hard_4 full_5; do
    NOVEL_VIS_LIST+=( "${TEXT_DIR}/tct_ngc_novel_${SPLIT}_visproto_emb_${TAG}.pth" )
done
PYTHONPATH="$REPO_ROOT" python tools/procrustes_text_visual.py \
    --text-emb "${TEXT_DIR}/tct_ngc_fullnames_30_embeddings_wedetect_tiny.pth" \
    --vis-proto-base "$BASE_VISPROTO" \
    --vis-proto-novel "${NOVEL_VIS_LIST[@]}" \
    --out-dir "$TEXT_DIR" \
    --out-r "$PROCRUSTES_R" 2>&1 | tee -a "$SUMMARY" | tail -5
# Note: procrustes_text_visual.py writes calibrated novel emb files using the
# input filenames' stem replacement: <split>_visproto_emb_clean.pth →
# <split>_visproto_calibrated_emb_clean.pth
echo >> "$SUMMARY"

# ────────────────────────────────────────────────────────────────────
# Step 7: Procrustes calfused × 4 splits + eval (DEAD-5 verify)
# ────────────────────────────────────────────────────────────────────
echo
echo "[7/8] build calfused emb + eval × 4 splits  (DEAD-5 verify on clean ckpt)"
echo "## Step 7 — Procrustes calfused × 4 splits" >> "$SUMMARY"
for SPLIT in main_3 pseudo_2 hard_4 full_5; do
    NOVEL_TEXT="${TEXT_DIR}/tct_ngc_novel_${SPLIT}_emb.pth"
    NOVEL_CAL_VIS="${TEXT_DIR}/tct_ngc_novel_${SPLIT}_visproto_calibrated_emb_${TAG}.pth"
    NOVEL_CALFUSED="${TEXT_DIR}/tct_ngc_novel_${SPLIT}_calfused_emb_${TAG}.pth"
    NOVEL_TEXT_JSON="${TEXT_DIR}/tct_ngc_novel_${SPLIT}.json"

    # Build calfused emb: route each class to text or calibrated visproto by
    # organ keyword (Serous/breast/ovarian → text; Resp/Thyroid → calvis)
    PYTHONPATH="$REPO_ROOT" python -c "
import torch, json, re
text = torch.load('$NOVEL_TEXT', map_location='cpu')
calvis = torch.load('$NOVEL_CAL_VIS', map_location='cpu')
groups = json.loads(open('$NOVEL_TEXT_JSON').read())
text_pat = re.compile(r'(Serous|breast|ovarian)', re.IGNORECASE)
fused = {}
for grp in groups:
    p = grp[0]
    if text_pat.search(p):
        fused[p] = text[p]
    else:
        fused[p] = calvis[p]
torch.save(fused, '$NOVEL_CALFUSED')
print(f'$SPLIT calfused: {len(fused)} classes saved → $NOVEL_CALFUSED')
"

    EVAL_DIR="${WORK_DIR}/eval_novel_${SPLIT}_calfused_${TAG}"
    echo "  [$SPLIT-calfused-eval]"
    PYTHONPATH="$REPO_ROOT" python tools/eval_novel_split.py \
        --config "$CONFIG" \
        --checkpoint "$CKPT" \
        --data-root "$DATA_ROOT_TEST" \
        --ann-file "${SPLIT_ANN[$SPLIT]}" \
        --text-json "$NOVEL_TEXT_JSON" \
        --text-emb "$NOVEL_CALFUSED" \
        --work-dir "$EVAL_DIR" 2>&1 | tee -a "$SUMMARY" | grep -E 'bbox_mAP:' | tail -1
done
echo >> "$SUMMARY"

# ────────────────────────────────────────────────────────────────────
# Step 8: Score fusion × 4 splits
# ────────────────────────────────────────────────────────────────────
echo
echo "[8/8] score fusion × 4 splits  (merge v2 text + visproto preds per class)"
echo "## Step 8 — Score fusion × 4 splits" >> "$SUMMARY"
for SPLIT in main_3 pseudo_2 hard_4 full_5; do
    TEXT_PREDS="${WORK_DIR}/eval_novel_${SPLIT}_v2text_${TAG}/preds_v2text.bbox.json"
    VIS_PREDS="${WORK_DIR}/eval_novel_${SPLIT}_visproto_${TAG}/preds_visproto.bbox.json"
    GT_ANN="${DATA_ROOT_TEST}${SPLIT_ANN[$SPLIT]}"
    OUT_MERGED="${WORK_DIR}/eval_novel_${SPLIT}_scorefuse_${TAG}/preds_scorefuse.bbox.json"

    if [[ ! -f "$TEXT_PREDS" ]] || [[ ! -f "$VIS_PREDS" ]]; then
        echo "  [$SPLIT] SKIP — missing predictions ($TEXT_PREDS or $VIS_PREDS)"
        continue
    fi
    mkdir -p "$(dirname "$OUT_MERGED")"
    echo "  [$SPLIT-scorefuse]"
    PYTHONPATH="$REPO_ROOT" python tools/fuse_novel_predictions.py \
        --text-preds "$TEXT_PREDS" \
        --vis-preds "$VIS_PREDS" \
        --gt-ann "$GT_ANN" \
        --out "$OUT_MERGED" 2>&1 | tee -a "$SUMMARY" | grep -E 'AP @\[ IoU=0.50:0.95.*all.*100' | head -1
done
echo >> "$SUMMARY"

# ────────────────────────────────────────────────────────────────────
echo
echo "=== DONE ==="
echo "summary: $SUMMARY"
echo
echo "Quick mAP scan:"
grep -E 'bbox_mAP:' "$SUMMARY" | tail -20
