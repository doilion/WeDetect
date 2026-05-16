#!/usr/bin/env bash
# Eval orchestrator for paper §A ablation table.
#
# For each of the 3 trained experiments:
#   - M1-1PSC (m1)
#   - M1-5attr平均 (m1_5attrmean)
#   - M1+M2-完整方法 (m2)
# runs on:
#   - base 25 test (instances_test_base_clean_dev30.json)
#   - novel 9 merged test (instances_test_novel_merged_9.json)
# using tools/eval_organ_restricted.py with OrganRestrictedCocoMetric.
#
# Uses corrected-protocol best ckpt where the saved best_*.pth doesn't match
# (i.e., M1-5attr平均 ep11 instead of saved ep10; M1+M2 ep10 instead of saved ep11).
#
# Output: work_dirs/<run>/paper_eval/{base25,novel9}.{log,results.json}
# Summary: work_dirs/paper_table_summary.txt
#
# Single-GPU sequential; takes ~3-5 min × 2 splits × 3 rows ≈ 25 min total.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PY="${PY:-$HOME/anaconda3/envs/wedetect/bin/python}"
GPU="${1:-3}"
SUMMARY="${REPO_ROOT}/work_dirs/paper_table_summary_$(date +%Y%m%d_%H%M%S).txt"
echo "=== Paper §A table eval $(date) ===" | tee "$SUMMARY"
echo "GPU: $GPU" | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

DATA_ROOT="/home1/liwenjie/TCT_NGC/"
NEG_EXCLUDE='respiratory tract-Impurity,Serous effusion-Negative samples,Thyroid gland-Negative samples,Urine-NHGUC,TCT_CCD-normal'

# Row tag -> (config, corrected-protocol best ckpt epoch, novel text emb [for non-M2], extra_args for M2)
declare -A ROWS=(
    [m1]="config/wedetect_tiny_tct_ngc_dev30_ochmta_m1_biomedclip_2gpu.py epoch_12"
    [m1_5attrmean]="config/wedetect_tiny_tct_ngc_dev30_ochmta_m1_5attrmean_biomedclip_2gpu.py epoch_11"
    [m2]="config/wedetect_tiny_tct_ngc_dev30_ochmta_m2_biomedclip_2gpu.py epoch_10"
)

# Per-row novel-9 specifics (text emb for 1-PSC / 5-attr; metadata for M2)
declare -A NOVEL_ARGS=(
    [m1]="--text-json data/texts/tct_ngc_novel_merged_9.json --text-emb data/texts/tct_ngc_novel_merged_9_emb_biomedclip.pth"
    [m1_5attrmean]="--text-json data/texts/tct_ngc_novel_merged_9.json --text-emb data/texts/tct_ngc_novel_merged_9_attrmean_biomedclip.pth"
    [m2]="--class-metadata data/texts/tct_ngc_class_metadata_novel_merged.pt"
)

for ROW in m1 m1_5attrmean m2; do
    read CONFIG EPOCH <<< "${ROWS[$ROW]}"
    WORK_DIR="work_dirs/wedetect_tiny_tct_ngc_dev30_ochmta_${ROW}_biomedclip_2gpu"
    CKPT="${WORK_DIR}/${EPOCH}.pth"
    if [[ ! -f "$CKPT" ]]; then
        echo "[skip] $ROW: ckpt missing: $CKPT" | tee -a "$SUMMARY"
        continue
    fi
    EVAL_BASE="${WORK_DIR}/paper_eval"
    mkdir -p "$EVAL_BASE"

    echo "=== ROW: $ROW   ckpt: $EPOCH   ===" | tee -a "$SUMMARY"

    # --- (1) base 25 test ---
    BASE_LOG="${EVAL_BASE}/base25_${EPOCH}.log"
    BASE_WD="${EVAL_BASE}/base25_${EPOCH}_workdir"
    if [[ -f "${BASE_LOG}.done" ]]; then
        echo "[skip-base] already done" | tee -a "$SUMMARY"
    else
        echo "[base25] running $ROW ${EPOCH} on GPU $GPU"
        CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. \
            ~/anaconda3/envs/wedetect/bin/python tools/eval_organ_restricted.py \
                --config "$CONFIG" \
                --checkpoint "$CKPT" \
                --data-root "$DATA_ROOT" \
                --ann-file annotations/instances_test_base_clean_dev30.json \
                --mask-file data/texts/tct_ngc_class_organ_mask_base30.pt \
                --exclude-class-names "$NEG_EXCLUDE" \
                --work-dir "$BASE_WD" \
                > "$BASE_LOG" 2>&1 || { echo "[base25 FAILED] $ROW, see $BASE_LOG"; continue; }
        touch "${BASE_LOG}.done"
    fi
    # Extract metrics
    BASE_METRICS=$(grep -E "coco/overall|coco/all_class/mAP" "$BASE_LOG" 2>/dev/null | tail -1 | grep -oE "(macro_mAP|all_class/mAP|instance_weighted_mAP): [0-9.]+" | tr '\n' ' ')
    echo "  base25: $BASE_METRICS" | tee -a "$SUMMARY"

    # Auto-update master experiment table (docs/experiment_results.{csv,md}).
    # Non-blocking: failure here doesn't abort the eval run.
    PYTHONPATH=. "$PY" tools/experiment_table.py ingest \
        --eval-workdir "$BASE_WD" \
        --config       "$CONFIG"  \
        --ckpt         "$EPOCH"   \
        --eval-tag     paper_eval \
        --split        base25     \
        --notes        "row=$ROW" \
        --no-regen 2>&1 | tee -a "$SUMMARY" || true

    # --- (2) novel 9 merged test ---
    NOVEL_LOG="${EVAL_BASE}/novel9_${EPOCH}.log"
    NOVEL_WD="${EVAL_BASE}/novel9_${EPOCH}_workdir"
    if [[ -f "${NOVEL_LOG}.done" ]]; then
        echo "[skip-novel9] already done" | tee -a "$SUMMARY"
    else
        echo "[novel9] running $ROW ${EPOCH} on GPU $GPU"
        CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. \
            ~/anaconda3/envs/wedetect/bin/python tools/eval_organ_restricted.py \
                --config "$CONFIG" \
                --checkpoint "$CKPT" \
                --data-root "$DATA_ROOT" \
                --ann-file annotations/instances_test_novel_merged_9.json \
                --mask-file data/texts/tct_ngc_class_organ_mask_novel_merged.pt \
                ${NOVEL_ARGS[$ROW]} \
                --work-dir "$NOVEL_WD" \
                > "$NOVEL_LOG" 2>&1 || { echo "[novel9 FAILED] $ROW, see $NOVEL_LOG"; continue; }
        touch "${NOVEL_LOG}.done"
    fi
    NOVEL_METRICS=$(grep -E "coco/overall|coco/all_class/mAP" "$NOVEL_LOG" 2>/dev/null | tail -1 | grep -oE "(macro_mAP|all_class/mAP|instance_weighted_mAP): [0-9.]+" | tr '\n' ' ')
    echo "  novel9: $NOVEL_METRICS" | tee -a "$SUMMARY"

    PYTHONPATH=. "$PY" tools/experiment_table.py ingest \
        --eval-workdir "$NOVEL_WD" \
        --config       "$CONFIG"   \
        --ckpt         "$EPOCH"    \
        --eval-tag     paper_eval  \
        --split        novel9      \
        --notes        "row=$ROW"  \
        --no-regen 2>&1 | tee -a "$SUMMARY" || true

    echo "" | tee -a "$SUMMARY"
done

# Regenerate the markdown view once at the end (cheaper than per-row).
PYTHONPATH=. "$PY" tools/experiment_table.py regen 2>&1 | tee -a "$SUMMARY" || true

echo "=== Done at $(date) ===" | tee -a "$SUMMARY"
echo "Summary: $SUMMARY"
