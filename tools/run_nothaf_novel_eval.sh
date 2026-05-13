#!/usr/bin/env bash
# noTHAF BiomedCLIP × 4 novel splits eval orchestrator.
#
# Runs eval_novel_split.py for each of {main_3, pseudo_2, hard_4, full_5}
# using the noTHAF biomedclip ckpt + per-split BiomedCLIP novel text embs.
#
# Picks ONE free GPU (mem < 2GB used) so this can run alongside other
# trainings without conflict. If no GPU is free, polls every 60s.
#
# Usage:
#   nohup bash tools/run_nothaf_novel_eval.sh > /tmp/nothaf_novel_eval.log 2>&1 &
#
# Output: results tee'd to noTHAF work_dir/novel_eval_summary.txt

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

DATA_ROOT="/home1/liwenjie/TCT_NGC/"
SUMMARY="${WORK_DIR}/novel_eval_summary.txt"

declare -A SPLIT_ANN=(
    [main_3]="annotations/instances_test_main_novel.json"
    [pseudo_2]="annotations/instances_test_pseudo_novel.json"
    [hard_4]="annotations/instances_hard_test.json"
    [full_5]="annotations/instances_test_novel.json"
)

# Wait for at least one GPU to have < 2000 MiB used (idle). Returns GPU index.
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

echo "[wait] $(date) waiting for a free GPU (mem < 2GB)..."
GPU=$(wait_for_free_gpu)
echo "[fire] $(date) using GPU $GPU"

echo "=== noTHAF BiomedCLIP × 4 novel splits eval $(date) ===" > "$SUMMARY"
echo "ckpt: $CKPT" >> "$SUMMARY"
echo "GPU: $GPU" >> "$SUMMARY"
echo >> "$SUMMARY"

for SPLIT in main_3 pseudo_2 hard_4 full_5; do
    EVAL_DIR="${WORK_DIR}/eval_novel_${SPLIT}"
    echo "" >> "$SUMMARY"
    echo "## Novel split: $SPLIT  $(date)" | tee -a "$SUMMARY"

    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$REPO_ROOT" python tools/eval_novel_split.py \
        --config "$CONFIG" \
        --checkpoint "$CKPT" \
        --data-root "$DATA_ROOT" \
        --ann-file "${SPLIT_ANN[$SPLIT]}" \
        --text-json "data/texts/tct_ngc_novel_${SPLIT}.json" \
        --text-emb "data/texts/tct_ngc_novel_${SPLIT}_emb_biomedclip.pth" \
        --outfile-prefix "${EVAL_DIR}/preds_v1text" \
        --work-dir "$EVAL_DIR" 2>&1 | tee -a "$SUMMARY" | grep -E 'bbox_mAP:' | tail -1
done

echo "" >> "$SUMMARY"
echo "=== DONE $(date) ===" >> "$SUMMARY"
echo "[done] $(date) → $SUMMARY"
