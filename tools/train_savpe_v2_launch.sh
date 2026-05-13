#!/usr/bin/env bash
# Launcher for Phase 5e SAVPE-v2 training.
#
# Enforces CUDA_VISIBLE_DEVICES=0,1 (the yoloe Claude session owns GPU 2+3).
#
# Modes:
#   bash tools/train_savpe_v2_launch.sh sanity  # single-GPU --sanity-only on GPU 0
#   bash tools/train_savpe_v2_launch.sh train   # 2-GPU DDP, 3 epochs, ~1h
#   bash tools/train_savpe_v2_launch.sh train_low_lambda  # λ_align=0.3 fallback
#
# Re-launch:
#   nohup bash tools/train_savpe_v2_launch.sh train > /tmp/savpe_v2_train.log 2>&1 &

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ── HARD-CODED GPU lease: only 0+1; 2+3 owned by yoloe session ───────────
export CUDA_VISIBLE_DEVICES=0,1
echo "[launch] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

source /home/25_liwenjie/anaconda3/etc/profile.d/conda.sh
conda activate wedetect

MODE="${1:-sanity}"

BASE_CONFIG="config/wedetect_tiny_tct_ngc_dev30_biomedclip_noTHAF_2gpu.py"
BASE_CKPT="work_dirs/wedetect_tiny_tct_ngc_dev30_biomedclip_noTHAF_2gpu/best_coco_bbox_mAP_epoch_11.pth"
TRAIN_ANN="/home1/liwenjie/TCT_NGC_640/annotations/instances_train_dev_disjoint_dev30.json"
DATA_ROOT="/home1/liwenjie/TCT_NGC_640/"
FULLNAMES="data/texts/tct_ngc_fullnames_30.json"
TEXT_CACHE="data/texts/tct_ngc_fullnames_30_embeddings_biomedclip.pth"

OUT_DIR="${OUT_DIR:-work_dirs/savpe_v2_aligned_v1}"

case "$MODE" in
  sanity)
    echo "[launch] sanity (single GPU 0)"
    PYTHONPATH="$REPO_ROOT" python tools/train_savpe_v2_aligned.py \
      --sanity-only \
      --base-config "$BASE_CONFIG" \
      --base-ckpt "$BASE_CKPT" \
      --train-ann "$TRAIN_ANN" \
      --data-root "$DATA_ROOT" \
      --fullnames-json "$FULLNAMES" \
      --text-cache "$TEXT_CACHE" \
      --batch 4 --workers 2 \
      --device cuda:0 \
      --out /tmp/savpe_v2_sanity
    ;;

  train)
    LAMBDA_ALIGN="${LAMBDA_ALIGN:-1.0}"
    LAMBDA_CROSS="${LAMBDA_CROSS:-0.1}"
    BATCH="${BATCH:-64}"
    EPOCHS="${EPOCHS:-3}"
    LR="${LR:-4e-3}"
    echo "[launch] DDP train: GPU 0+1, λ_align=$LAMBDA_ALIGN λ_cross=$LAMBDA_CROSS batch=$BATCH epochs=$EPOCHS lr=$LR"
    echo "[launch] OUT_DIR=$OUT_DIR"
    torchrun --nproc_per_node=2 --master_port=29501 tools/train_savpe_v2_aligned.py \
      --base-config "$BASE_CONFIG" \
      --base-ckpt "$BASE_CKPT" \
      --train-ann "$TRAIN_ANN" \
      --data-root "$DATA_ROOT" \
      --fullnames-json "$FULLNAMES" \
      --text-cache "$TEXT_CACHE" \
      --lambda-align "$LAMBDA_ALIGN" \
      --lambda-cross "$LAMBDA_CROSS" \
      --epochs "$EPOCHS" \
      --batch "$BATCH" \
      --workers 8 \
      --lr "$LR" \
      --out "$OUT_DIR"
    ;;

  train_low_lambda)
    LAMBDA_ALIGN=0.3 OUT_DIR=work_dirs/savpe_v2_aligned_lambda03 \
      bash "$0" train
    ;;

  *)
    echo "Usage: $0 {sanity|train|train_low_lambda}"
    exit 1
    ;;
esac

echo "[launch] done MODE=$MODE"
