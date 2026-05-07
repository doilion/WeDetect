#!/usr/bin/env bash
set -euo pipefail

cd /home/25_liwenjie/code/WeDetect
source /home/25_liwenjie/anaconda3/etc/profile.d/conda.sh
conda activate wedetect

export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/matplotlib-wedetect
mkdir -p "$MPLCONFIGDIR"

FULL_CFG=${FULL_CFG:-config/wedetect_tiny_tct_ngc_dev32_cache640_fullnames_2gpu.py}
FULL_WORK=${FULL_WORK:-work_dirs/wedetect_tiny_tct_ngc_dev32_cache640_fullnames_2gpu}
FULL_ANALYSIS="$FULL_WORK/analysis"

RANDOM_CFG=${RANDOM_CFG:-config/wedetect_tiny_tct_ngc_dev32_cache640_fullnames_randomemb_2gpu.py}
RANDOM_WORK=${RANDOM_WORK:-work_dirs/wedetect_tiny_tct_ngc_dev32_cache640_fullnames_randomemb_2gpu}
RANDOM_ANALYSIS="$RANDOM_WORK/analysis"
POST_GPU=${POST_GPU:-0}
RANDOM_GPUS=${RANDOM_GPUS:-2}
RANDOM_CUDA_VISIBLE_DEVICES=${RANDOM_CUDA_VISIBLE_DEVICES:-0,1}
RANDOM_PORT=${RANDOM_PORT:-29645}

mkdir -p "$FULL_ANALYSIS" "$RANDOM_WORK" "$RANDOM_ANALYSIS"

if [[ ! -f "$FULL_WORK/epoch_12.pth" ]]; then
  echo "[postprocess] fullnames training is not complete: missing $FULL_WORK/epoch_12.pth"
  echo "[postprocess] aborting postprocess and random embedding launch"
  exit 2
fi

echo "[postprocess] waiting for fullnames checkpoint files"
ls -lh "$FULL_WORK"/epoch_*.pth

BEST_CKPT="$(ls -t "$FULL_WORK"/best_coco_bbox_mAP_epoch_*.pth 2>/dev/null | head -n 1 || true)"
if [[ -z "$BEST_CKPT" ]]; then
  BEST_CKPT="$(ls -t "$FULL_WORK"/epoch_*.pth | head -n 1)"
fi
echo "[postprocess] best checkpoint: $BEST_CKPT"

echo "[postprocess] plotting train curves from log"
python tools/plot_ngc_training_curves.py \
  --log "$FULL_WORK/train_tmux.log" \
  --out-dir "$FULL_ANALYSIS"

echo "[postprocess] computing validation loss curve from available epoch checkpoints"
CUDA_VISIBLE_DEVICES="$POST_GPU" python tools/compute_ngc_val_loss.py \
  --config "$FULL_CFG" \
  --checkpoint-glob "$FULL_WORK/epoch_*.pth" \
  --out "$FULL_ANALYSIS/val_loss_by_epoch.csv" \
  --batch-size 16 \
  --num-workers 4

echo "[postprocess] plotting combined train and validation loss curves"
python tools/plot_ngc_training_curves.py \
  --log "$FULL_WORK/train_tmux.log" \
  --out-dir "$FULL_ANALYSIS" \
  --val-loss-csv "$FULL_ANALYSIS/val_loss_by_epoch.csv"

echo "[postprocess] generating prediction versus GT panels"
CUDA_VISIBLE_DEVICES="$POST_GPU" python tools/visualize_ngc_predictions.py \
  --config "$FULL_CFG" \
  --checkpoint "$BEST_CKPT" \
  --out-dir "$FULL_ANALYSIS/pred_gt_by_class" \
  --samples-per-class 2 \
  --score-thr 0.20 \
  --device cuda:0

echo "[postprocess] running filtered evaluation for fullnames best checkpoint"
CUDA_VISIBLE_DEVICES="$POST_GPU" python tools/evaluate_ngc_filtered.py \
  --config "$FULL_CFG" \
  --checkpoint "$BEST_CKPT" \
  --work-dir "$FULL_ANALYSIS/filtered_eval" \
  --min-annotations 100 \
  --exclude-keywords negative normal nilm impurity

echo "[randomemb] launching random embedding experiment on CUDA_VISIBLE_DEVICES=$RANDOM_CUDA_VISIBLE_DEVICES with $RANDOM_GPUS process(es)"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$RANDOM_CUDA_VISIBLE_DEVICES"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NCCL_DEBUG=WARN
if [[ "$RANDOM_GPUS" == "1" ]]; then
  python train.py "$RANDOM_CFG" --amp 2>&1 | tee "$RANDOM_WORK/train_tmux.log"
else
  PORT="$RANDOM_PORT" bash dist_train.sh "$RANDOM_CFG" "$RANDOM_GPUS" --amp 2>&1 | tee "$RANDOM_WORK/train_tmux.log"
fi

RANDOM_BEST_CKPT="$(ls -t "$RANDOM_WORK"/best_coco_bbox_mAP_epoch_*.pth 2>/dev/null | head -n 1 || true)"
if [[ -z "$RANDOM_BEST_CKPT" ]]; then
  RANDOM_BEST_CKPT="$(ls -t "$RANDOM_WORK"/epoch_*.pth | head -n 1)"
fi
echo "[randomemb] best checkpoint: $RANDOM_BEST_CKPT"

echo "[randomemb] plotting train curves from log"
python tools/plot_ngc_training_curves.py \
  --log "$RANDOM_WORK/train_tmux.log" \
  --out-dir "$RANDOM_ANALYSIS"

echo "[randomemb] computing validation loss curve from epoch checkpoints"
CUDA_VISIBLE_DEVICES="$POST_GPU" python tools/compute_ngc_val_loss.py \
  --config "$RANDOM_CFG" \
  --checkpoint-glob "$RANDOM_WORK/epoch_*.pth" \
  --out "$RANDOM_ANALYSIS/val_loss_by_epoch.csv" \
  --batch-size 16 \
  --num-workers 4

echo "[randomemb] plotting combined train and validation loss curves"
python tools/plot_ngc_training_curves.py \
  --log "$RANDOM_WORK/train_tmux.log" \
  --out-dir "$RANDOM_ANALYSIS" \
  --val-loss-csv "$RANDOM_ANALYSIS/val_loss_by_epoch.csv"

echo "[randomemb] running filtered evaluation for random embedding best checkpoint"
CUDA_VISIBLE_DEVICES="$POST_GPU" python tools/evaluate_ngc_filtered.py \
  --config "$RANDOM_CFG" \
  --checkpoint "$RANDOM_BEST_CKPT" \
  --work-dir "$RANDOM_ANALYSIS/filtered_eval" \
  --min-annotations 100 \
  --exclude-keywords negative normal nilm impurity
