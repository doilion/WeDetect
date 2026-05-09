#!/usr/bin/env bash
# Phase 3 orchestration for dev30 disjoint baseline.
# Runs the full eval+viz+clinical pipeline against the BEST checkpoint
# from the dev30 training run. Designed to be fired in tmux right after
# tools/run_ngc_dev30_disjoint_2gpu.sh DONE marker appears.
set -euo pipefail

cd /home/25_liwenjie/code/WeDetect

source /home/25_liwenjie/anaconda3/etc/profile.d/conda.sh
conda activate wedetect

export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export MPLCONFIGDIR=/tmp/matplotlib-wedetect
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

WORK="work_dirs/wedetect_tiny_tct_ngc_dev30_cache640_fullnames_disjoint_2gpu"
CFG="config/wedetect_tiny_tct_ngc_dev30_cache640_fullnames_disjoint_2gpu.py"
ANALYSIS="${WORK}/analysis"
PHASE3_LOG="${WORK}/phase3.log"
mkdir -p "${ANALYSIS}"

BEST=$(ls -t ${WORK}/best_coco_bbox_mAP_epoch_*.pth 2>/dev/null | head -1 || true)
if [[ -z "${BEST}" ]]; then
  BEST=$(ls -t ${WORK}/epoch_*.pth | head -1)
  echo "[warn] no best_*.pth found, falling back to latest: ${BEST}"
fi

TEST_DATA_ROOT="/home1/liwenjie/TCT_NGC/"
TEST_ANN="annotations/instances_test_base_clean_dev30.json"
VAL_ANN_ABS="/home1/liwenjie/TCT_NGC_640/annotations/instances_val_dev_disjoint_dev30.json"
TEST_ANN_ABS="${TEST_DATA_ROOT}${TEST_ANN}"
TRAIN_ANN_ABS="/home1/liwenjie/TCT_NGC_640/annotations/instances_train_dev_disjoint_dev30.json"

NEG30="respiratory tract-Impurity,Serous effusion-Negative samples,Thyroid gland-Negative samples,Urine-NHGUC,TCT_CCD-normal"

WEAK30='Thyroid gland-AUC|Urine-AUC|Urine-SHGUC|TCT_CCD-monilia|TCT_CCD-asch|TCT_CCD-ec|respiratory tract-Lymphocyte|Thyroid gland-NS'
COHORT30='Thyroid gland-NS|Thyroid gland-Macrophages|Urine-HGUC|respiratory tract-Alveolar macrophages'

{
  printf "\n==== NGC_DEV30_PHASE3_START %s ====\n" "$(date "+%F %T")"
  printf "BEST=%s\n" "${BEST}"

  # ---- Step 1: sequential val + test eval (single GPU 2; GPU 0/1/3 unhealthy after Xid 79) ----
  GPU_ID="${PHASE3_GPU:-2}"
  echo "[step1] val eval + test_base eval on GPU ${GPU_ID} (sequential, single-GPU)"

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU_ID} \
    python test_exclude_negative.py \
    --config "${CFG}" --checkpoint "${BEST}" \
    --work-dir "${WORK}/eval_classwise_val" \
    --outfile-prefix "${WORK}/eval_classwise_val/preds" \
    --exclude-class-names "${NEG30}" \
    2>&1 | tee "${ANALYSIS}/eval_val.stdout"
  VAL_RC=${PIPESTATUS[0]}
  echo "[step1a] val rc=${VAL_RC}"
  [[ ${VAL_RC} -eq 0 ]] || { echo "[fatal] val eval failed"; exit 1; }

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU_ID} \
    python test_exclude_negative.py \
    --config "${CFG}" --checkpoint "${BEST}" \
    --data-root "${TEST_DATA_ROOT}" --ann-file "${TEST_ANN}" \
    --work-dir "${WORK}/eval_classwise_test" \
    --outfile-prefix "${WORK}/eval_classwise_test/preds" \
    --exclude-class-names "${NEG30}" \
    2>&1 | tee "${ANALYSIS}/eval_test.stdout"
  TEST_RC=${PIPESTATUS[0]}
  echo "[step1b] test rc=${TEST_RC}"
  [[ ${TEST_RC} -eq 0 ]] || { echo "[fatal] test eval failed"; exit 1; }

  VAL_LOG=$(ls -t ${WORK}/eval_classwise_val/20*/20*.log | head -1)
  TEST_LOG=$(ls -t ${WORK}/eval_classwise_test/20*/20*.log | head -1)
  echo "VAL_LOG=${VAL_LOG}"
  echo "TEST_LOG=${TEST_LOG}"

  # ---- Step 2: 12-epoch val loss curve (GPU 2) ----
  echo "[step2] val loss curve over 12 epochs (GPU ${GPU_ID})"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU_ID} \
    python tools/compute_ngc_val_loss.py \
    --config "${CFG}" \
    --checkpoint-glob "${WORK}/epoch_*.pth" \
    --out "${ANALYSIS}/val_loss_full.csv" \
    2>&1 | tee "${ANALYSIS}/val_loss.stdout"

  # ---- Step 3: per-class CSV ----
  echo "[step3] analyze_ngc_disjoint_results"
  python tools/analyze_ngc_disjoint_results.py \
    --train-ann "${TRAIN_ANN_ABS}" \
    --val-ann   "${VAL_ANN_ABS}" \
    --test-ann  "${TEST_ANN_ABS}" \
    --val-log   "${VAL_LOG}" \
    --test-log  "${TEST_LOG}" \
    --text-emb  data/texts/tct_ngc_fullnames_30_embeddings_wedetect_tiny.pth \
    --text-json data/texts/tct_ngc_fullnames_30.json \
    --out       "${ANALYSIS}/disjoint_results_per_class.csv"

  # ---- Step 4: plots ----
  echo "[step4] D1-D5 plots"
  python tools/plot_classwise_ap.py --log "${VAL_LOG}" \
    --out "${ANALYSIS}/classwise_ap_dev30_val.png" \
    --title "Per-class AP — dev30 val (exclude-neg, 25 classes)"

  python tools/plot_classwise_ap.py --log "${TEST_LOG}" \
    --out "${ANALYSIS}/classwise_ap_dev30_test.png" \
    --title "Per-class AP — dev30 test_base (exclude-neg, 25 classes)"

  python tools/plot_classwise_compare.py \
    --val-log "${VAL_LOG}" --test-log "${TEST_LOG}" \
    --out "${ANALYSIS}/classwise_ap_dev30_val_vs_test.png" \
    --val-label "dev30 val" --test-label "dev30 test_base"

  DEV32_TEST_LOG=$(ls -t work_dirs/wedetect_tiny_tct_ngc_dev32_cache640_fullnames_disjoint_2gpu/eval_test_base_disjoint/20*/20*.log 2>/dev/null | head -1 || true)
  if [[ -n "${DEV32_TEST_LOG}" ]]; then
    python tools/plot_classwise_compare.py \
      --val-log "${DEV32_TEST_LOG}" --test-log "${TEST_LOG}" \
      --out "${ANALYSIS}/classwise_ap_dev32_vs_dev30_test.png" \
      --val-label "dev32 test_base" --test-label "dev30 test_base"
  else
    echo "[warn] no dev32 test_base log found, skipping D4"
  fi

  python tools/plot_ngc_training_curves.py \
    --log "${WORK}/train_tmux.log" \
    --val-loss-csv "${ANALYSIS}/val_loss_full.csv" \
    --out-dir "${ANALYSIS}/"

  # ---- Step 5: viz panels (GPU ${GPU_ID}, sequential) ----
  echo "[step5] viz F1/F2/F3 (GPU ${GPU_ID}, sequential)"

  # F1: clean gallery, all 30 classes, val
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU_ID} python tools/visualize_ngc_predictions.py \
    --config "${CFG}" --checkpoint "${BEST}" \
    --out-dir "${ANALYSIS}/viz_val_clean" \
    --out-label dev30_val \
    --samples-per-class 4 --score-thr 0.2 --device cuda:0 \
    > "${ANALYSIS}/viz_F1.stdout" 2>&1

  # F2: weak-class failure, val, low threshold
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU_ID} python tools/visualize_ngc_predictions.py \
    --config "${CFG}" --checkpoint "${BEST}" \
    --out-dir "${ANALYSIS}/viz_val_failure_lowthr" \
    --out-label dev30_val_failure \
    --class-regex "${WEAK30}" \
    --samples-per-class 6 --score-thr 0.05 --device cuda:0 \
    > "${ANALYSIS}/viz_F2.stdout" 2>&1

  # F3: cohort-reversal on test_base
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU_ID} python tools/visualize_ngc_predictions.py \
    --config "${CFG}" --checkpoint "${BEST}" \
    --data-root "${TEST_DATA_ROOT}" --ann-file "${TEST_ANN}" \
    --out-dir "${ANALYSIS}/viz_testbase_cohort_reversal" \
    --out-label dev30_test_cohort_reversal \
    --class-regex "${COHORT30}" \
    --samples-per-class 4 --score-thr 0.2 --device cuda:0 \
    > "${ANALYSIS}/viz_F3.stdout" 2>&1

  # ---- Step 6: clinical metrics ----
  echo "[step6] clinical metrics M1-M4"
  python tools/eval_clinical_metrics.py \
    --preds "${WORK}/eval_classwise_val/preds.bbox.json" \
    --ann   "${VAL_ANN_ABS}" \
    --cost-config tools/clinical_cost_config_dev30.json \
    --out   "${WORK}/clinical_metrics_val"

  python tools/eval_clinical_metrics.py \
    --preds "${WORK}/eval_classwise_test/preds.bbox.json" \
    --ann   "${TEST_ANN_ABS}" \
    --cost-config tools/clinical_cost_config_dev30.json \
    --out   "${WORK}/clinical_metrics_test"

  printf "==== NGC_DEV30_PHASE3_DONE %s ====\n" "$(date "+%F %T")"
} 2>&1 | tee -a "${PHASE3_LOG}"
