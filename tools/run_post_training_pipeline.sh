#!/usr/bin/env bash
# Post-training auto-orchestrator for the THAF + clean dev30 ablation pipeline.
#
# Sequence:
#   1. Wait for THAF XLM-R training (epoch_12.pth + no active process)
#   2. Run eval_thaf_all_splits.sh xlmr (GPU 2) + biomedclip (GPU 3) in parallel
#   3. Wait for clean dev30 retrain (epoch_12.pth + no active process)
#   4. Run eval_baseline_all.sh (GPU 0)
#   5. Compile ablation table via compile_ablation_table.py
#
# Run detached:
#   nohup bash tools/run_post_training_pipeline.sh > work_dirs/post_training_pipeline.log 2>&1 &
#
# All output is also tee'd to work_dirs/post_training_pipeline.log automatically.

set -uo pipefail   # NO -e: don't abort if one eval fails — try the rest

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

LOG="${REPO_ROOT}/work_dirs/post_training_pipeline.log"
mkdir -p "$(dirname "$LOG")"
touch "$LOG"

# All stdout/stderr → log + console
exec > >(tee -a "$LOG") 2>&1

THAF_XLMR_DIR="work_dirs/wedetect_tiny_tct_ngc_dev30_thaf_xlmr_2gpu"
THAF_BIO_DIR="work_dirs/wedetect_tiny_tct_ngc_dev30_thaf_biomedclip_2gpu"
CLEAN_DIR="work_dirs/wedetect_tiny_tct_ngc_dev30_cache640_fullnames_disjoint_clean_2gpu"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "==================================================================="
echo "[$(ts)] orchestrator started (pid $$)"
echo "  - log: $LOG"
echo "  - THAF XLM-R dir:    $THAF_XLMR_DIR"
echo "  - THAF BiomedCLIP:   $THAF_BIO_DIR (already trained, just re-eval)"
echo "  - clean dev30:       $CLEAN_DIR"
echo "==================================================================="

# Wait until BOTH (a) ep12 ckpt exists AND (b) no training process is active
# for the given pgrep pattern. Polls every 2 minutes.
wait_for_training() {
    local DIR="$1"
    local NAME="$2"
    local PROC_PATTERN="$3"
    local EP12="${DIR}/epoch_12.pth"

    echo "[$(ts)] waiting for $NAME ..."
    while true; do
        local CKPT_OK="no"
        local PROC_OK="no"
        [[ -f "$EP12" ]] && CKPT_OK="yes"
        if ! pgrep -f "$PROC_PATTERN" > /dev/null 2>&1; then
            PROC_OK="yes"
        fi
        if [[ "$CKPT_OK" == "yes" && "$PROC_OK" == "yes" ]]; then
            echo "[$(ts)] $NAME ready (ep12.pth ✓ no active process ✓)"
            return 0
        fi
        # Lightweight progress hint every cycle
        local LATEST_LOG
        LATEST_LOG=$(find "$DIR" -name '*.log' -printf '%T@ %p\n' 2>/dev/null \
                     | sort -nr | head -1 | awk '{print $2}')
        if [[ -n "${LATEST_LOG:-}" ]]; then
            local LATEST_LINE
            LATEST_LINE=$(grep -oE 'Epoch\(train\) +\[[0-9]+\]\[[0-9]+/3238\][^a]*eta: [0-9:]+' \
                          "$LATEST_LOG" 2>/dev/null | tail -1 | head -c 120)
            [[ -n "$LATEST_LINE" ]] && echo "[$(ts)]   ($NAME progress) $LATEST_LINE"
        fi
        sleep 120
    done
}

# Run a sub-script with bounded output (avoid swamping the orchestrator log)
run_eval() {
    local LABEL="$1"
    local OUT_FILE="$2"
    shift 2
    echo "[$(ts)] >>> $LABEL"
    "$@" > "$OUT_FILE" 2>&1
    local RC=$?
    if [[ $RC -eq 0 ]]; then
        echo "[$(ts)] <<< $LABEL OK (full log: $OUT_FILE)"
        # Surface key mAP lines
        grep -E 'bbox_mAP:|bbox_mAP_copypaste|AP @\[ IoU=0.50:0.95.*all.*100' "$OUT_FILE" \
            | tail -10 | sed 's/^/    /'
    else
        echo "[$(ts)] <<< ⚠ $LABEL FAILED (rc=$RC, see $OUT_FILE)"
    fi
    return $RC
}

# ────────────────────────────────────────────────────────────────────
# STAGE 1: wait for THAF XLM-R
# ────────────────────────────────────────────────────────────────────
echo
echo "[$(ts)] === STAGE 1 — wait for THAF XLM-R training ==="
wait_for_training "$THAF_XLMR_DIR" "THAF XLM-R" "thaf_xlmr_2gpu"

# ────────────────────────────────────────────────────────────────────
# STAGE 2: parallel THAF eval (XLM-R on GPU 2, BiomedCLIP on GPU 3)
# ────────────────────────────────────────────────────────────────────
echo
echo "[$(ts)] === STAGE 2 — parallel THAF eval on freed GPU 2+3 ==="

CUDA_VISIBLE_DEVICES=2 bash tools/eval_thaf_all_splits.sh xlmr \
    > "${THAF_XLMR_DIR}/eval_orchestrator.log" 2>&1 &
PID_XLMR=$!
echo "[$(ts)] launched THAF XLM-R eval on GPU 2 (pid $PID_XLMR)"

CUDA_VISIBLE_DEVICES=3 bash tools/eval_thaf_all_splits.sh biomedclip \
    > "${THAF_BIO_DIR}/eval_orchestrator.log" 2>&1 &
PID_BIO=$!
echo "[$(ts)] launched BiomedCLIP THAF eval on GPU 3 (pid $PID_BIO)"

wait $PID_XLMR; RC_XLMR=$?
wait $PID_BIO;  RC_BIO=$?

if [[ $RC_XLMR -eq 0 ]]; then
    echo "[$(ts)] THAF XLM-R eval OK"
    grep -E 'bbox_mAP:|bbox_mAP_copypaste' "${THAF_XLMR_DIR}/eval_orchestrator.log" \
        | tail -8 | sed 's/^/    /'
else
    echo "[$(ts)] ⚠ THAF XLM-R eval failed (rc=$RC_XLMR)"
fi
if [[ $RC_BIO -eq 0 ]]; then
    echo "[$(ts)] BiomedCLIP THAF eval OK"
    grep -E 'bbox_mAP:|bbox_mAP_copypaste' "${THAF_BIO_DIR}/eval_orchestrator.log" \
        | tail -8 | sed 's/^/    /'
else
    echo "[$(ts)] ⚠ BiomedCLIP THAF eval failed (rc=$RC_BIO)"
fi

# ────────────────────────────────────────────────────────────────────
# STAGE 3: wait for clean dev30
# ────────────────────────────────────────────────────────────────────
echo
echo "[$(ts)] === STAGE 3 — wait for clean dev30 retrain ==="
wait_for_training "$CLEAN_DIR" "clean dev30" "disjoint_clean_2gpu"

# ────────────────────────────────────────────────────────────────────
# STAGE 4: baseline full eval suite
# ────────────────────────────────────────────────────────────────────
echo
echo "[$(ts)] === STAGE 4 — baseline full eval (8 steps × 4 splits) ==="
run_eval "baseline full eval" "${CLEAN_DIR}/eval_orchestrator.log" \
    env CUDA_VISIBLE_DEVICES=0 bash tools/eval_baseline_all.sh

# ────────────────────────────────────────────────────────────────────
# STAGE 5: compile ablation table
# ────────────────────────────────────────────────────────────────────
echo
echo "[$(ts)] === STAGE 5 — compile ablation table ==="
if [[ -f tools/compile_ablation_table.py ]]; then
    PYTHONPATH="$REPO_ROOT" python tools/compile_ablation_table.py 2>&1 \
        | tee "${REPO_ROOT}/work_dirs/ablation_table.md" || \
        echo "[$(ts)] ⚠ ablation compile failed"
else
    echo "[$(ts)] ⚠ compile_ablation_table.py not found, skipping"
fi

echo
echo "==================================================================="
echo "[$(ts)] orchestrator DONE — full pipeline complete"
echo "  ablation table: ${REPO_ROOT}/work_dirs/ablation_table.md"
echo "  full log:       $LOG"
echo "==================================================================="
