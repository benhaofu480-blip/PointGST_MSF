#!/usr/bin/env bash
# Official PCN test (1200 test): single_MSF_trial seed42, ckpt-best + ckpt-last.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
PYTHON="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python"

EXP_DIR="${ROOT}/experiments/AdaPoinTr_MSF_single_MSF_trial/PCN_models/exp_single_MSF_trial_seed42"
CFG="cfgs/PCN_models/AdaPoinTr_MSF_single_MSF_trial.yaml"
LOG_DIR="${ROOT}/logs/complete"
GPU_BEST="${GPU_BEST:-0}"
GPU_LAST="${GPU_LAST:-1}"

CKPT_BEST="${EXP_DIR}/ckpt-best.pth"
CKPT_LAST="${EXP_DIR}/ckpt-last.pth"
LOG_BEST="${LOG_DIR}/test_single_MSF_trial_seed42_best.log"
LOG_LAST="${LOG_DIR}/test_single_MSF_trial_seed42_last.log"
TAG_BEST="single_MSF_trial_seed42_best"
TAG_LAST="single_MSF_trial_seed42_last"

[[ -f "$CKPT_BEST" ]] || { echo "Missing $CKPT_BEST" >&2; exit 1; }
[[ -f "$CKPT_LAST" ]] || { echo "Missing $CKPT_LAST" >&2; exit 1; }

if pgrep -f "main.py --test.*${TAG_BEST}" >/dev/null 2>&1; then
  echo "ERROR: best test already running." >&2
  exit 1
fi
if pgrep -f "main.py --test.*${TAG_LAST}" >/dev/null 2>&1; then
  echo "ERROR: last test already running." >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
: >"$LOG_BEST"
echo "=== $(date) single_MSF_trial ckpt-best (ep140) GPU${GPU_BEST} ===" | tee -a "$LOG_BEST"
CUDA_VISIBLE_DEVICES="$GPU_BEST" nohup "$PYTHON" -u main.py --test \
  --ckpts "$CKPT_BEST" \
  --config "$CFG" \
  --exp_name "test_${TAG_BEST}" \
  --model pgst \
  --num_workers 4 \
  >>"$LOG_BEST" 2>&1 &
echo "best_pid=$! log=$LOG_BEST gpu=$GPU_BEST"

: >"$LOG_LAST"
echo "=== $(date) single_MSF_trial ckpt-last (ep150) GPU${GPU_LAST} ===" | tee -a "$LOG_LAST"
CUDA_VISIBLE_DEVICES="$GPU_LAST" nohup "$PYTHON" -u main.py --test \
  --ckpts "$CKPT_LAST" \
  --config "$CFG" \
  --exp_name "test_${TAG_LAST}" \
  --model pgst \
  --num_workers 4 \
  >>"$LOG_LAST" 2>&1 &
echo "last_pid=$! log=$LOG_LAST gpu=$GPU_LAST"
echo "Monitor: tail -30 $LOG_BEST  |  tail -30 $LOG_LAST"
