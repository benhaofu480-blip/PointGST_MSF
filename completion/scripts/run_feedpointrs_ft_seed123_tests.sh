#!/usr/bin/env bash
# Official PCN test: Exp1-ON seed123, ckpt-best + ckpt-last.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$(dirname "$0")/_logs_dir.sh"

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
PYTHON="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python"

EXP_DIR="${ROOT}/experiments/AdaPoinTr_MSF_Pure_Group_sigmoid_feedpointrs_ft/PCN_models/exp_MSF_Pure_Group_sigmoid_feedpointrs_ft_crop02_04_seed123"
CFG="cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid_feedpointrs_ft.yaml"
LOG_BEST="${LOG_ROOT}/test_feedpointrs_crop02_04_seed123_best.log"
LOG_LAST="${LOG_ROOT}/test_feedpointrs_crop02_04_seed123_last.log"
WATCH="${LOG_ROOT}/watch_feedpointrs_seed123_tests.log"

GPU_BEST="${GPU_BEST:-0}"
GPU_LAST="${GPU_LAST:-1}"

[[ -f "${EXP_DIR}/ckpt-best.pth" ]] || { echo "Missing ckpt-best" >&2; exit 1; }
[[ -f "${EXP_DIR}/ckpt-last.pth" ]] || { echo "Missing ckpt-last" >&2; exit 1; }

if pgrep -f "main.py --test.*seed123" >/dev/null 2>&1; then
  echo "ERROR: seed123 test already running." >&2
  exit 1
fi

# Avoid loading 300MB ckpts on slow disks; best/last differ by mtime at end of training.
BEST_MTIME=$(stat -c %Y "${EXP_DIR}/ckpt-best.pth")
LAST_MTIME=$(stat -c %Y "${EXP_DIR}/ckpt-last.pth")
SAME_EPOCH=0
[[ "$BEST_MTIME" == "$LAST_MTIME" ]] && SAME_EPOCH=1

: >"$LOG_BEST"
echo "=== $(date) seed123 best GPU${GPU_BEST} same_epoch=${SAME_EPOCH} ===" | tee "$WATCH"
CUDA_VISIBLE_DEVICES="$GPU_BEST" nohup "$PYTHON" -u main.py --test \
  --ckpts "${EXP_DIR}/ckpt-best.pth" \
  --config "$CFG" \
  --exp_name test_feedpointrs_crop02_04_seed123_best \
  --model pgst \
  --num_workers 4 \
  >>"$LOG_BEST" 2>&1 &
BEST_PID=$!
echo "best_pid=$BEST_PID log=$LOG_BEST"

if [[ "$SAME_EPOCH" == "1" ]]; then
  echo "best==last epoch; skip second test."
  exit 0
fi

(
  exec >>"$WATCH" 2>&1
  echo "=== wait best Test[200/1200] ==="
  for _ in $(seq 1 360); do
    if grep -q 'Test\[200/1200\]' "$LOG_BEST" 2>/dev/null; then break; fi
    if ! kill -0 "$BEST_PID" 2>/dev/null; then tail -20 "$LOG_BEST"; exit 1; fi
    sleep 15
  done
  echo "=== start last GPU${GPU_LAST} ==="
  : >"$LOG_LAST"
  CUDA_VISIBLE_DEVICES="$GPU_LAST" nohup "$PYTHON" -u main.py --test \
    --ckpts "${EXP_DIR}/ckpt-last.pth" \
    --config "$CFG" \
    --exp_name test_feedpointrs_crop02_04_seed123_last \
    --model pgst \
    --num_workers 4 \
    >>"$LOG_LAST" 2>&1 &
  echo "last_pid=$! log=$LOG_LAST"
) &

echo "Monitor: tail -f $LOG_BEST $WATCH"
