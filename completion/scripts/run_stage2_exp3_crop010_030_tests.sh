#!/usr/bin/env bash
# Official PCN test for Exp3-C crop [0.10, 0.30]: ckpt-best first, ckpt-last after best hits 200/1200.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
PYTHON="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python"

EXP_DIR="${ROOT}/experiments/AdaPoinTr_MSF_Pure_Group_sigmoid_feedpointrs_ft_crop010_030/PCN_models/exp_stage2_exp3_crop010_030_seed42"
CFG="cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid_feedpointrs_ft_crop010_030.yaml"
LOG_BEST="/tmp/test_stage2_exp3_crop010_030_best.log"
LOG_LAST="/tmp/test_stage2_exp3_crop010_030_last.log"
WATCH="/tmp/watch_stage2_exp3_crop010_030_tests.log"

GPU_BEST="${GPU_BEST:-0}"
GPU_LAST="${GPU_LAST:-1}"

[[ -f "${EXP_DIR}/ckpt-best.pth" ]] || { echo "Missing ckpt-best: ${EXP_DIR}/ckpt-best.pth" >&2; exit 1; }
[[ -f "${EXP_DIR}/ckpt-last.pth" ]] || { echo "Missing ckpt-last: ${EXP_DIR}/ckpt-last.pth" >&2; exit 1; }

if pgrep -f "main.py --test.*crop010_030" >/dev/null 2>&1; then
  echo "ERROR: crop010_030 test already running." >&2
  pgrep -af "main.py --test.*crop010_030" || true
  exit 1
fi

: >"$LOG_BEST"
echo "=== $(date) start best (GPU${GPU_BEST}) ===" | tee "$WATCH"
CUDA_VISIBLE_DEVICES="$GPU_BEST" nohup "$PYTHON" -u main.py --test \
  --ckpts "${EXP_DIR}/ckpt-best.pth" \
  --config "$CFG" \
  --exp_name test_stage2_exp3_crop010_030_best \
  --model pgst \
  --num_workers 4 \
  >>"$LOG_BEST" 2>&1 &
BEST_PID=$!
echo "best_pid=$BEST_PID log=$LOG_BEST"

(
  exec >>"$WATCH" 2>&1
  echo "=== $(date) wait for best Test[200/1200] ==="
  for _ in $(seq 1 360); do
    if grep -q 'Test\[200/1200\]' "$LOG_BEST" 2>/dev/null; then
      break
    fi
    if ! kill -0 "$BEST_PID" 2>/dev/null; then
      echo "best test exited early"; tail -20 "$LOG_BEST"; exit 1
    fi
    sleep 15
  done
  if ! grep -q 'Test\[200/1200\]' "$LOG_BEST" 2>/dev/null; then
    echo "timeout waiting for Test[200/1200]"; exit 1
  fi
  echo "=== $(date) start last (GPU${GPU_LAST}) ==="
  : >"$LOG_LAST"
  CUDA_VISIBLE_DEVICES="$GPU_LAST" nohup "$PYTHON" -u main.py --test \
    --ckpts "${EXP_DIR}/ckpt-last.pth" \
    --config "$CFG" \
    --exp_name test_stage2_exp3_crop010_030_last \
    --model pgst \
    --num_workers 4 \
    >>"$LOG_LAST" 2>&1 &
  echo "last_pid=$! log=$LOG_LAST"
) &

echo "watch_pid=$!"
echo "Monitor:"
echo "  tail -f $LOG_BEST"
echo "  tail -f $WATCH"
