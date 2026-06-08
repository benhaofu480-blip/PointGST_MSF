#!/usr/bin/env bash
# Official PCN test: Exp3 fixed crop (0.30), ckpt-best + ckpt-last.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$(dirname "$0")/_logs_dir.sh"

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
PYTHON="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python"

EXP_DIR="${ROOT}/experiments/AdaPoinTr_MSF_Pure_Group_sigmoid_feedpointrs_ft_crop030_fixed/PCN_models/exp_stage2_exp3_crop030_fixed_seed42"
CFG="cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid_feedpointrs_ft_crop030_fixed.yaml"
LOG_BEST="${LOG_ROOT}/test_stage2_exp3_crop030_fixed_best.log"
LOG_LAST="${LOG_ROOT}/test_stage2_exp3_crop030_fixed_last.log"
WATCH="${LOG_ROOT}/watch_stage2_exp3_crop030_fixed_tests.log"

GPU_BEST="${GPU_BEST:-0}"
GPU_LAST="${GPU_LAST:-1}"

[[ -f "${EXP_DIR}/ckpt-best.pth" ]] || { echo "Missing ckpt-best" >&2; exit 1; }
[[ -f "${EXP_DIR}/ckpt-last.pth" ]] || { echo "Missing ckpt-last" >&2; exit 1; }

if pgrep -f "main.py --test.*crop030_fixed" >/dev/null 2>&1; then
  echo "ERROR: crop030_fixed test already running." >&2
  exit 1
fi

SAME_EPOCH=$("$PYTHON" - <<'PY'
import torch
from pathlib import Path
exp = Path("experiments/AdaPoinTr_MSF_Pure_Group_sigmoid_feedpointrs_ft_crop030_fixed/PCN_models/exp_stage2_exp3_crop030_fixed_seed42")
b = torch.load(exp / "ckpt-best.pth", map_location="cpu")
l = torch.load(exp / "ckpt-last.pth", map_location="cpu")
print(int(b["epoch"] == l["epoch"]))
PY
)

: >"$LOG_BEST"
echo "=== $(date) start best (GPU${GPU_BEST}), same_epoch=${SAME_EPOCH} ===" | tee "$WATCH"
CUDA_VISIBLE_DEVICES="$GPU_BEST" nohup "$PYTHON" -u main.py --test \
  --ckpts "${EXP_DIR}/ckpt-best.pth" \
  --config "$CFG" \
  --exp_name test_stage2_exp3_crop030_fixed_best \
  --model pgst \
  --num_workers 4 \
  >>"$LOG_BEST" 2>&1 &
BEST_PID=$!
echo "best_pid=$BEST_PID log=$LOG_BEST"

if [[ "$SAME_EPOCH" == "1" ]]; then
  echo "best and last are the same epoch; skip second test."
  exit 0
fi

(
  exec >>"$WATCH" 2>&1
  echo "=== $(date) wait for best Test[200/1200] ==="
  for _ in $(seq 1 360); do
    if grep -q 'Test\[200/1200\]' "$LOG_BEST" 2>/dev/null; then break; fi
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
    --exp_name test_stage2_exp3_crop030_fixed_last \
    --model pgst \
    --num_workers 4 \
    >>"$LOG_LAST" 2>&1 &
  echo "last_pid=$! log=$LOG_LAST"
) &

echo "Monitor: tail -f $LOG_BEST ; tail -f $WATCH"
