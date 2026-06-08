#!/usr/bin/env bash
# Official PCN test: Stage-1 complete seed42, ckpt-best only.
# best 与 epoch-100 同为第 100 轮（val CDL1 最优也在该轮），无需重复测 last。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
PYTHON="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python"

EXP_DIR="${ROOT}/experiments/AdaPoinTr_MSF_Pure_Group_sigmoid_complete/PCN_models/exp_MSF_Pure_Group_sigmoid_complete_seed42"
CFG="cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid_complete.yaml"
LOG_DIR="${ROOT}/logs/complete"
LOG_BEST="${LOG_DIR}/test_sigmoid_complete_seed42_best.log"

GPU_BEST="${GPU_BEST:-0}"

mkdir -p "$LOG_DIR"

[[ -f "${EXP_DIR}/ckpt-best.pth" ]] || { echo "Missing ckpt-best" >&2; exit 1; }

if pgrep -f "main.py --test.*sigmoid_complete_seed42_best" >/dev/null 2>&1; then
  echo "ERROR: complete seed42 best test already running." >&2
  exit 1
fi

: >"$LOG_BEST"
echo "=== $(date) Stage-1 complete seed42 best-only GPU${GPU_BEST} ==="
CUDA_VISIBLE_DEVICES="$GPU_BEST" nohup "$PYTHON" -u main.py --test \
  --ckpts "${EXP_DIR}/ckpt-best.pth" \
  --config "$CFG" \
  --exp_name test_sigmoid_complete_seed42_best \
  --model pgst \
  --num_workers 4 \
  >>"$LOG_BEST" 2>&1 &
echo "best_pid=$! log=$LOG_BEST"
echo "Monitor: tail -f $LOG_BEST"
