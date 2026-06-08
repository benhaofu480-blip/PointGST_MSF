#!/usr/bin/env bash
# Official PCN test: Stage-2 feedpointrs ckpt-best (ep25) on GPU0, then ckpt-last (ep50) on GPU1.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
PYTHON="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python"

EXP_DIR="${ROOT}/experiments/AdaPoinTr_MSF_Pure_Group_sigmoid_feedpointrs_ft_complete/PCN_models/exp_MSF_Pure_Group_sigmoid_feedpointrs_ft_crop02_04_complete_s1ep150_seed42"
CFG="cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid_feedpointrs_ft_complete.yaml"
LOG_DIR="${ROOT}/logs/complete"
GPU_BEST="${GPU_BEST:-0}"
GPU_LAST="${GPU_LAST:-1}"

CKPT_BEST="${EXP_DIR}/ckpt-best.pth"
CKPT_LAST="${EXP_DIR}/ckpt-last.pth"
CKPT_BEST_TMP="/tmp/feedpointrs_s1ep150_ckpt-best.pth"
CKPT_LAST_TMP="/tmp/feedpointrs_s1ep150_ckpt-last.pth"

LOG_BEST="${LOG_DIR}/test_feedpointrs_s1ep150_seed42_best.log"
LOG_LAST="${LOG_DIR}/test_feedpointrs_s1ep150_seed42_last.log"
TAG_BEST="feedpointrs_s1ep150_seed42_best"
TAG_LAST="feedpointrs_s1ep150_seed42_last"

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

echo "Copy ckpts to /tmp for faster load..."
cp -f "$CKPT_BEST" "$CKPT_BEST_TMP"
cp -f "$CKPT_LAST" "$CKPT_LAST_TMP"

wait_started() {
  local log="$1" tag="$2" max_wait="${3:-600}"
  local i=0
  while (( i < max_wait )); do
    if pgrep -f "main.py --test.*${tag}" >/dev/null 2>&1; then
      if grep -qE 'Test\[|Loading weights|args.test : True' "$log" 2>/dev/null; then
        return 0
      fi
    fi
    sleep 2
    i=$((i + 2))
  done
  echo "ERROR: ${tag} test did not start (log=${log})" >&2
  tail -15 "$log" >&2 || true
  return 1
}

mkdir -p "$LOG_DIR"
: >"$LOG_BEST"
echo "=== $(date) Stage-2 feedpointrs ckpt-best (ep25) GPU${GPU_BEST} ===" | tee -a "$LOG_BEST"
CUDA_VISIBLE_DEVICES="$GPU_BEST" nohup "$PYTHON" -u main.py --test \
  --ckpts "$CKPT_BEST_TMP" \
  --config "$CFG" \
  --exp_name "test_${TAG_BEST}" \
  --model pgst \
  --num_workers 4 \
  >>"$LOG_BEST" 2>&1 &
echo "best_pid=$! log=$LOG_BEST gpu=$GPU_BEST"

wait_started "$LOG_BEST" "$TAG_BEST" 600
echo "Best test confirmed on GPU ${GPU_BEST}."

: >"$LOG_LAST"
echo "=== $(date) Stage-2 feedpointrs ckpt-last (ep50) GPU${GPU_LAST} ===" | tee -a "$LOG_LAST"
CUDA_VISIBLE_DEVICES="$GPU_LAST" nohup "$PYTHON" -u main.py --test \
  --ckpts "$CKPT_LAST_TMP" \
  --config "$CFG" \
  --exp_name "test_${TAG_LAST}" \
  --model pgst \
  --num_workers 4 \
  >>"$LOG_LAST" 2>&1 &
echo "last_pid=$! log=$LOG_LAST gpu=$GPU_LAST"

wait_started "$LOG_LAST" "$TAG_LAST" 600
echo "Last test confirmed on GPU ${GPU_LAST}."
echo "Monitor:"
echo "  tail -f $LOG_BEST"
echo "  tail -f $LOG_LAST"
