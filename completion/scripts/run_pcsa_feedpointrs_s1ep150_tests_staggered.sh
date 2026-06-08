#!/usr/bin/env bash
# Official PCN test: PCSA Stage-2 feedpointrs ckpt-best (ep25) then ckpt-last (ep50), staggered load.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
PYTHON="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python"
LOG_DIR="${ROOT}/logs/complete"
WATCH="${LOG_DIR}/watch_pcsa_feedpointrs_s1ep150_tests.log"
GPU_BEST="${GPU_BEST:-0}"
GPU_LAST="${GPU_LAST:-1}"

EXP_DIR="${ROOT}/experiments/AdaPoinTr_PCSA_feedpointrs_ft_complete/PCN_models/exp_PCSA_feedpointrs_ft_crop02_04_complete_s1ep150_seed42"
CFG="cfgs/PCN_models/AdaPoinTr_PCSA_feedpointrs_ft_complete.yaml"
LOG_BEST="${LOG_DIR}/test_pcsa_feedpointrs_s1ep150_seed42_best.log"
LOG_LAST="${LOG_DIR}/test_pcsa_feedpointrs_s1ep150_seed42_last.log"
TAG_BEST="pcsa_feedpointrs_s1ep150_seed42_best"
TAG_LAST="pcsa_feedpointrs_s1ep150_seed42_last"
CKPT_BEST_TMP="/tmp/pcsa_feedpointrs_s1ep150_ckpt-best.pth"
CKPT_LAST_TMP="/tmp/pcsa_feedpointrs_s1ep150_ckpt-last.pth"

mkdir -p "$LOG_DIR"
exec >>"$WATCH" 2>&1

[[ -f "${EXP_DIR}/ckpt-best.pth" ]] || { echo "Missing ckpt-best" >&2; exit 1; }
[[ -f "${EXP_DIR}/ckpt-last.pth" ]] || { echo "Missing ckpt-last" >&2; exit 1; }

if pgrep -f "main.py --test.*${TAG_BEST}" >/dev/null 2>&1; then
  echo "ERROR: ${TAG_BEST} already running." >&2
  exit 1
fi
if pgrep -f "main.py --test.*${TAG_LAST}" >/dev/null 2>&1; then
  echo "ERROR: ${TAG_LAST} already running." >&2
  exit 1
fi

echo "Copy ckpts to /tmp for faster load..."
cp -f "${EXP_DIR}/ckpt-best.pth" "$CKPT_BEST_TMP"
cp -f "${EXP_DIR}/ckpt-last.pth" "$CKPT_LAST_TMP"

wait_ckpt_loaded() {
  local log="$1" pid="$2" label="$3"
  echo "=== $(date) wait ${label}: ckpt loaded (Test[200/1200]) ==="
  for _ in $(seq 1 360); do
    if grep -q 'Test\[200/1200\]' "$log" 2>/dev/null; then
      grep 'Test\[200/1200\]' "$log" | tail -1
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "${label} exited before Test[200/1200]"
      tail -20 "$log" || true
      return 1
    fi
    sleep 10
  done
  echo "timeout waiting Test[200/1200] for ${label}"
  return 1
}

wait_test_done() {
  local pid="$1" label="$2" log="$3"
  if kill -0 "$pid" 2>/dev/null; then
    wait "$pid" || true
  fi
  echo "=== $(date) DONE ${label} ==="
  grep -E 'Overall|TEST RESULTS' "$log" | tail -5 || tail -8 "$log"
  echo ""
}

echo "=== PCSA feedpointrs complete seed42 stagger tests $(date) GPU_BEST=${GPU_BEST} GPU_LAST=${GPU_LAST} ==="

: >"$LOG_BEST"
echo "=== $(date) START best (ep25) GPU${GPU_BEST} ==="
CUDA_VISIBLE_DEVICES="$GPU_BEST" nohup "$PYTHON" -u main.py --test \
  --ckpts "$CKPT_BEST_TMP" \
  --config "$CFG" \
  --exp_name "test_${TAG_BEST}" \
  --model pgst \
  --num_workers 4 \
  >>"$LOG_BEST" 2>&1 &
BEST_PID=$!
echo "best_pid=${BEST_PID} log=${LOG_BEST}"

wait_ckpt_loaded "$LOG_BEST" "$BEST_PID" "best"

: >"$LOG_LAST"
echo "=== $(date) START last (ep50) GPU${GPU_LAST} (best still running) ==="
CUDA_VISIBLE_DEVICES="$GPU_LAST" nohup "$PYTHON" -u main.py --test \
  --ckpts "$CKPT_LAST_TMP" \
  --config "$CFG" \
  --exp_name "test_${TAG_LAST}" \
  --model pgst \
  --num_workers 4 \
  >>"$LOG_LAST" 2>&1 &
LAST_PID=$!
echo "last_pid=${LAST_PID} log=${LOG_LAST}"

wait_test_done "$BEST_PID" "best" "$LOG_BEST"
wait_test_done "$LAST_PID" "last" "$LOG_LAST"

echo "=== all PCSA feedpointrs seed42 tests finished $(date) ==="
