#!/usr/bin/env bash
# Official PCN test @ep150 ckpt-best: Sigmoid on GPU0, then PCSA on GPU1 after Sigmoid is confirmed running.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
PYTHON="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python"
LOG_DIR="${ROOT}/logs/complete"
mkdir -p "$LOG_DIR"

GPU_SIGMOID="${GPU_SIGMOID:-0}"
GPU_PCSA="${GPU_PCSA:-1}"

SIG_EXP="${ROOT}/experiments/AdaPoinTr_MSF_Pure_Group_sigmoid_complete/PCN_models/exp_MSF_Pure_Group_sigmoid_complete_seed42"
SIG_CFG="cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid_complete.yaml"
SIG_LOG="${LOG_DIR}/test_sigmoid_complete_ep150_seed42_best.log"
SIG_TAG="sigmoid_complete_ep150_seed42_best"

PCS_EXP="${ROOT}/experiments/AdaPoinTr_PCSA_complete/PCN_models/exp_PCSA_complete_seed42"
PCS_CFG="cfgs/PCN_models/AdaPoinTr_PCSA_complete.yaml"
PCS_LOG="${LOG_DIR}/test_pcsa_complete_ep150_seed42_best.log"
PCS_TAG="pcsa_complete_ep150_seed42_best"

[[ -f "${SIG_EXP}/ckpt-best.pth" ]] || { echo "Missing ${SIG_EXP}/ckpt-best.pth" >&2; exit 1; }
[[ -f "${PCS_EXP}/ckpt-best.pth" ]] || { echo "Missing ${PCS_EXP}/ckpt-best.pth" >&2; exit 1; }

wait_started() {
  local log="$1" tag="$2" max_wait="${3:-300}"
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
  echo "ERROR: ${tag} test did not start within ${max_wait}s (log=${log})" >&2
  tail -20 "$log" >&2 || true
  return 1
}

if pgrep -f "main.py --test.*${SIG_TAG}" >/dev/null 2>&1; then
  echo "ERROR: Sigmoid ep150 test already running." >&2
  exit 1
fi
if pgrep -f "main.py --test.*${PCS_TAG}" >/dev/null 2>&1; then
  echo "ERROR: PCSA ep150 test already running." >&2
  exit 1
fi

: >"$SIG_LOG"
echo "=== $(date) Sigmoid complete ep150 ckpt-best GPU${GPU_SIGMOID} ===" | tee -a "$SIG_LOG"
CUDA_VISIBLE_DEVICES="$GPU_SIGMOID" nohup "$PYTHON" -u main.py --test \
  --ckpts "${SIG_EXP}/ckpt-best.pth" \
  --config "$SIG_CFG" \
  --exp_name "test_${SIG_TAG}" \
  --model pgst \
  --num_workers 4 \
  >>"$SIG_LOG" 2>&1 &
SIG_PID=$!
echo "Sigmoid started pid=${SIG_PID} gpu=${GPU_SIGMOID} log=${SIG_LOG}"

wait_started "$SIG_LOG" "$SIG_TAG" 600
echo "Sigmoid test confirmed running on GPU ${GPU_SIGMOID}."

: >"$PCS_LOG"
echo "=== $(date) PCSA complete ep150 ckpt-best GPU${GPU_PCSA} ===" | tee -a "$PCS_LOG"
CUDA_VISIBLE_DEVICES="$GPU_PCSA" nohup "$PYTHON" -u main.py --test \
  --ckpts "${PCS_EXP}/ckpt-best.pth" \
  --config "$PCS_CFG" \
  --exp_name "test_${PCS_TAG}" \
  --model pcsa \
  --num_workers 4 \
  >>"$PCS_LOG" 2>&1 &
PCS_PID=$!
echo "PCSA started pid=${PCS_PID} gpu=${GPU_PCSA} log=${PCS_LOG}"

wait_started "$PCS_LOG" "$PCS_TAG" 600
echo "PCSA test confirmed running on GPU ${GPU_PCSA}."
echo "Monitor:"
echo "  tail -f ${SIG_LOG}"
echo "  tail -f ${PCS_LOG}"
