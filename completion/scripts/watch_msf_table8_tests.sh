#!/usr/bin/env bash
# 等 PCSA best+last 都结束后，MSF 同样错开 ckpt 读取、双卡并行 test
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
PYTHON="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python"
LOG_DIR="${ROOT}/logs/complete"
WATCH="${LOG_DIR}/watch_table8_align_tests.log"
GPU_A="${TABLE8_GPU_A:-0}"
GPU_B="${TABLE8_GPU_B:-1}"

exec >>"$WATCH" 2>&1
echo "=== MSF watcher started $(date) ==="

wait_pcsa_done() {
  echo "=== wait PCSA tests finish ==="
  for _ in $(seq 1 720); do
    if ! pgrep -f "${PYTHON}.*/main.py --test.*pcsa_table8_align_seed42" >/dev/null 2>&1; then
      echo "PCSA tests done at $(date)"
      return 0
    fi
    sleep 15
  done
  echo "timeout waiting PCSA"
  return 1
}

wait_ckpt_loaded() {
  local log="$1" pid="$2"
  for _ in $(seq 1 360); do
    grep -q 'Test\[200/1200\]' "$log" 2>/dev/null && return 0
    kill -0 "$pid" 2>/dev/null || return 1
    sleep 10
  done
  return 1
}

MSF_EXP="${ROOT}/experiments/AdaPoinTr_MSF_table8_align/PCN_models/exp_MSF_table8_align_seed42"
MSF_CFG="cfgs/PCN_models/AdaPoinTr_MSF_table8_align.yaml"
LOG_BEST="${LOG_DIR}/test_msf_table8_align_seed42_best.log"
LOG_LAST="${LOG_DIR}/test_msf_table8_align_seed42_last.log"

wait_pcsa_done

: >"$LOG_BEST"
echo "=== $(date) START MSF best GPU${GPU_A} ==="
CUDA_VISIBLE_DEVICES="$GPU_A" nohup "$PYTHON" -u main.py --test \
  --ckpts "${MSF_EXP}/ckpt-best.pth" --config "$MSF_CFG" \
  --exp_name test_msf_table8_align_seed42_best --model pgst --num_workers 4 \
  >>"$LOG_BEST" 2>&1 &
PID_B=$!
echo "msf best pid=$PID_B"

wait_ckpt_loaded "$LOG_BEST" "$PID_B"
echo "=== $(date) START MSF last GPU${GPU_B} ==="
: >"$LOG_LAST"
CUDA_VISIBLE_DEVICES="$GPU_B" nohup "$PYTHON" -u main.py --test \
  --ckpts "${MSF_EXP}/ckpt-last.pth" --config "$MSF_CFG" \
  --exp_name test_msf_table8_align_seed42_last --model pgst --num_workers 4 \
  >>"$LOG_LAST" 2>&1 &
PID_L=$!
echo "msf last pid=$PID_L"

wait "$PID_B" 2>/dev/null || true
wait "$PID_L" 2>/dev/null || true
echo "=== MSF tests done $(date) ==="
grep -E 'Overall' "$LOG_BEST" | tail -1 || true
grep -E 'Overall' "$LOG_LAST" | tail -1 || true
