#!/usr/bin/env bash
# Table8 official test：错开 ckpt 读取，测试阶段可双卡并行
# 策略：先起 best，等 Test[200/1200]（权重已加载）再起 last；PCSA 一组完成后同样方式跑 MSF
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
mkdir -p "$LOG_DIR"

exec >>"$WATCH" 2>&1

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
      tail -15 "$log" || true
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
  grep -E 'Overall|TEST RESULTS|TEST\] Metrics' "$log" | tail -5 || tail -8 "$log"
  echo ""
}

run_pair() {
  local tag="$1" exp_dir="$2" cfg="$3"
  local log_best="${LOG_DIR}/test_${tag}_best.log"
  local log_last="${LOG_DIR}/test_${tag}_last.log"
  local exp_best="test_${tag}_best"
  local exp_last="test_${tag}_last"

  [[ -f "${exp_dir}/ckpt-best.pth" ]] || { echo "Missing ${exp_dir}/ckpt-best.pth"; return 1; }
  [[ -f "${exp_dir}/ckpt-last.pth" ]] || { echo "Missing ${exp_dir}/ckpt-last.pth"; return 1; }

  if pgrep -f "${PYTHON}.*/main.py --test.*${exp_best}" >/dev/null 2>&1; then
    echo "=== $(date) ${tag} best already running, attach watcher ==="
    BEST_PID=$(pgrep -f "${PYTHON}.*/main.py --test.*${exp_best}" | head -1)
  else
    : >"$log_best"
    echo "=== $(date) START ${tag} best GPU${GPU_A} ==="
    CUDA_VISIBLE_DEVICES="$GPU_A" nohup "$PYTHON" -u main.py --test \
      --ckpts "${exp_dir}/ckpt-best.pth" \
      --config "$cfg" \
      --exp_name "$exp_best" \
      --model pgst \
      --num_workers 4 \
      >>"$log_best" 2>&1 &
    BEST_PID=$!
    echo "best_pid=${BEST_PID} log=${log_best}"
  fi

  wait_ckpt_loaded "$log_best" "$BEST_PID" "${tag} best" || return 1

  if pgrep -f "${PYTHON}.*/main.py --test.*${exp_last}" >/dev/null 2>&1; then
    echo "${tag} last already running"
    LAST_PID=$(pgrep -f "${PYTHON}.*/main.py --test.*${exp_last}" | head -1)
  else
    : >"$log_last"
    echo "=== $(date) START ${tag} last GPU${GPU_B} (best still running) ==="
    CUDA_VISIBLE_DEVICES="$GPU_B" nohup "$PYTHON" -u main.py --test \
      --ckpts "${exp_dir}/ckpt-last.pth" \
      --config "$cfg" \
      --exp_name "$exp_last" \
      --model pgst \
      --num_workers 4 \
      >>"$log_last" 2>&1 &
    LAST_PID=$!
    echo "last_pid=${LAST_PID} log=${log_last}"
  fi

  wait_test_done "$BEST_PID" "${tag} best" "$log_best"
  wait_test_done "$LAST_PID" "${tag} last" "$log_last"
}

PCS_EXP="${ROOT}/experiments/AdaPoinTr_PCSA_table8_align/PCN_models/exp_PCSA_table8_align_seed42"
PCS_CFG="cfgs/PCN_models/AdaPoinTr_PCSA_table8_align.yaml"
MSF_EXP="${ROOT}/experiments/AdaPoinTr_MSF_table8_align/PCN_models/exp_MSF_table8_align_seed42"
MSF_CFG="cfgs/PCN_models/AdaPoinTr_MSF_table8_align.yaml"

echo "=== table8 stagger tests $(date) GPU_A=${GPU_A} GPU_B=${GPU_B} ==="

run_pair "pcsa_table8_align_seed42" "$PCS_EXP" "$PCS_CFG"
run_pair "msf_table8_align_seed42" "$MSF_EXP" "$MSF_CFG"

echo "=== all table8 align tests finished $(date) ==="
