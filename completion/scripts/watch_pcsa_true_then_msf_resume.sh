#!/usr/bin/env bash
# 串行：先真 PCSA 100ep 训完，再 MSF Sigmoid 续训 50ep（避免同时读大 ckpt）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python"
LOG_DIR="${ROOT}/logs/complete"
WATCH="${LOG_DIR}/watch_pcsa_true_then_msf_resume.log"
mkdir -p "$LOG_DIR"

exec >>"$WATCH" 2>&1
echo "=== watch started $(date) ==="

train_running() {
  pgrep -f "${PYTHON}.*/main.py" >/dev/null 2>&1
}

wait_train_done() {
  local tag="$1"
  echo "=== wait ${tag} training finish ==="
  for _ in $(seq 1 2880); do
    if ! train_running; then
      echo "${tag} done at $(date)"
      return 0
    fi
    sleep 30
  done
  echo "timeout waiting ${tag}"
  return 1
}

# 1) 真 PCSA
if pgrep -f "${PYTHON}.*/main.py.*AdaPoinTr_PCSA_table8_true.yaml" >/dev/null 2>&1; then
  echo "PCSA true already running, skip launch."
elif train_running; then
  echo "ERROR: other main.py train running; abort."
  exit 1
else
  bash scripts/train_pcsa_table8_true.sh
  for _ in $(seq 1 60); do
    pgrep -f "${PYTHON}.*/main.py.*AdaPoinTr_PCSA_table8_true.yaml" >/dev/null 2>&1 && break
    sleep 5
  done
  if ! pgrep -f "${PYTHON}.*/main.py.*AdaPoinTr_PCSA_table8_true.yaml" >/dev/null 2>&1; then
    echo "ERROR: PCSA true did not start"
    tail -20 "${LOG_DIR}/train_pcsa_table8_true_seed42.log" || true
    exit 1
  fi
fi
wait_train_done "PCSA true"

# 2) MSF Sigmoid resume
sleep 10
bash scripts/train_msf_table8_align_resume_ep101_150.sh
sleep 20
if ! pgrep -f "${PYTHON}.*/main.py.*AdaPoinTr_MSF_table8_align.yaml" >/dev/null 2>&1; then
  echo "ERROR: MSF resume did not start"
  tail -20 "${LOG_DIR}/train_msf_table8_align_ep101_150_seed42.log" || true
  exit 1
fi
wait_train_done "MSF table8 resume"

echo "=== all done $(date) ==="
