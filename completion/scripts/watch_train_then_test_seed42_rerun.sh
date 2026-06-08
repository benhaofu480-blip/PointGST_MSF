#!/usr/bin/env bash
# Wait for seed42_rerun training, then test ckpt-best; after Test[200/1200], test ckpt-last.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python"
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

TRAIN_LOG="/tmp/msf_hard_ft_cov005_seed42_rerun.log"
EXP_DIR="experiments/AdaPoinTr_MSF_Pure_Group_sigmoid_hard_ft_cov005/PCN_models/exp_MSF_Pure_Group_sigmoid_hard_ft_cov005_seed42_rerun"
CFG="cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid_hard_ft_cov005.yaml"
LOG_BEST="/tmp/test_hard_ft_cov005_seed42_rerun_best.log"
LOG_LAST="/tmp/test_hard_ft_cov005_seed42_rerun_last.log"
WATCH_LOG="/tmp/watch_seed42_rerun_test.log"

exec >>"$WATCH_LOG" 2>&1
echo "=== watch started $(date) ==="

# 1) Wait for training to finish
echo "[watch] waiting for training (grep Early Stop or max epoch / process exit)..."
while pgrep -f "main.py.*cov005_seed42_rerun" >/dev/null 2>&1; do
  sleep 60
done

# wait until process really gone
for _ in $(seq 1 120); do
  pgrep -f "main.py.*cov005_seed42_rerun" >/dev/null 2>&1 || break
  sleep 10
done

while [[ ! -f "${EXP_DIR}/ckpt-best.pth" ]]; do
  sleep 30
done
echo "[watch] training done; ckpt-best ready"

grep -E 'Validation\] EPOCH|Early Stop|Save checkpoint.*best' "$TRAIN_LOG" | tail -8 || true

# 2) Test ckpt-best on GPU1
: >"$LOG_BEST"
echo "[watch] starting ckpt-best test on GPU1 $(date)"
CUDA_VISIBLE_DEVICES=1 "$PY" -u main.py --test \
  --ckpts "${EXP_DIR}/ckpt-best.pth" \
  --config "$CFG" \
  --exp_name test_hard_ft_cov005_seed42_rerun_best \
  --model pgst \
  --num_workers 4 \
  >>"$LOG_BEST" 2>&1 &
PID_BEST=$!
echo "[watch] best test pid=$PID_BEST log=$LOG_BEST"

# 3) Wait Test[200/1200] then start ckpt-last on GPU0
echo "[watch] waiting for Test[200/1200] in best log..."
for _ in $(seq 1 200); do
  if grep -q 'Test\[200/1200\]' "$LOG_BEST" 2>/dev/null; then
    echo "[watch] hit Test[200/1200] at $(date)"
    grep 'Test\[200/1200\]' "$LOG_BEST" | tail -1
    break
  fi
  if ! kill -0 "$PID_BEST" 2>/dev/null; then
    echo "[watch] best test exited early"; tail -20 "$LOG_BEST"
    break
  fi
  sleep 15
done

: >"$LOG_LAST"
echo "[watch] starting ckpt-last test on GPU0 $(date)"
CUDA_VISIBLE_DEVICES=0 "$PY" -u main.py --test \
  --ckpts "${EXP_DIR}/ckpt-last.pth" \
  --config "$CFG" \
  --exp_name test_hard_ft_cov005_seed42_rerun_last \
  --model pgst \
  --num_workers 4 \
  >>"$LOG_LAST" 2>&1 &
PID_LAST=$!
echo "[watch] last test pid=$PID_LAST log=$LOG_LAST"

wait "$PID_BEST" 2>/dev/null || true
wait "$PID_LAST" 2>/dev/null || true

echo "=== all tests done $(date) ==="
echo "--- ckpt-best ---"
grep -E "Overall|TEST\] Metrics" "$LOG_BEST" | tail -3 || tail -5 "$LOG_BEST"
echo "--- ckpt-last ---"
grep -E "Overall|TEST\] Metrics" "$LOG_LAST" | tail -3 || tail -5 "$LOG_LAST"
