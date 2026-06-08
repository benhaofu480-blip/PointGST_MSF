#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python"
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
EXP="experiments/AdaPoinTr_MSF_Pure_Group_sigmoid_hard_ft_cov005_critical30/PCN_models/exp_MSF_Pure_Group_sigmoid_hard_ft_cov005_critical30_seed42"
CFG="cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid_hard_ft_cov005_critical30.yaml"
LOG_BEST="/tmp/test_critical30_seed42_best.log"
LOG_LAST="/tmp/test_critical30_seed42_last.log"
WATCH="/tmp/watch_critical30_test.log"

exec >>"$WATCH" 2>&1
echo "=== $(date) wait for best Test[200/1200] ==="
for _ in $(seq 1 240); do
  if grep -q 'Test\[200/1200\]' "$LOG_BEST" 2>/dev/null; then break; fi
  if ! pgrep -f "test_critical30_seed42_best" >/dev/null 2>&1; then
    echo "best test stopped"; tail -10 "$LOG_BEST"; exit 1
  fi
  sleep 15
done
echo "=== $(date) start last on GPU0 ==="
: >"$LOG_LAST"
CUDA_VISIBLE_DEVICES=0 nohup "$PY" -u main.py --test \
  --ckpts "${EXP}/ckpt-last.pth" --config "$CFG" \
  --exp_name test_critical30_seed42_last --model pgst --num_workers 4 >>"$LOG_LAST" 2>&1 &
echo "last_pid=$!"
while pgrep -f "test_critical30_seed42" >/dev/null 2>&1; do sleep 30; done
echo "=== done $(date) ==="
grep -E "Overall|TEST\] Metrics" "$LOG_BEST" | tail -2
grep -E "Overall|TEST\] Metrics" "$LOG_LAST" | tail -2
