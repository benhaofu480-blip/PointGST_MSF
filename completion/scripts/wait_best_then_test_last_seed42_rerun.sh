#!/usr/bin/env bash
# After best test hits Test[200/1200], start ckpt-last test on GPU0.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python"
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

EXP_DIR="experiments/AdaPoinTr_MSF_Pure_Group_sigmoid_hard_ft_cov005/PCN_models/exp_MSF_Pure_Group_sigmoid_hard_ft_cov005_seed42_rerun"
CFG="cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid_hard_ft_cov005.yaml"
LOG_BEST="/tmp/test_hard_ft_cov005_seed42_rerun_best.log"
LOG_LAST="/tmp/test_hard_ft_cov005_seed42_rerun_last.log"

echo "[wait] polling $LOG_BEST for Test[200/1200]..."
for _ in $(seq 1 240); do
  if grep -q 'Test\[200/1200\]' "$LOG_BEST" 2>/dev/null; then
    echo "[wait] hit Test[200/1200]"
    break
  fi
  if ! pgrep -f "test_hard_ft_cov005_seed42_rerun_best" >/dev/null 2>&1; then
    echo "[wait] best test process ended"
    break
  fi
  sleep 15
done

: >"$LOG_LAST"
echo "[wait] starting ckpt-last on GPU0"
CUDA_VISIBLE_DEVICES=0 nohup "$PY" -u main.py --test \
  --ckpts "${EXP_DIR}/ckpt-last.pth" \
  --config "$CFG" \
  --exp_name test_hard_ft_cov005_seed42_rerun_last \
  --model pgst --num_workers 4 >>"$LOG_LAST" 2>&1 &
echo "last_pid=$! log=$LOG_LAST"
