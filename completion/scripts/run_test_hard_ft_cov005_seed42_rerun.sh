#!/usr/bin/env bash
# Official PCN test: static hard_ft seed42 rerun ckpt-best.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
PYTHON="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python"

CKPT="${ROOT}/experiments/AdaPoinTr_MSF_Pure_Group_sigmoid_hard_ft_cov005/PCN_models/exp_MSF_Pure_Group_sigmoid_hard_ft_cov005_seed42_rerun/ckpt-best.pth"
CFG="cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid_hard_ft_cov005.yaml"
EXP="test_hard_ft_cov005_seed42_rerun_best"
LOG="/tmp/test_hard_ft_cov005_seed42_rerun_best.log"

[[ -f "$CKPT" ]] || { echo "Missing: $CKPT (wait for training)" >&2; exit 1; }

GPU="${GPU:-1}"
echo "Test seed42 rerun ckpt-best on GPU ${GPU}  log=$LOG"
CUDA_VISIBLE_DEVICES="$GPU" nohup "$PYTHON" -u main.py --test \
  --ckpts "$CKPT" \
  --config "$CFG" \
  --exp_name "$EXP" \
  --model pgst \
  --num_workers 4 \
  >>"$LOG" 2>&1 &

echo "pid=$!"
