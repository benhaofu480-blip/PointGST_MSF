#!/usr/bin/env bash
# Official PCN test for hard_ft λ=0.15 ckpt-best.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
PYTHON="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python"

CKPT="${ROOT}/experiments/AdaPoinTr_MSF_Pure_Group_sigmoid_hard_ft_cov015/PCN_models/exp_MSF_Pure_Group_sigmoid_hard_ft_cov015/ckpt-best.pth"
CFG="cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid.yaml"
EXP="test_hard_ft_cov015_best"
LOG_DIR="${ROOT}/experiments/AdaPoinTr_MSF_Pure_Group_sigmoid_hard_ft_cov015/PCN_models/${EXP}"
mkdir -p "$LOG_DIR"

if [[ ! -f "$CKPT" ]]; then
  echo "Missing: $CKPT" >&2
  exit 1
fi

GPU="${GPU:-1}"
echo "Test λ=0.15 ckpt-best on GPU ${GPU}"
echo "  ckpt: $CKPT"
echo "  log:  ${LOG_DIR}/nohup.out"

CUDA_VISIBLE_DEVICES="$GPU" nohup "$PYTHON" -u main.py --test \
  --ckpts "$CKPT" \
  --config "$CFG" \
  --exp_name "$EXP" \
  --model pgst \
  --num_workers 4 \
  > "${LOG_DIR}/nohup.out" 2>&1 &

echo $! > "${LOG_DIR}/pid"
echo "pid=$(cat "${LOG_DIR}/pid")"
echo "watch: tail -f ${LOG_DIR}/nohup.out"
