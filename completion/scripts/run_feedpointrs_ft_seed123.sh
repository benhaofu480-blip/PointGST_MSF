#!/usr/bin/env bash
# Exp1-ON replicate: FeedPoinTrS double-pass, crop [0.2,0.4], 50ep, seed123.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$(dirname "$0")/_logs_dir.sh"

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
PYTHON="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python"
GPU="${GPU:-1}"
SEED=123
EXP="exp_MSF_Pure_Group_sigmoid_feedpointrs_ft_crop02_04_seed${SEED}"
LOG="${LOG_ROOT}/feedpointrs_ft_crop02_04_seed${SEED}.log"
CFG="cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid_feedpointrs_ft.yaml"

CKPT="experiments/AdaPoinTr_MSF_Pure_Group_sigmoid/PCN_models/exp_MSF_Pure_Group_sigmoid/ckpt-best.pth"
[[ -f "$CKPT" ]] || { echo "Missing $CKPT" >&2; exit 1; }

if pgrep -f "main.py.*feedpointrs_ft_crop02_04_seed${SEED}" >/dev/null 2>&1; then
  echo "ERROR: seed${SEED} training already running." >&2
  exit 1
fi

: > "$LOG"
CUDA_VISIBLE_DEVICES="$GPU" nohup "$PYTHON" -u main.py \
  --launcher none \
  --config "$CFG" \
  --exp_name "$EXP" \
  --model pgst \
  --seed "$SEED" \
  --start_ckpts "$CKPT" \
  --num_workers 4 \
  >> "$LOG" 2>&1 &

echo "pid=$!"
echo "exp=$EXP"
echo "log=$LOG"
echo "Monitor: grep -E 'Validation|Early Stop' $LOG | tail -5"
