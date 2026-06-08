#!/usr/bin/env bash
# Exp1-ON ablation: double-pass feedback, fixed crop_ratio=0.30 (min=max).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$(dirname "$0")/_logs_dir.sh"

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
PYTHON="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python"
GPU="${GPU:-1}"
SEED="${SEED:-42}"
EXP="exp_stage2_exp3_crop030_fixed_seed${SEED}"
LOG="${LOG_ROOT}/stage2_exp3_crop030_fixed_seed${SEED}.log"
CFG="cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid_feedpointrs_ft_crop030_fixed.yaml"

CKPT="experiments/AdaPoinTr_MSF_Pure_Group_sigmoid/PCN_models/exp_MSF_Pure_Group_sigmoid/ckpt-best.pth"
if [[ ! -f "$CKPT" ]]; then
  echo "Missing sigmoid ckpt: $CKPT" >&2
  exit 1
fi

if pgrep -f "main.py.*crop030_fixed" >/dev/null 2>&1; then
  echo "ERROR: crop030_fixed training already running." >&2
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
echo "Monitor: grep -E 'FeedPoinTrS|Validation|Early Stop' $LOG | tail -5"
