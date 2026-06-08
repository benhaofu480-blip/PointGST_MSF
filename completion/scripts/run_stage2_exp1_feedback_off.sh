#!/usr/bin/env bash
# Stage-2 Exp1-OFF: single-pass finetune from C sigmoid ckpt-best (feedback disabled).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
PYTHON="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python"
GPU="${GPU:-1}"
SEED="${SEED:-42}"
EXP="exp_stage2_exp1_feedback_off_seed${SEED}"
LOG="/tmp/stage2_exp1_feedback_off_seed${SEED}.log"
CFG="cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid_stage2_singlepass_ft.yaml"

CKPT="experiments/AdaPoinTr_MSF_Pure_Group_sigmoid/PCN_models/exp_MSF_Pure_Group_sigmoid/ckpt-best.pth"
if [[ ! -f "$CKPT" ]]; then
  echo "Missing sigmoid ckpt: $CKPT" >&2
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
  > "$LOG" 2>&1 &

echo "pid=$!"
echo "exp=$EXP"
echo "log=$LOG"
echo "Monitor: grep -E 'Validation|Early Stop|Overall' $LOG"
