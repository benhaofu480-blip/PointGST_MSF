#!/usr/bin/env bash
# Resume FeedPoinTrS finetune from ep35 ckpt-last, train ep36-50 (15 epochs).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
PYTHON="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python"
GPU="${GPU:-1}"
SEED="${SEED:-42}"
EXP="exp_MSF_Pure_Group_sigmoid_feedpointrs_ft_crop02_04_seed${SEED}"
LOG="/tmp/feedpointrs_ft_crop02_04_seed${SEED}_ep36_50.log"
CKPT_LAST="${ROOT}/experiments/AdaPoinTr_MSF_Pure_Group_sigmoid_feedpointrs_ft/PCN_models/${EXP}/ckpt-last.pth"

if [[ ! -f "$CKPT_LAST" ]]; then
  echo "Missing ckpt-last: $CKPT_LAST" >&2
  exit 1
fi

: > "$LOG"
CUDA_VISIBLE_DEVICES="$GPU" nohup "$PYTHON" -u main.py \
  --launcher none \
  --config cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid_feedpointrs_ft.yaml \
  --exp_name "$EXP" \
  --model pgst \
  --seed "$SEED" \
  --resume \
  --num_workers 4 \
  > "$LOG" 2>&1 &

echo "pid=$!"
echo "log=$LOG"
echo "Resume from: $CKPT_LAST"
echo "Monitor: grep -E 'RESUME|Validation|Overall|Early Stop|Training] EPOCH' $LOG"
