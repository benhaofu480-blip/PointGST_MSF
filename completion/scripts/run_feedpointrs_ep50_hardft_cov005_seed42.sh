#!/usr/bin/env bash
# Stage-3: FeedPoinTrS ep50 -> hard mix (from ep50 mining) -> 50ep open-loop L_cover ft.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
PYTHON="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python"
GPU="${GPU:-1}"
SEED="${SEED:-42}"

MIX_LIST="data/PCN_hard_from_feedpointrs_ep50/hard_train_ft_mix.txt"
CKPT="experiments/AdaPoinTr_MSF_Pure_Group_sigmoid_feedpointrs_ft/PCN_models/exp_MSF_Pure_Group_sigmoid_feedpointrs_ft_crop02_04_seed42/ckpt-epoch-050.pth"
CFG="cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid_feedpointrs_hard_ft_cov005.yaml"
EXP="exp_feedpointrs_ep50_hardft_cov005_seed${SEED}"
LOG="/tmp/feedpointrs_ep50_hardft_cov005_seed${SEED}.log"

for f in "$MIX_LIST" "$CKPT"; do
  [[ -f "$f" ]] || { echo "Missing: $f (run run_mine_hard_from_feedpointrs_ep50.sh first)" >&2; exit 1; }
done

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
echo "log=$LOG"
echo "exp=experiments/AdaPoinTr_MSF_Pure_Group_sigmoid_feedpointrs_hard_ft_cov005/PCN_models/${EXP}"
echo "Monitor: grep -E 'cover|Validation|Overall|Early Stop|Training] EPOCH' $LOG"
