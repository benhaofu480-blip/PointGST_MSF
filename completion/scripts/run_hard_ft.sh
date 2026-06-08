#!/usr/bin/env bash
# Phase-2 hard-sample fine-tune: C sigmoid ckpt-best + hard_train_ft_mix + L_cover.
# CosLR: initial_epochs=0 (no warmup); lr stepped at epoch start (see runner.py).
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${MSF_DDP_GPUS:-0,1}"
export LD_LIBRARY_PATH="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

MIX_LIST="data/PCN_hard/hard_train_ft_mix.txt"
if [[ ! -f "$MIX_LIST" ]]; then
  echo "Missing $MIX_LIST — run: python scripts/mine_hard_pc_samples.py" >&2
  exit 1
fi

CKPT="experiments/AdaPoinTr_MSF_Pure_Group_sigmoid/PCN_models/exp_MSF_Pure_Group_sigmoid/ckpt-best.pth"
CONFIG="cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid_hard_ft.yaml"
EXP="exp_MSF_Pure_Group_sigmoid_hard_ft"
LOG="/tmp/msf_sigmoid_hard_ft.log"

RESUME_ARGS=()
if [[ "${RESUME:-0}" == "1" ]]; then
  RESUME_ARGS=(--resume)
  echo "Resume from ${EXP} ckpt-last (epochs continue until max_epoch in yaml)"
else
  echo "Fresh start from C ckpt-best"
fi

START_ARGS=(--start_ckpts "$CKPT")
if [[ "${RESUME:-0}" == "1" ]]; then
  START_ARGS=()
fi

echo "Hard FT: mix=$(wc -l < "$MIX_LIST") lines  log=$LOG  RESUME=${RESUME:-0}"
exec "$PYTHON" -u main.py \
  --launcher none \
  --config "$CONFIG" \
  --exp_name "$EXP" \
  "${START_ARGS[@]}" \
  --model pgst \
  "${RESUME_ARGS[@]}" \
  --num_workers "${MSF_NUM_WORKERS:-4}" \
  2>&1 | tee -a "$LOG"
