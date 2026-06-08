#!/usr/bin/env bash
# DEPRECATED: use run_feedpointrs_ft_complete_s1ep150_seed42.sh (S1 ep150 init, 50ep).
# Stage-2 FeedPoinTrS-FT on PCN full train (complete), dual-GPU DDP.
# Usage: bash scripts/run_feedpointrs_ft_complete_seed42.sh
#   MSF_SEED=42 MSF_DDP_PORT=29514 bash scripts/run_feedpointrs_ft_complete_seed42.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${MSF_DDP_GPUS:-0,1}"
export LD_LIBRARY_PATH="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

PYTHON="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python"
CONFIG="cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid_feedpointrs_ft_complete.yaml"
SEED="${MSF_SEED:-42}"
EXP="exp_MSF_Pure_Group_sigmoid_feedpointrs_ft_crop02_04_complete_seed${SEED}"
PORT="${MSF_DDP_PORT:-29514}"
LOG_DIR="logs/complete"
EXP_DIR="experiments/AdaPoinTr_MSF_Pure_Group_sigmoid_feedpointrs_ft_complete/PCN_models/${EXP}"
TRAIN_LOG="${LOG_DIR}/train_feedpointrs_seed${SEED}.log"

S1_CKPT="experiments/AdaPoinTr_MSF_Pure_Group_sigmoid_complete/PCN_models/exp_MSF_Pure_Group_sigmoid_complete_seed${SEED}/ckpt-best.pth"

if pgrep -f "main.py.*AdaPoinTr_MSF_Pure_Group_sigmoid_feedpointrs_ft_complete.yaml" >/dev/null 2>&1; then
  echo "ERROR: feedpointrs_complete training already running." >&2
  pgrep -af "main.py.*feedpointrs_ft_complete" || true
  exit 1
fi

if [[ ! -f "$S1_CKPT" ]]; then
  echo "ERROR: Stage-1 complete ckpt not found:" >&2
  echo "  $S1_CKPT" >&2
  echo "Wait for Stage-1 complete training to finish, or set MSF_SEED to match." >&2
  exit 1
fi

mkdir -p "$LOG_DIR" "$EXP_DIR"
: > "$TRAIN_LOG"

echo "Starting Stage-2 FeedPoinTrS-FT on PCN full train."
echo "  CONFIG=$CONFIG"
echo "  EXP=$EXP"
echo "  Stage-1 ckpt=$S1_CKPT"
echo "  max_epoch=100  val_freq=5  crop=[0.2,0.4]"
echo "  GPUs=$CUDA_VISIBLE_DEVICES  DDP port=$PORT  seed=$SEED"
echo "  Log: $TRAIN_LOG"

nohup "$PYTHON" -u -m torch.distributed.run \
  --nproc_per_node=2 --master_port="$PORT" \
  main.py --launcher pytorch \
  --config "$CONFIG" \
  --exp_name "$EXP" \
  --model pgst \
  --seed "$SEED" \
  --start_ckpts "$S1_CKPT" \
  --num_workers 4 \
  > "$TRAIN_LOG" 2>&1 &

TRAIN_PID=$!
echo "PID=${TRAIN_PID}"
echo "Monitor: tail -f ${TRAIN_LOG}"
echo "Ckpt dir: ${EXP_DIR}"
