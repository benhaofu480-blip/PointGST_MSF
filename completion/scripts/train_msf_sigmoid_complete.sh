#!/usr/bin/env bash
# Stage-1 纯组级 Sigmoid，PCN 全量 train（28974），双卡 DDP。
# 日志目录：logs/complete/（文件夹名标注 complete）
# Usage: bash scripts/train_msf_sigmoid_complete.sh
#   MSF_SEED=42 MSF_DDP_PORT=29513 bash scripts/train_msf_sigmoid_complete.sh

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${MSF_DDP_GPUS:-0,1}"
export LD_LIBRARY_PATH="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

PYTHON="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python"
CONFIG="cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid_complete.yaml"
SEED="${MSF_SEED:-42}"
EXP="exp_MSF_Pure_Group_sigmoid_complete_seed${SEED}"
PORT="${MSF_DDP_PORT:-29513}"
LOG_DIR="logs/complete"
EXP_DIR="experiments/AdaPoinTr_MSF_Pure_Group_sigmoid_complete/PCN_models/${EXP}"
TRAIN_LOG="${LOG_DIR}/train_sigmoid_seed${SEED}.log"

if pgrep -f "main.py.*AdaPoinTr_MSF_Pure_Group_sigmoid_complete.yaml" >/dev/null 2>&1; then
  echo "ERROR: sigmoid_complete training already running."
  pgrep -af "main.py.*sigmoid_complete" || true
  exit 1
fi

mkdir -p "$LOG_DIR" "$EXP_DIR"
: > "$TRAIN_LOG"

echo "Starting Stage-1 sigmoid on PCN full train (max_epoch=100, GitHub official)."
echo "  CONFIG=$CONFIG"
echo "  EXP=$EXP"
echo "  GPUs=$CUDA_VISIBLE_DEVICES  DDP port=$PORT  seed=$SEED"
echo "  Log: $TRAIN_LOG"

nohup "$PYTHON" -u -m torch.distributed.run \
  --nproc_per_node=2 --master_port="$PORT" \
  main.py --launcher pytorch \
  --config "$CONFIG" \
  --exp_name "$EXP" \
  --seed "$SEED" \
  --num_workers 4 \
  --model pgst \
  > "$TRAIN_LOG" 2>&1 &

TRAIN_PID=$!
echo "PID=${TRAIN_PID}"
echo "Monitor: tail -f ${TRAIN_LOG}"
echo "Ckpt dir: ${EXP_DIR}"
