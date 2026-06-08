#!/usr/bin/env bash
# Sigmoid + 轻量点级通道残差（C+point），从 AdaPoinTr_ps55.pth 双卡 DDP 训练。
# Usage: bash scripts/train_msf_sigmoid_point.sh

set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${MSF_DDP_GPUS:-0,1}"
export LD_LIBRARY_PATH="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

PYTHON="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python"
CONFIG="cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid_point.yaml"
EXP="exp_MSF_Pure_Group_sigmoid_point"
PORT="${MSF_DDP_PORT:-29509}"
LOG_DIR="experiments/AdaPoinTr_MSF_Pure_Group_sigmoid_point/PCN_models/${EXP}"

if pgrep -f "main.py.*AdaPoinTr_MSF_Pure_Group_sigmoid_point.yaml" >/dev/null 2>&1; then
  echo "ERROR: sigmoid_point training already running."
  exit 1
fi

mkdir -p "$LOG_DIR"

echo "Starting sigmoid+point (point_delta_scale=0.05, DDP port $PORT)..."
nohup "$PYTHON" -u -m torch.distributed.run \
  --nproc_per_node=2 --master_port="$PORT" \
  main.py --launcher pytorch \
  --config "$CONFIG" \
  --exp_name "$EXP" \
  --num_workers 4 \
  --model pgst \
  > "$LOG_DIR/train.log" 2>&1 &
echo "PID=$!  log=$LOG_DIR/train.log"
