#!/usr/bin/env bash
# 原版 PCSA（Paper_related/PGST.py 同构实现），PCN 全量 train，100ep，双卡 DDP。
# 日志目录：logs/complete/
# Usage: bash scripts/train_pcsa_complete.sh

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${MSF_DDP_GPUS:-0,1}"
export LD_LIBRARY_PATH="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

PYTHON="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python"
CONFIG="cfgs/PCN_models/AdaPoinTr_PCSA_complete.yaml"
SEED="${MSF_SEED:-42}"
EXP="exp_PCSA_complete_seed${SEED}"
PORT="${MSF_DDP_PORT:-29515}"
LOG_DIR="logs/complete"
EXP_DIR="experiments/AdaPoinTr_PCSA_complete/PCN_models/${EXP}"
TRAIN_LOG="${LOG_DIR}/train_pcsa_seed${SEED}.log"

if pgrep -f "main.py.*AdaPoinTr_PCSA_complete.yaml" >/dev/null 2>&1; then
  echo "ERROR: PCSA complete training already running." >&2
  pgrep -af "main.py.*PCSA_complete" || true
  exit 1
fi

mkdir -p "$LOG_DIR" "$EXP_DIR"
: > "$TRAIN_LOG"

echo "Starting original PCSA on PCN full train (max_epoch=100)."
echo "  CONFIG=$CONFIG"
echo "  EXP=$EXP"
echo "  adapter_mode=pcsa  GPUs=$CUDA_VISIBLE_DEVICES  port=$PORT  seed=$SEED"
echo "  Log: $TRAIN_LOG"

nohup "$PYTHON" -u -m torch.distributed.run \
  --nproc_per_node=2 --master_port="$PORT" \
  main.py --launcher pytorch \
  --config "$CONFIG" \
  --exp_name "$EXP" \
  --model pcsa \
  --seed "$SEED" \
  --num_workers 4 \
  > "$TRAIN_LOG" 2>&1 &

TRAIN_PID=$!
echo "PID=${TRAIN_PID}"
echo "Monitor: tail -f ${TRAIN_LOG}"
echo "Ckpt dir: ${EXP_DIR}"
