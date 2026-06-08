#!/usr/bin/env bash
# 真 PCSA Table VIII 对齐：adapter_mode=pcsa，--model pcsa，100ep，双卡 DDP
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${MSF_DDP_GPUS:-0,1}"
export LD_LIBRARY_PATH="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

PYTHON="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python"
CONFIG="cfgs/PCN_models/AdaPoinTr_PCSA_table8_true.yaml"
SEED="${MSF_SEED:-42}"
EXP="exp_PCSA_table8_true_seed${SEED}"
PORT="${MSF_DDP_PORT:-29522}"
LOG_DIR="logs/complete"
EXP_DIR="experiments/AdaPoinTr_PCSA_table8_true/PCN_models/${EXP}"
TRAIN_LOG="${LOG_DIR}/train_pcsa_table8_true_seed${SEED}.log"
CKPT_SRC="ckpt/AdaPoinTr_ps55.pth"

if pgrep -f "${PYTHON}.*/main.py.*AdaPoinTr_PCSA_table8_true.yaml" >/dev/null 2>&1; then
  echo "ERROR: PCSA table8 true training already running." >&2
  exit 1
fi

mkdir -p "$LOG_DIR" "$EXP_DIR"
: > "$TRAIN_LOG"

echo "Starting TRUE PCSA (adapter_mode=pcsa) table8 align."
echo "  CONFIG=$CONFIG  EXP=$EXP  --model pcsa"
echo "  center_num=[512,256] lr=1e-4 bs=16 max_epoch=100"
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

echo "PID=${!}"
echo "Monitor: tail -f ${TRAIN_LOG}"
