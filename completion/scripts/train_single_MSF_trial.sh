#!/usr/bin/env bash
# single_MSF_trial: AdaPoinTr_ps55 init, MSF + decoder block 7 + rebuild head trainable.
# Backup before this experiment: tools/backup/pre_single_msf_trial/
# Usage: bash scripts/train_single_MSF_trial.sh
#   MSF_SEED=42 MSF_DDP_PORT=29518 MSF_DDP_GPUS=0,1 bash scripts/train_single_MSF_trial.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${MSF_DDP_GPUS:-0,1}"
export LD_LIBRARY_PATH="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

PYTHON="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python"
CONFIG="cfgs/PCN_models/AdaPoinTr_MSF_single_MSF_trial.yaml"
SEED="${MSF_SEED:-42}"
EXP="exp_single_MSF_trial_seed${SEED}"
PORT="${MSF_DDP_PORT:-29518}"
LOG_DIR="logs/complete"
EXP_DIR="experiments/AdaPoinTr_MSF_single_MSF_trial/PCN_models/${EXP}"
TRAIN_LOG="${LOG_DIR}/single_MSF_trial_seed${SEED}.log"

if pgrep -f "main.py.*AdaPoinTr_MSF_single_MSF_trial.yaml" >/dev/null 2>&1; then
  echo "ERROR: single_MSF_trial training already running." >&2
  pgrep -af "main.py.*single_MSF_trial" || true
  exit 1
fi

mkdir -p "$LOG_DIR" "$EXP_DIR"
: > "$TRAIN_LOG"

echo "single_MSF_trial: MSF + decoder block 7 + rebuild head (ps55 init)."
echo "  CONFIG=$CONFIG"
echo "  EXP=$EXP"
echo "  optimizer.part=gft_single_decoder  trainable_decoder_block=7"
echo "  max_epoch=150  val_freq=10  lr=2e-4  LambdaLR"
echo "  GPUs=$CUDA_VISIBLE_DEVICES  DDP port=$PORT  seed=$SEED"
echo "  Log: $TRAIN_LOG"

nohup "$PYTHON" -u -m torch.distributed.run \
  --nproc_per_node=2 --master_port="$PORT" \
  main.py --launcher pytorch \
  --config "$CONFIG" \
  --exp_name "$EXP" \
  --model pgst \
  --seed "$SEED" \
  --num_workers 4 \
  > "$TRAIN_LOG" 2>&1 &

TRAIN_PID=$!
echo "PID=${TRAIN_PID}"
echo "Monitor: tail -f ${TRAIN_LOG}"
echo "Ckpt dir: ${EXP_DIR}"
