#!/usr/bin/env bash
# Resume PCSA complete from ep100 ckpt-last, train ep101-150 (50 epochs).
# --resume reads experiment_path/config.yaml (must have max_epoch: 150).
# Usage: bash scripts/train_pcsa_complete_resume_ep101_150.sh
#   MSF_SEED=42 MSF_DDP_PORT=29515 MSF_DDP_GPUS=0,1 bash scripts/train_pcsa_complete_resume_ep101_150.sh

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
CKPT_LAST="${EXP_DIR}/ckpt-last.pth"
TRAIN_LOG="${LOG_DIR}/train_pcsa_complete_ep101_150_seed${SEED}.log"

if [[ ! -f "$CKPT_LAST" ]]; then
  echo "ERROR: missing ckpt-last for resume: $CKPT_LAST"
  exit 1
fi

EXP_CFG="${EXP_DIR}/config.yaml"
if ! grep -qE '^max_epoch[[:space:]]*:[[:space:]]*150' "$EXP_CFG" 2>/dev/null; then
  echo "ERROR: ${EXP_CFG} must have max_epoch: 150 before resume."
  exit 1
fi

if pgrep -f "main.py.*AdaPoinTr_PCSA_complete.yaml" >/dev/null 2>&1; then
  echo "ERROR: PCSA complete training already running."
  pgrep -af "main.py.*PCSA_complete" || true
  exit 1
fi

mkdir -p "$LOG_DIR"
: > "$TRAIN_LOG"

echo "Resume PCSA complete ep101-150 (max_epoch=150, --resume, --model pcsa)."
echo "  CONFIG=$CONFIG"
echo "  EXP=$EXP"
echo "  Resume: $CKPT_LAST"
echo "  GPUs=$CUDA_VISIBLE_DEVICES  DDP port=$PORT  seed=$SEED"
echo "  Log: $TRAIN_LOG"
echo "  Note: first batch may take 1-2 min (NFS dataloader); be patient."

nohup "$PYTHON" -u -m torch.distributed.run \
  --nproc_per_node=2 --master_port="$PORT" \
  main.py --launcher pytorch \
  --config "$CONFIG" \
  --exp_name "$EXP" \
  --model pcsa \
  --seed "$SEED" \
  --num_workers 4 \
  --resume \
  > "$TRAIN_LOG" 2>&1 &

TRAIN_PID=$!
echo "PID=${TRAIN_PID}"
echo "Monitor: tail -f ${TRAIN_LOG}"
echo "Ckpt dir: ${EXP_DIR}"
