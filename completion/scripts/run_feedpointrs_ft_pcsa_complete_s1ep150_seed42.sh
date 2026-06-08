#!/usr/bin/env bash
# Stage-2 FeedPoinTrS-FT (crop [0.2,0.4]) on PCN full train, init from PCSA Stage-1 complete ep150 ckpt-best.
# Usage: bash scripts/run_feedpointrs_ft_pcsa_complete_s1ep150_seed42.sh
#   MSF_SEED=42 MSF_DDP_PORT=29517 MSF_DDP_GPUS=0,1 bash scripts/run_feedpointrs_ft_pcsa_complete_s1ep150_seed42.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${MSF_DDP_GPUS:-0,1}"
export LD_LIBRARY_PATH="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

PYTHON="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python"
CONFIG="cfgs/PCN_models/AdaPoinTr_PCSA_feedpointrs_ft_complete.yaml"
SEED="${MSF_SEED:-42}"
EXP="exp_PCSA_feedpointrs_ft_crop02_04_complete_s1ep150_seed${SEED}"
PORT="${MSF_DDP_PORT:-29517}"
LOG_DIR="logs/complete"
EXP_DIR="experiments/AdaPoinTr_PCSA_feedpointrs_ft_complete/PCN_models/${EXP}"
TRAIN_LOG="${LOG_DIR}/train_pcsa_feedpointrs_s1ep150_seed${SEED}.log"

S1_CKPT_NFS="experiments/AdaPoinTr_PCSA_complete/PCN_models/exp_PCSA_complete_seed42/ckpt-best.pth"
S1_CKPT_TMP="/tmp/pcsa_complete_s1ep150_ckpt-best.pth"
if [[ ! -f "$S1_CKPT_TMP" ]] || [[ "$S1_CKPT_NFS" -nt "$S1_CKPT_TMP" ]]; then
  echo "Copy Stage-1 ckpt to $S1_CKPT_TMP (avoid NFS stall on load)..."
  cp -f "$S1_CKPT_NFS" "$S1_CKPT_TMP"
fi
S1_CKPT="$S1_CKPT_TMP"

if pgrep -f "main.py.*AdaPoinTr_PCSA_feedpointrs_ft_complete.yaml" >/dev/null 2>&1; then
  echo "ERROR: PCSA feedpointrs_complete training already running." >&2
  pgrep -af "main.py.*PCSA_feedpointrs_ft_complete" || true
  exit 1
fi

if [[ ! -f "$S1_CKPT_NFS" ]]; then
  echo "ERROR: PCSA Stage-1 ep150 ckpt-best not found: $S1_CKPT_NFS" >&2
  exit 1
fi

mkdir -p "$LOG_DIR" "$EXP_DIR"
: > "$TRAIN_LOG"

echo "Stage-2 FeedPoinTrS-FT (init PCSA Stage-1 ep150 ckpt-best)."
echo "  CONFIG=$CONFIG"
echo "  EXP=$EXP"
echo "  S1 ckpt=$S1_CKPT"
echo "  max_epoch=50  val_freq=5  early_stop_patience=25"
echo "  lr=5e-5  CosLR t_max=51  crop=[0.2,0.4]  pass_weight=2/1"
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
