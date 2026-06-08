#!/usr/bin/env bash
# MSF Sigmoid 论文 Table VIII PCN 补全设定对齐：center_num=[512,256], lr=1e-4, CosLR, 100ep
# Usage: bash scripts/train_msf_table8_align.sh

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${MSF_DDP_GPUS:-0,1}"
export LD_LIBRARY_PATH="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

PYTHON="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python"
CONFIG="cfgs/PCN_models/AdaPoinTr_MSF_table8_align.yaml"
SEED="${MSF_SEED:-42}"
EXP="exp_MSF_table8_align_seed${SEED}"
PORT="${MSF_DDP_PORT:-29521}"
LOG_DIR="logs/complete"
EXP_DIR="experiments/AdaPoinTr_MSF_table8_align/PCN_models/${EXP}"
TRAIN_LOG="${LOG_DIR}/train_msf_table8_align_seed${SEED}.log"
CKPT_SRC="ckpt/AdaPoinTr_ps55.pth"
CKPT_TMP="/tmp/AdaPoinTr_ps55_msf_table8.pth"

if pgrep -f "main.py.*AdaPoinTr_MSF_table8_align.yaml" >/dev/null 2>&1; then
  echo "ERROR: MSF table8 training already running." >&2
  pgrep -af "AdaPoinTr_MSF_table8_align" || true
  exit 1
fi

mkdir -p "$LOG_DIR" "$EXP_DIR"
if [[ -f "$CKPT_SRC" ]]; then
  cp -f "$CKPT_SRC" "$CKPT_TMP"
  echo "Copied pretrained ckpt to ${CKPT_TMP}"
fi

: > "$TRAIN_LOG"

echo "Starting Table-VIII-aligned MSF Sigmoid training."
echo "  CONFIG=$CONFIG"
echo "  EXP=$EXP"
echo "  center_num=[512,256] lr=1e-4 bs=16 max_epoch=100 CosLR val_freq=5"
echo "  --model pgst (MSF sigmoid adapter)  pretrained=AdaPoinTr_ps55.pth  seed=$SEED"
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

echo "PID=${!}"
echo "Monitor: tail -f ${TRAIN_LOG}"
echo "Ckpt dir: ${EXP_DIR}"
