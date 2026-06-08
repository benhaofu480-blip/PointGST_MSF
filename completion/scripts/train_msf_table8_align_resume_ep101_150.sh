#!/usr/bin/env bash
# MSF Sigmoid table8：从 ep100 ckpt-last 续训至 150（+50 轮），双卡 DDP
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
CKPT_LAST="${EXP_DIR}/ckpt-last.pth"
EXP_CFG="${EXP_DIR}/config.yaml"
TRAIN_LOG="${LOG_DIR}/train_msf_table8_align_ep101_150_seed${SEED}.log"

[[ -f "$CKPT_LAST" ]] || { echo "ERROR: missing $CKPT_LAST"; exit 1; }

if pgrep -f "${PYTHON}.*/main.py.*AdaPoinTr_MSF_table8_align.yaml" >/dev/null 2>&1; then
  echo "ERROR: MSF table8 training already running." >&2
  exit 1
fi

# --resume 读 experiment_path/config.yaml，需 max_epoch=150、t_max=150
if [[ -f "$EXP_CFG" ]]; then
  sed -i 's/^max_epoch[[:space:]]*:.*/max_epoch : 150/' "$EXP_CFG"
  sed -i 's/t_max:[[:space:]]*100/t_max: 150/' "$EXP_CFG"
else
  echo "ERROR: missing $EXP_CFG"; exit 1
fi

mkdir -p "$LOG_DIR"
: > "$TRAIN_LOG"

echo "Resume MSF Sigmoid table8 ep101-150 (--resume from ckpt-last)."
echo "  EXP=$EXP  ckpt=$CKPT_LAST"
echo "  Patched $EXP_CFG -> max_epoch=150, t_max=150"
echo "  Log: $TRAIN_LOG"

nohup "$PYTHON" -u -m torch.distributed.run \
  --nproc_per_node=2 --master_port="$PORT" \
  main.py --launcher pytorch \
  --config "$CONFIG" \
  --exp_name "$EXP" \
  --model pgst \
  --seed "$SEED" \
  --num_workers 4 \
  --resume \
  > "$TRAIN_LOG" 2>&1 &

echo "PID=${!}"
echo "Monitor: tail -f ${TRAIN_LOG}"
