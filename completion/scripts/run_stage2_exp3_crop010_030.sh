#!/usr/bin/env bash
# Stage-2 Exp3: crop ratio ablation [0.10, 0.30], dual-pass feedback finetune.
# True dual-GPU DDP (2 processes, one GPU each). Requires CUDA_VISIBLE_DEVICES=0,1.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${GPU:-0,1}"
export LD_LIBRARY_PATH="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29521}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

PYTHON="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python"
SEED="${SEED:-42}"
EXP="exp_stage2_exp3_crop010_030_seed${SEED}"
LOG="/tmp/stage2_exp3_crop010_030_seed${SEED}.log"
CFG="cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid_feedpointrs_ft_crop010_030.yaml"

CKPT="experiments/AdaPoinTr_MSF_Pure_Group_sigmoid/PCN_models/exp_MSF_Pure_Group_sigmoid/ckpt-best.pth"
if [[ ! -f "$CKPT" ]]; then
  echo "Missing sigmoid ckpt: $CKPT" >&2
  exit 1
fi

if pgrep -f "main.py.*feedpointrs_ft_crop010_030" >/dev/null 2>&1; then
  echo "ERROR: crop010_030 training already running." >&2
  pgrep -af "main.py.*feedpointrs_ft_crop010_030" || true
  exit 1
fi

: > "$LOG"
nohup "$PYTHON" -u -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=2 \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  main.py --launcher pytorch \
  --config "$CFG" \
  --exp_name "$EXP" \
  --model pgst \
  --seed "$SEED" \
  --start_ckpts "$CKPT" \
  --num_workers 4 \
  > "$LOG" 2>&1 &

echo "pid=$!"
echo "mode=DDP nproc=2 GPUs=$CUDA_VISIBLE_DEVICES port=$MASTER_PORT"
echo "exp=$EXP"
echo "log=$LOG"
echo "Monitor: grep -E 'Distributed Data parallel|FeedPoinTrS|Validation|Early Stop' $LOG"
