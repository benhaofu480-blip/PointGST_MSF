#!/usr/bin/env bash
# F2: Sigmoid MSF + KNN route -> rebuild head。从 ps55 训练。
# 默认：单进程 DataParallel 双卡（CUDA 0,1 + --launcher none），避免 torchrun 子进程卡死。
# 坚持 DDP 两进程：MSF_USE_DDP=1 bash scripts/train_msf_sigmoid_rebuild.sh
# Phase0: AdaPoinTr_MSF_Pure_Group_sigmoid.yaml  (msf_route_mode: none)

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${MSF_DDP_GPUS:-0,1}"
export LD_LIBRARY_PATH="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

PYTHON="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python"
CONFIG="cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid_rebuild.yaml"
EXP="exp_MSF_Pure_Group_sigmoid_rebuild"
LOG_DIR="experiments/AdaPoinTr_MSF_Pure_Group_sigmoid_rebuild/PCN_models/${EXP}"
TMP_LOG="/tmp/msf_sigmoid_rebuild_train.log"
NUM_WORKERS="${MSF_NUM_WORKERS:-4}"

if pgrep -f "main.py.*AdaPoinTr_MSF_Pure_Group_sigmoid_rebuild.yaml" >/dev/null 2>&1; then
  echo "ERROR: sigmoid_rebuild training already running."
  pgrep -af "main.py.*sigmoid_rebuild" || true
  exit 1
fi

mkdir -p "$LOG_DIR"
: > "$TMP_LOG"
: > "$LOG_DIR/train.log"

if [[ "${MSF_USE_DDP:-0}" == "1" ]]; then
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  export MASTER_PORT="${MSF_DDP_PORT:-29512}"
  export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  echo "Mode: DDP 2-process (${MASTER_ADDR}:${MASTER_PORT}) GPUs=${CUDA_VISIBLE_DEVICES}"
  nohup "$PYTHON" -u -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=2 \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    main.py --launcher pytorch \
    --config "$CONFIG" \
    --exp_name "$EXP" \
    --num_workers "$NUM_WORKERS" \
    --model pgst \
    >> "$TMP_LOG" 2>&1 &
else
  echo "Mode: DataParallel 2-GPU single-process GPUs=${CUDA_VISIBLE_DEVICES}"
  nohup "$PYTHON" -u main.py \
    --launcher none \
    --config "$CONFIG" \
    --exp_name "$EXP" \
    --num_workers "$NUM_WORKERS" \
    --model pgst \
    >> "$TMP_LOG" 2>&1 &
fi

TRAIN_PID=$!
echo "PID=${TRAIN_PID}"
echo "Log: ${TMP_LOG}  (also tail ${LOG_DIR}/train.log after copy)"
echo "Monitor: tail -f ${TMP_LOG}"

# 后台同步到实验目录（不阻塞启动）
(
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    cp -f "$TMP_LOG" "$LOG_DIR/train.log" 2>/dev/null || true
    sleep 30
  done
  cp -f "$TMP_LOG" "$LOG_DIR/train.log" 2>/dev/null || true
) &
