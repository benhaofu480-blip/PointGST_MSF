#!/usr/bin/env bash
# Launch Tanh (B) or Sigmoid (C) ablation from AdaPoinTr_ps55.pth (dual-GPU DDP).
# Usage: bash scripts/train_msf_ablation_bc.sh tanh|sigmoid
# B and C must run one at a time (each needs both GPUs). Do not overlap with Softmax baseline (A).

set -euo pipefail
VARIANT="${1:-}"
if [[ "$VARIANT" != "tanh" && "$VARIANT" != "sigmoid" ]]; then
  echo "Usage: $0 tanh|sigmoid"
  exit 1
fi

cd "$(dirname "$0")/.."

# 双卡 DDP 必须至少 2 张可见 GPU。若用 `CUDA_VISIBLE_DEVICES=0 nohup ...` 启动编排器，
# 继承的 `CUDA_VISIBLE_DEVICES=0` 会导致 local_rank=1 -> invalid device ordinal。
# 默认强制 0,1；单卡机器请改 MSF_DDP_GPUS 或改为单进程训练（本脚本仍为 2 进程）。
export CUDA_VISIBLE_DEVICES="${MSF_DDP_GPUS:-0,1}"
export LD_LIBRARY_PATH="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

PYTHON="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python"

is_training_cfg() {
  pgrep -f "main.py.*${1}" >/dev/null 2>&1
}

for cfg in \
  "AdaPoinTr_MSF_Pure_Group_tanh.yaml" \
  "AdaPoinTr_MSF_Pure_Group_sigmoid.yaml"; do
  if is_training_cfg "$cfg"; then
    echo "ERROR: Training already running (${cfg}). Wait until it finishes."
    exit 1
  fi
done
# Block B/C only if Softmax (A) is actively training (orchestrator waits for A separately).
if is_training_cfg "AdaPoinTr_MSF_Pure_Group.yaml"; then
  echo "ERROR: Softmax baseline (A) still running. Use run_msf_ablation_overnight.sh or wait."
  exit 1
fi

if [[ "$VARIANT" == "tanh" ]]; then
  CONFIG=cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_tanh.yaml
  EXP=exp_MSF_Pure_Group_tanh
  PORT=29507
  LOG_DIR=experiments/AdaPoinTr_MSF_Pure_Group_tanh/PCN_models/exp_MSF_Pure_Group_tanh
else
  CONFIG=cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid.yaml
  EXP=exp_MSF_Pure_Group_sigmoid
  PORT=29508
  LOG_DIR=experiments/AdaPoinTr_MSF_Pure_Group_sigmoid/PCN_models/exp_MSF_Pure_Group_sigmoid
fi

mkdir -p "$LOG_DIR"

echo "Starting $VARIANT ablation (DDP, port $PORT)..."
nohup "$PYTHON" -u -m torch.distributed.run \
  --nproc_per_node=2 --master_port="$PORT" \
  main.py --launcher pytorch \
  --config "$CONFIG" \
  --exp_name "$EXP" \
  --num_workers 4 \
  --model pgst \
  > "$LOG_DIR/train.log" 2>&1 &
echo "PID=$!  log=$LOG_DIR/train.log"
