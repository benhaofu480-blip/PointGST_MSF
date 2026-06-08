#!/usr/bin/env bash
# 双卡 F2 训练：先写日志再 import，避免“空 log + 无 GPU”误判。
# Usage: bash scripts/run_rebuild_dual_gpu.sh
# 看日志（勿开太多 tail -f，会 inotify 耗尽）:  watch -n 10 'tail -30 /tmp/msf_sigmoid_rebuild_train.log'

set -euo pipefail
cd "$(dirname "$0")/.."

LOG="/tmp/msf_sigmoid_rebuild_train.log"
PYTHON="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python"
CONFIG="cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid_rebuild.yaml"
EXP="exp_MSF_Pure_Group_sigmoid_rebuild"
EXP_LOG_DIR="experiments/AdaPoinTr_MSF_Pure_Group_sigmoid_rebuild/PCN_models/${EXP}"

if pgrep -f "main.py.*AdaPoinTr_MSF_Pure_Group_sigmoid_rebuild.yaml" >/dev/null 2>&1; then
  echo "ERROR: already running:" >&2
  pgrep -af "main.py.*sigmoid_rebuild" >&2 || true
  exit 1
fi

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${MSF_DDP_GPUS:-0,1}"
export LD_LIBRARY_PATH="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

mkdir -p "$EXP_LOG_DIR"
: > "$LOG"

exec >>"$LOG" 2>&1
echo "========== $(date '+%F %T') run_rebuild_dual_gpu =========="
echo "GPUs=${CUDA_VISIBLE_DEVICES}  cwd=$(pwd)"
echo "Step 1/2: import torch (may take 1-5 min on busy server)..."
"$PYTHON" -u -c "import torch; print('torch OK, cuda=', torch.cuda.is_available(), 'n=', torch.cuda.device_count())"
echo "Step 2/2: start training (DataParallel, launcher none)..."
exec "$PYTHON" -u main.py \
  --launcher none \
  --config "$CONFIG" \
  --exp_name "$EXP" \
  --num_workers "${MSF_NUM_WORKERS:-0}" \
  --model pgst
