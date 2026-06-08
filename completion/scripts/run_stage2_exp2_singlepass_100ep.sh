#!/usr/bin/env bash
# Stage-2 Exp2: single-pass 100 epoch (fair compute vs Exp1-ON 50ep double-pass).
# Same crop [0.2,0.4] and lr=5e-5 as Exp1-ON; feedback disabled.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$(dirname "$0")/_logs_dir.sh"

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
PYTHON="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python"
GPU="${GPU:-1}"
SEED="${SEED:-42}"
EXP="exp_stage2_exp2_singlepass_100ep_seed${SEED}"
LOG="${LOG_ROOT}/stage2_exp2_singlepass_100ep_seed${SEED}.log"
CFG="cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid_stage2_singlepass_100ep.yaml"

CKPT="experiments/AdaPoinTr_MSF_Pure_Group_sigmoid/PCN_models/exp_MSF_Pure_Group_sigmoid/ckpt-best.pth"
if [[ ! -f "$CKPT" ]]; then
  echo "Missing sigmoid ckpt: $CKPT" >&2
  exit 1
fi

if pgrep -f "main.py.*singlepass_100ep" >/dev/null 2>&1; then
  echo "ERROR: Exp2 100ep training already running." >&2
  exit 1
fi

: > "$LOG"
CUDA_VISIBLE_DEVICES="$GPU" nohup "$PYTHON" -u main.py \
  --launcher none \
  --config "$CFG" \
  --exp_name "$EXP" \
  --model pgst \
  --seed "$SEED" \
  --start_ckpts "$CKPT" \
  --num_workers 4 \
  >> "$LOG" 2>&1 &

echo "pid=$!"
echo "exp=$EXP"
echo "log=$LOG"
echo "Monitor: grep -E 'Epoch|Validation|Early Stop' $LOG | tail -5"
