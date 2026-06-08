#!/usr/bin/env bash
# Static hard_ft λ=0.05, seed=42, full val(800) — reproducibility rerun (OHEM off).
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${MSF_DDP_GPUS:-1}"
export LD_LIBRARY_PATH="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

CKPT="experiments/AdaPoinTr_MSF_Pure_Group_sigmoid/PCN_models/exp_MSF_Pure_Group_sigmoid/ckpt-best.pth"
CFG="cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid_hard_ft_cov005.yaml"
EXP="exp_MSF_Pure_Group_sigmoid_hard_ft_cov005_seed42_rerun"
LOG="${MSF_LOG:-/tmp/msf_hard_ft_cov005_seed42_rerun.log}"

for f in data/PCN_hard/hard_train_ft_mix.txt "$CKPT"; do
  [[ -f "$f" ]] || { echo "Missing: $f" >&2; exit 1; }
done

echo "GPU=$CUDA_VISIBLE_DEVICES  seed=42  static hard_ft (OHEM off)  val=full(800)"
echo "exp=$EXP  log=$LOG"
exec "$PYTHON" -u main.py \
  --launcher none \
  --config "$CFG" \
  --exp_name "$EXP" \
  --start_ckpts "$CKPT" \
  --model pgst \
  --seed 42 \
  --num_workers "${MSF_NUM_WORKERS:-4}" \
  >>"$LOG" 2>&1
