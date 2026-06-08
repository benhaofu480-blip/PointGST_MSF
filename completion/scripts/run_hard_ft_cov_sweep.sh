#!/usr/bin/env bash
# λ (cover_weight) sweep: 0.15 then 0.05, no LR warmup, from C sigmoid ckpt-best.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${MSF_DDP_GPUS:-0}"
export LD_LIBRARY_PATH="/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

CKPT="experiments/AdaPoinTr_MSF_Pure_Group_sigmoid/PCN_models/exp_MSF_Pure_Group_sigmoid/ckpt-best.pth"
MIX_LIST="data/PCN_hard/hard_train_ft_mix.txt"

if [[ ! -f "$MIX_LIST" ]]; then
  echo "Missing $MIX_LIST — run: python scripts/mine_hard_pc_samples.py" >&2
  exit 1
fi
if [[ ! -f "$CKPT" ]]; then
  echo "Missing $CKPT" >&2
  exit 1
fi

run_one() {
  local cfg="$1"
  local exp="$2"
  local log="$3"
  echo "========== cover_weight sweep: $exp =========="
  echo "config=$cfg  log=$log"
  "$PYTHON" -u main.py \
    --launcher none \
    --config "$cfg" \
    --exp_name "$exp" \
    --start_ckpts "$CKPT" \
    --model pgst \
    --num_workers "${MSF_NUM_WORKERS:-4}" \
    2>&1 | tee "$log"
}

# 0.15 first, then 0.05 (override RUN_ONLY=015|005 to run a single job)
if [[ "${RUN_ONLY:-}" == "015" ]]; then
  run_one \
    "cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid_hard_ft_cov015.yaml" \
    "exp_MSF_Pure_Group_sigmoid_hard_ft_cov015" \
    "/tmp/msf_hard_ft_cov015.log"
elif [[ "${RUN_ONLY:-}" == "005" ]]; then
  run_one \
    "cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid_hard_ft_cov005.yaml" \
    "exp_MSF_Pure_Group_sigmoid_hard_ft_cov005" \
    "/tmp/msf_hard_ft_cov005.log"
else
  run_one \
    "cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid_hard_ft_cov015.yaml" \
    "exp_MSF_Pure_Group_sigmoid_hard_ft_cov015" \
    "/tmp/msf_hard_ft_cov015.log"
  run_one \
    "cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid_hard_ft_cov005.yaml" \
    "exp_MSF_Pure_Group_sigmoid_hard_ft_cov005" \
    "/tmp/msf_hard_ft_cov005.log"
fi

echo "Done. Logs: /tmp/msf_hard_ft_cov015.log /tmp/msf_hard_ft_cov005.log"
