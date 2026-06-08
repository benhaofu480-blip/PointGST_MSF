#!/usr/bin/env bash
# PCN_core：扫 test 挑 15 例（Stage-2 CD 更优、每类最多 2 个）并出 2×3 对比图
set -euo pipefail
cd "$(dirname "$0")/.."
source ~/data/fubenhao_data/miniforge3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate pgst
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
OUT="Vision/output/stage1_vs_stage2_pcn_core/batch15"
mkdir -p "$OUT"
LOG="$OUT/run.log"
echo "start $(date -Iseconds)" | tee "$LOG"
python -u vis_compare_stage1_vs_stage2.py --batch 15 2>&1 | tee -a "$LOG"
echo "done $(date -Iseconds)" | tee -a "$LOG"
