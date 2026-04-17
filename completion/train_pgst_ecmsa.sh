#!/bin/bash
# ECMSA: Eigenvalue-Conditioned Multi-Scale Spectral Adapter
# Point cloud completion on PCN dataset
# 
# Usage:
#   bash train_pgst_ecmsa.sh          # Baseline (PCSA only)
#   # Then edit yaml to enable improvements:
#   #   use_ecfr: true   -> ECFR eigenvalue gating
#   #   use_mssa: true   -> Multi-Scale Separate Adaptation
#   #   both true        -> ECMSA full
#
# SWAP优化：num_workers=0 消除多进程竞争，配合内存缓存降低IO压力

export CUDA_VISIBLE_DEVICES=0
nice -n 10 python main.py \
    --config cfgs/PCN_models/AdaPoinTr_pgst_ecmsa.yaml \
    --exp_name pgst_ecmsa \
    --start_ckpts ckpts/AdaPoinTr_ps55.pth \
    --model pgst \
    --val_freq 5 \
    --num_workers 0
