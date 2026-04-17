#!/bin/bash
# SAMA: Spectrum-Aware Missing Adaptation
# Point cloud completion on PCN dataset
# 
# Usage:
#   bash train_pgst_sama.sh          # Baseline (PCSA only)
#   # Then edit yaml to enable improvements:
#   #   use_sama: true   -> Enable SAMA
#   #   use_mssa: true   -> Enable Multi-Scale Separate Adaptation
#   #   both true        -> SAMA + MSSA
#
# SWAP优化：num_workers=0 消除多进程竞争，配合内存缓存降低IO压力

export CUDA_VISIBLE_DEVICES=0
nice -n 10 python main.py \
    --config cfgs/PCN_models/AdaPoinTr_pgst_sama.yaml \
    --exp_name pgst_sama \
    --model pgst \
    --val_freq 5 \
    --num_workers 0
