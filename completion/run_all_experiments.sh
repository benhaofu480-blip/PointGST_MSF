#!/bin/bash
# Three experiments on 3x RTX 4090D
# Exp1: Baseline (PCSA only)          -> GPU 0
# Exp2: +SAMA (Spectrum-Aware)        -> GPU 1
# Exp3: +MSSA (Multi-Scale Separate)  -> GPU 2

cd ~/autodl-tmp/PointGST-main_pure/completion/

# Experiment 1: Baseline
CUDA_VISIBLE_DEVICES=0 python main.py \
    --config cfgs/PCN_models/AdaPoinTr_pgst_baseline.yaml \
    --exp_name exp1_baseline \
    --model pgst \
    --val_freq 5 \
    --num_workers 8 \
    > log_exp1_baseline.txt 2>&1 &

echo "Exp1 (Baseline) started on GPU 0, PID: $!"

# Experiment 2: +SAMA
CUDA_VISIBLE_DEVICES=1 python main.py \
    --config cfgs/PCN_models/AdaPoinTr_pgst_sama.yaml \
    --exp_name exp2_sama \
    --model pgst \
    --val_freq 5 \
    --num_workers 8 \
    > log_exp2_sama.txt 2>&1 &

echo "Exp2 (+SAMA) started on GPU 1, PID: $!"

# Experiment 3: +MSSA
CUDA_VISIBLE_DEVICES=2 python main.py \
    --config cfgs/PCN_models/AdaPoinTr_pgst_mssa.yaml \
    --exp_name exp3_mssa \
    --model pgst \
    --val_freq 5 \
    --num_workers 8 \
    > log_exp3_mssa.txt 2>&1 &

echo "Exp3 (+MSSA) started on GPU 2, PID: $!"

echo ""
echo "=========================================="
echo "All three experiments started!"
echo "=========================================="
echo "Logs:"
echo "  - log_exp1_baseline.txt (GPU 0)"
echo "  - log_exp2_sama.txt     (GPU 1)"
echo "  - log_exp3_mssa.txt     (GPU 2)"
echo ""
echo "Monitor:"
echo "  watch -n 1 nvidia-smi"
echo "  tail -f log_exp1_baseline.txt"
echo ""
echo "Estimated time: ~10-12 hours each (50 epochs)"
