#!/bin/bash
source /home/fubenhao/data/fubenhao_data/miniforge3/etc/profile.d/conda.sh
conda activate pgst

cd /home/fubenhao/data/fubenhao_data/PointGST-main_pure/completion

export CUDA_VISIBLE_DEVICES=1

python -u main.py \
    --config cfgs/PCN_models/AdaPoinTr_PCSA_baseline.yaml \
    --exp_name exp_PCSA_baseline \
    --model pcsa \
    --val_freq 10
