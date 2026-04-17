#!/bin/bash

# 续训脚本 - 从指定实验目录继续训练

RESUME_PATH=${1:-"./experiments/finetune_shapenetpart_pgst/mae/shapenetpart_reproduce/20260321-224112"}

CUDA_VISIBLE_DEVICES=0 python main.py \
    --resume \
    --resume_path $RESUME_PATH \
    --config cfgs/mae/finetune_shapenetpart_pgst.yaml \
    --exp_name shapenetpart_reproduce \
    --seed 0 \
    --num_workers 4 \
    --val_freq 10
