# SWAP优化：num_workers=0 消除多进程竞争，降低CPU和SWAP压力
bash ./scripts/train.sh 0 \
    --config cfgs/PCN_models/AdaPoinTr_pgst.yaml \
    --exp_name pgst \
    --start_ckpts ckpts/AdaPoinTr_ps55.pth \
    --model pgst \
    --num_workers 0