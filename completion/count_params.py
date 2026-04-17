import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'completion'))
import torch
import traceback

try:
    from utils.config import get_config
    from models.build import build_model_from_cfg
    import types

    # Build AdaPoinTr (no PCSA)
    args = types.SimpleNamespace()
    args.config = 'cfgs/PCN_models/AdaPoinTr.yaml'
    args.model = 'linear'
    args.exp_name = 'test'
    args.resume = False
    args.start_ckpts = None
    args.test = False
    args.launcher = 'none'
    args.local_rank = 0
    args.distributed = False
    args.use_gpu = False
    args.seed = None
    args.deterministic = False
    args.ckpts = None
    args.experiment_path = './exp_test'
    args.tfboard_path = './tfboard_test'
    args.log_name = 'AdaPoinTr'

    config = get_config(args)
    model = build_model_from_cfg(config)

    total = sum(p.numel() for p in model.parameters())
    for name, module in model.named_children():
        count = sum(p.numel() for p in module.parameters())
        print(f'{name}: {count/1e6:.3f}M ({count})')

    print(f'\nAdaPoinTr Total: {total/1e6:.3f}M ({total})')

    # Count PCSA params in PGST version
    args2 = types.SimpleNamespace()
    args2.config = 'cfgs/PCN_models/AdaPoinTr_pgst_baseline.yaml'
    args2.model = 'pgst'
    args2.exp_name = 'test'
    args2.resume = False
    args2.start_ckpts = None
    args2.test = False
    args2.launcher = 'none'
    args2.local_rank = 0
    args2.distributed = False
    args2.use_gpu = False
    args2.seed = None
    args2.deterministic = False
    args2.ckpts = None
    args2.experiment_path = './exp_test2'
    args2.tfboard_path = './tfboard_test2'
    args2.log_name = 'AdaPoinTr_pgst'

    config2 = get_config(args2)
    model2 = build_model_from_cfg(config2)

    total2 = sum(p.numel() for p in model2.parameters())
    for name, module in model2.named_children():
        count = sum(p.numel() for p in module.parameters())
        print(f'{name}: {count/1e6:.3f}M ({count})')

    print(f'\nAdaPoinTr_PGST Total: {total2/1e6:.3f}M ({total2})')
    print(f'PCSA extra: {(total2-total)/1e3:.1f}K ({total2-total})')

    # part=gft trainable params
    gft_count = 0
    for name, param in model2.named_parameters():
        if ('adapt' in name) or ('Adapter' in name) or ('cls' in name) or ('head' in name) or ('decoder' in name):
            gft_count += param.numel()
    print(f'part=gft trainable: {gft_count/1e6:.3f}M ({gft_count})')

    # PCSA-only params
    pcsa_count = 0
    for name, param in model2.named_parameters():
        if 'gft_adapter' in name:
            pcsa_count += param.numel()
    print(f'PCSA-only params: {pcsa_count/1e6:.3f}M ({pcsa_count})')

    # Base model (without PCSA) trainable with gft
    base_gft = 0
    for name, param in model.named_parameters():
        if ('adapt' in name) or ('Adapter' in name) or ('cls' in name) or ('head' in name) or ('decoder' in name):
            base_gft += param.numel()
    print(f'AdaPoinTr part=gft trainable: {base_gft/1e6:.3f}M ({base_gft})')

except Exception as e:
    traceback.print_exc()
