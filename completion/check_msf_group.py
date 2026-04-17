import torch, sys
sys.path.insert(0, '.')
from utils.config import cfg_from_yaml_file
from tools import builder

config = cfg_from_yaml_file('cfgs/PCN_models/AdaPoinTr_core_new.yaml')
config.model.NAME = 'AdaPoinTr_PGST'
model = builder.build_model_from_cfg(config.model)
dp = torch.nn.DataParallel(model)

print('='*70)
print('[CHECK 1] MSF-related parameters: are they trainable?')
print('='*70)

msf_keywords = ['msf', 'MSF', 'spectral', 'freq', 'fft', 'scale_gate', 'gft_adapter']
for name, param in dp.module.named_parameters():
    if any(k in name.lower() for k in ['msf', 'spectral', 'freq', 'fft']):
        is_pertoken = ('coarse_pred' in name) or ('mlp_query' in name) or \
                      ('query_ranking' in name) or ('global_coarse_pred' in name) or \
                      ('global_token_pe' in name)
        is_whitelist = ('adapt' in name) or ('Adapter' in name) or ('cls' in name) or \
                       ('head' in name) or ('FacT' in name) or ('tfts' in name) or \
                       ('decoder' in name) or ('gft_adapter' in name) or is_pertoken
        
        if is_whitelist:
            status = 'TRAINABLE'
            group = 'PERTOKEN' if is_pertoken else 'normal'
        else:
            status = '*** FROZEN ***'
            group = '--'
        
        print(f'{status} | {group:>8s} | {name}')

print()
print('='*70)
print('[CHECK 2] gft_adapter parameters (the existing pertoken adapters):')
print('='*70)
for name, param in dp.module.named_parameters():
    if 'gft_adapter' in name:
        print(f'TRAINABLE | PERTOKEN | {name}')

print()
print('='*70)
print('[CHECK 3] gate_net parameters:')
print('='*70)
for name, param in dp.module.named_parameters():
    if 'gate_net' in name:
        is_pertoken = ('coarse_pred' in name) or ('mlp_query' in name) or \
                      ('query_ranking' in name) or ('global_coarse_pred' in name) or \
                      ('global_token_pe' in name) or ('gate_net' in name)  # 假设修复后
        is_whitelist = ('adapt' in name) or ('Adapter' in name) or ('cls' in name) or \
                       ('head' in name) or ('FacT' in name) or ('tfts' in name) or \
                       ('decoder' in name) or ('gft_adapter' in name) or is_pertoken
        if is_whitelist:
            status = 'TRAINABLE (after fix)'
        else:
            status = 'FROZEN (current)'
        print(f'{status}          | {name}')
