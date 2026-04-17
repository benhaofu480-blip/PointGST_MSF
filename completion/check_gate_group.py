import torch, sys
sys.path.insert(0, '.')
from utils.config import cfg_from_yaml_file
from tools import builder

config = cfg_from_yaml_file('cfgs/PCN_models/AdaPoinTr_core_new.yaml')
config.model.NAME = 'AdaPoinTr_PGST'
model = builder.build_model_from_cfg(config.model)
dp = torch.nn.DataParallel(model)

print('='*70)
print('[CRITICAL CHECK] gate_net parameter group assignment')
print('='*70)

for name, param in dp.module.named_parameters():
    if 'gate_net' not in name:
        continue

    is_pertoken = ('coarse_pred' in name) or ('mlp_query' in name) or \
                  ('query_ranking' in name) or ('global_coarse_pred' in name) or \
                  ('global_token_pe' in name) or ('gate_net' in name)
    
    is_whitelist = ('adapt' in name) or ('Adapter' in name) or ('cls' in name) or \
                   ('head' in name) or ('FacT' in name) or ('tfts' in name) or \
                   ('decoder' in name) or ('gft_adapter' in name) or is_pertoken

    if is_whitelist:
        if is_pertoken:
            group = 'PERTOKEN (lr=0.0006)'
        else:
            group = 'normal (lr=0.0002)'
        status = 'TRAINABLE'
    else:
        group = 'N/A'
        status = '*** FROZEN ***'

    print(f'{status} | {group:>20s} | {name}')
