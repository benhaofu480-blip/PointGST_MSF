import torch
ckpt = torch.load('ckpts/AdaPoinTr_ps55.pth', map_location='cpu')
if 'model' in ckpt:
    sd = ckpt['model']
elif 'base_model' in ckpt:
    sd = ckpt['base_model']
else:
    sd = ckpt

total = 0
for k, v in sd.items():
    n = v.numel()
    total += n
    print(f'{k}: {n} ({n/1e3:.1f}K)')

print(f'\nCheckpoint total params: {total:,} = {total/1e6:.3f}M')
print(f'Checkpoint keys: {len(sd)}')
