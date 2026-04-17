#!/usr/bin/env python
"""Quick statistical analysis of global offset quality (200 samples)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch, numpy as np
from easydict import EasyDict
from torch.utils.data import DataLoader
from datasets.build import build_dataset_from_cfg
from tools import builder
from utils.config import cfg_from_yaml_file
from extensions.chamfer_dist import ChamferDistanceL1
from pointnet2_ops import pointnet2_utils
from models.PGST import xyz2key, get_basis, sort
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

config = cfg_from_yaml_file('cfgs/PCN_models/pertoken_core_gpu1_ratio384.yaml')
config.model.NAME = 'AdaPoinTr_PGST'
model = builder.model_builder(config.model)
builder.load_model(model, './experiments/pertoken_core_gpu1_ratio384/PCN_models/exp_pertoken_global_offset/ckpt-best.pth')
model.cuda().eval()

test_cfg = EasyDict()
for k, v in config.dataset.test.get('_base_', {}).items(): test_cfg[k] = v
for k, v in config.dataset.test.get('others', {}).items(): test_cfg[k] = v
test_cfg['NAME'] = 'PCN'; test_cfg['subset'] = 'test'
val_dataset = build_dataset_from_cfg(test_cfg)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

chamfer = ChamferDistanceL1()
bm = model.base_model
num_local = bm.center_num[-1]
num_global = bm.num_global_points

all_offset_norms = []
all_offsets_per_axis = [[], [], []]
all_anchor_dists = []
all_global_dists = []
all_local_dists = []
all_cd_local = []
all_cd_global = []
global_worse_count = 0
total_count = 0
improved_points = 0
total_points = 0

MAX_SAMPLES = 200

with torch.no_grad():
    for i, (_, _, data) in enumerate(val_loader):
        if i >= MAX_SAMPLES:
            break
        partial, gt = data
        partial, gt = partial.cuda(), gt.cuda()
        ret = model(partial)
        coarse, fine = ret[0], ret[1]

        coor, f = bm.grouper(partial, bm.center_num)
        pe = bm.pos_embed(coor)
        x = bm.input_proj(f)
        B, G, _ = coor.shape
        c = coor * 100
        key = xyz2key(c[:, :, 1], c[:, :, 0], c[:, :, 2])
        _, idx0 = torch.sort(key)
        _, idx1 = torch.sort(idx0)
        sub_center = sort(coor, idx0)
        sub_U0, _ = get_basis(sub_center.reshape(B * (G // 16), 16, 3))
        sub_U0 = sub_U0.reshape(B, G // 16, 16, 16)
        sub_U1, _ = get_basis(sub_center.reshape(B * (G // 32), 32, 3))
        sub_U1 = sub_U1.reshape(B, G // 32, 32, 32)
        x = bm.encoder(x + pe, coor, [sub_U0, sub_U1], [idx0, idx1])
        token_features = bm.increase_dim(x)

        fm_cond = torch.cat([token_features, coor], dim=-1)
        coarse_local = bm.flow_offset.sample(coor, fm_cond, num_steps=10)

        sel_idx = pointnet2_utils.furthest_point_sample(coor, num_global).long()
        sel_coor = torch.gather(coor, 1, sel_idx.unsqueeze(-1).expand(-1, -1, 3))
        sel_feat = torch.gather(token_features, 1, sel_idx.unsqueeze(-1).expand(-1, -1, token_features.size(-1)))
        global_offset = bm.global_offset_net(torch.cat([sel_feat, sel_coor], dim=-1))
        coarse_global = sel_coor + global_offset

        cd_local = chamfer(coarse_local, gt).item() * 1000
        cd_global = chamfer(coarse_global, gt).item() * 1000
        all_cd_local.append(cd_local)
        all_cd_global.append(cd_global)

        if cd_global > cd_local:
            global_worse_count += 1
        total_count += 1

        offset_np = global_offset[0].cpu().numpy()
        offset_norms = np.linalg.norm(offset_np, axis=1)
        all_offset_norms.append(offset_norms)
        for axis in range(3):
            all_offsets_per_axis[axis].extend(offset_np[:, axis].tolist())

        gt_np = gt[0].cpu().numpy()
        for j in range(num_global):
            anchor = sel_coor[0, j].cpu().numpy()
            gpt = coarse_global[0, j].cpu().numpy()
            ad = np.min(np.linalg.norm(gt_np - anchor, axis=1))
            gd = np.min(np.linalg.norm(gt_np - gpt, axis=1))
            all_anchor_dists.append(ad)
            all_global_dists.append(gd)
            total_points += 1
            if gd < ad:
                improved_points += 1

        for j in range(num_local):
            lpt = coarse_local[0, j].cpu().numpy()
            all_local_dists.append(np.min(np.linalg.norm(gt_np - lpt, axis=1)))

        if (i + 1) % 50 == 0:
            print(f'Processed {i+1}/{MAX_SAMPLES}...')

all_offset_norms = np.concatenate(all_offset_norms)
all_anchor_dists = np.array(all_anchor_dists)
all_global_dists = np.array(all_global_dists)
all_local_dists = np.array(all_local_dists)
all_cd_local = np.array(all_cd_local)
all_cd_global = np.array(all_cd_global)

print(f'\n{"="*60}')
print(f'  STATISTICS: pertoken_global_offset ({MAX_SAMPLES} samples)')
print(f'{"="*60}')
print(f'\nGlobal CD > Local CD: {global_worse_count}/{total_count} ({100*global_worse_count/total_count:.1f}%)')
print(f'Mean CD_local={all_cd_local.mean():.2f}  Mean CD_global={all_cd_global.mean():.2f}  '
      f'Ratio={all_cd_global.mean()/all_cd_local.mean():.2f}x')

print(f'\n--- Global Offset Norms ---')
print(f'  mean={all_offset_norms.mean():.4f}  median={np.median(all_offset_norms):.4f}  '
      f'std={all_offset_norms.std():.4f}  max={all_offset_norms.max():.4f}  min={all_offset_norms.min():.4f}')
print(f'  percentiles: 25%={np.percentile(all_offset_norms,25):.4f}  75%={np.percentile(all_offset_norms,75):.4f}  '
      f'95%={np.percentile(all_offset_norms,95):.4f}')

print(f'\n--- NN Distance to GT Surface ---')
print(f'  Anchor : mean={all_anchor_dists.mean():.4f}  median={np.median(all_anchor_dists):.4f}')
print(f'  Global : mean={all_global_dists.mean():.4f}  median={np.median(all_global_dists):.4f}')
print(f'  Local  : mean={all_local_dists.mean():.4f}  median={np.median(all_local_dists):.4f}')
print(f'\n  Offset moved closer to GT: {improved_points}/{total_points} ({100*improved_points/total_points:.1f}%)')
print(f'  Ratio global/local: {all_global_dists.mean()/all_local_dists.mean():.2f}x')

# Per-axis stats
for i, name in enumerate(['X', 'Y', 'Z']):
    vals = np.array(all_offsets_per_axis[i])
    print(f'  Offset {name}: mean={vals.mean():.4f}  std={vals.std():.4f}  '
          f'min={vals.min():.4f}  max={vals.max():.4f}')

# ============ PLOT ============
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

ax = axes[0, 0]
ax.hist(all_offset_norms, bins=100, color='tomato', alpha=0.7, edgecolor='black', linewidth=0.3)
ax.axvline(all_offset_norms.mean(), color='red', ls='--', lw=2, label=f'mean={all_offset_norms.mean():.4f}')
ax.axvline(np.median(all_offset_norms), color='blue', ls='--', lw=2, label=f'median={np.median(all_offset_norms):.4f}')
ax.set_xlabel('Offset L2 norm'); ax.set_ylabel('Count')
ax.set_title('Global Offset Magnitude'); ax.legend()

ax = axes[0, 1]
labels = ['Anchor', 'Global', 'Local']
means = [all_anchor_dists.mean(), all_global_dists.mean(), all_local_dists.mean()]
medians = [np.median(all_anchor_dists), np.median(all_global_dists), np.median(all_local_dists)]
x_pos = np.arange(3)
bars = ax.bar(x_pos, means, 0.5, color=['yellow', 'blue', 'red'], alpha=0.7, edgecolor='black')
ax.bar(x_pos, medians, 0.3, color=['orange', 'navy', 'darkred'], alpha=0.9, edgecolor='black')
ax.set_xticks(x_pos); ax.set_xticklabels(labels)
ax.set_ylabel('Mean NN distance to GT')
ax.set_title('Distance to GT (wide=mean, narrow=median)')
for idx_v, (m, md) in enumerate(zip(means, medians)):
    ax.text(idx_v, max(m, md) + 0.002, f'{m:.4f}\n{md:.4f}', ha='center', fontsize=8)

ax = axes[0, 2]
ax.hist(all_anchor_dists, bins=80, alpha=0.5, color='yellow', edgecolor='orange',
        label=f'Anchor ({all_anchor_dists.mean():.4f})')
ax.hist(all_global_dists, bins=80, alpha=0.5, color='blue', edgecolor='navy',
        label=f'Global ({all_global_dists.mean():.4f})')
ax.hist(all_local_dists, bins=80, alpha=0.5, color='red', edgecolor='darkred',
        label=f'Local ({all_local_dists.mean():.4f})')
ax.set_xlabel('NN distance to GT'); ax.set_ylabel('Count')
ax.set_title('NN Distance Distribution'); ax.legend(fontsize=8)

ax = axes[1, 0]
ax.boxplot([all_anchor_dists, all_global_dists, all_local_dists], labels=['Anchor', 'Global', 'Local'])
ax.set_ylabel('NN distance to GT')
ax.set_title('Distance to GT (Boxplot)')

ax = axes[1, 1]
ax.hist(all_offset_norms, bins=100, color='steelblue', alpha=0.7, cumulative=True, density=True)
ax.set_xlabel('Offset L2 norm'); ax.set_ylabel('CDF')
ax.set_title('Global Offset CDF')
ax.axvline(all_offset_norms.mean(), color='red', ls='--', label=f'mean')
ax.axvline(np.median(all_offset_norms), color='blue', ls='--', label=f'median')
ax.legend()

ax = axes[1, 2]
n_plot = min(5000, len(all_anchor_dists))
ax.scatter(all_anchor_dists[:n_plot], all_global_dists[:n_plot], s=2, alpha=0.3, c='blue')
lim_max = max(all_anchor_dists.max(), all_global_dists.max())
ax.plot([0, lim_max], [0, lim_max], 'r--', alpha=0.5, label='y=x (no change)')
ax.set_xlabel('Anchor NN-dist to GT'); ax.set_ylabel('Global NN-dist to GT')
ax.set_title(f'Anchor vs Global Dist to GT\n(improved: {100*improved_points/total_points:.1f}%)')
ax.legend(fontsize=8)

plt.suptitle('pertoken_global_offset — Global Point Diagnosis (200 samples)', fontsize=14, fontweight='bold')
plt.tight_layout()
out_path = './visualize_output/pertoken_offset_diag/global_statistics.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'\nSaved: {out_path}')
