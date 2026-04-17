"""
对比两个 checkpoint 的生成效果（exp4 vs exp5）
用法: python visualize_compare_two.py --ckpt1 ckpt1.pth --ckpt2 ckpt2.pth --config xxx.yaml
"""
import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from tools import builder
from utils.config import cfg_from_yaml_file
from datasets.build import build_dataset_from_cfg
from easydict import EasyDict
from torch.utils.data import DataLoader
from extensions.chamfer_dist import ChamferDistanceL1


def plot_point_cloud(points, ax, title, color='blue', s=0.5):
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, s=s, alpha=0.6)
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    ax.view_init(elev=20, azim=45)
    max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                          points[:, 1].max() - points[:, 1].min(),
                          points[:, 2].max() - points[:, 2].min()]).max() / 2.0
    mid = np.array([points[:, 0].mean(), points[:, 1].mean(), points[:, 2].mean()])
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)


def load_model(config, checkpoint):
    config_copy = cfg_from_yaml_file(config)
    config_copy.model.NAME = 'AdaPoinTr_PGST'
    model = builder.model_builder(config_copy.model)
    builder.load_model(model, checkpoint)
    model.cuda()
    model.eval()
    return model


def main(args):
    out_dir = './visualize_compare'
    os.makedirs(out_dir, exist_ok=True)

    print("Loading model 1 (exp4 baseline pertoken)...")
    model1 = load_model(args.config, args.ckpt1)
    print("Loading model 2 (exp5 diff lr)...")
    model2 = load_model(args.config, args.ckpt2)

    config = cfg_from_yaml_file(args.config)
    test_cfg = EasyDict()
    if '_base_' in config.dataset.test:
        for key, val in config.dataset.test['_base_'].items():
            test_cfg[key] = val
    if 'others' in config.dataset.test:
        for key, val in config.dataset.test['others'].items():
            test_cfg[key] = val
    test_cfg['NAME'] = test_cfg.get('NAME', 'PCN')
    test_cfg['subset'] = test_cfg.get('subset', 'test')

    val_dataset = build_dataset_from_cfg(test_cfg)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    synset_to_name = {
        '04256520': 'sofa', '03001627': 'chair', '02958343': 'car',
        '04530566': 'watercraft', '04379243': 'table', '02691156': 'airplane',
        '02933112': 'cabinet', '03636649': 'lamp'
    }

    chamfer_fn = ChamferDistanceL1()
    collected = {}

    with torch.no_grad():
        for idx, (taxonomy_id, model_id, data) in enumerate(val_loader):
            taxonomy_name = synset_to_name.get(taxonomy_id[0], taxonomy_id[0])
            count = collected.get(taxonomy_name, 0)
            if count >= args.num_samples:
                continue

            partial, gt = data
            partial = partial.cuda()
            gt = gt.cuda()

            ret1 = model1(partial)
            coarse1 = ret1[0][0].cpu().numpy()
            fine1 = ret1[1][0].cpu().numpy()
            cd1 = chamfer_fn(ret1[1], gt).item() * 1000

            ret2 = model2(partial)
            coarse2 = ret2[0][0].cpu().numpy()
            fine2 = ret2[1][0].cpu().numpy()
            cd2 = chamfer_fn(ret2[1], gt).item() * 1000

            partial_np = partial[0].cpu().numpy()
            gt_np = gt[0].cpu().numpy()

            prefix = f"{out_dir}/{taxonomy_name}_{count}"

            # 5-col comparison: partial | GT | exp4 coarse | exp4 fine | exp5 coarse | exp5 fine
            fig = plt.figure(figsize=(24, 8))

            ax0 = fig.add_subplot(151, projection='3d')
            plot_point_cloud(partial_np, ax0, 'Partial Input', color='gray', s=0.2)

            ax1 = fig.add_subplot(152, projection='3d')
            plot_point_cloud(gt_np, ax1, 'Ground Truth', color='green', s=0.2)

            ax2 = fig.add_subplot(153, projection='3d')
            plot_point_cloud(coarse1, ax2, f'exp4 coarse\nCD={cd1:.2f}', color='orange', s=6)

            ax3 = fig.add_subplot(154, projection='3d')
            plot_point_cloud(fine1, ax3, f'exp4 fine\nCD={cd1:.2f}', color='blue', s=0.2)

            ax4 = fig.add_subplot(155, projection='3d')
            plot_point_cloud(fine2, ax4, f'exp5 fine (5x lr)\nCD={cd2:.2f}', color='red', s=0.2)

            plt.suptitle(f'{taxonomy_name.upper()} #{count}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{prefix}_compare.png", dpi=150, bbox_inches='tight')
            plt.close()

            # Coarse overlay comparison
            fig2 = plt.figure(figsize=(15, 5))
            ax5 = fig2.add_subplot(131, projection='3d')
            plot_point_cloud(gt_np, ax5, 'Ground Truth', color='green', s=0.2)
            ax6 = fig2.add_subplot(132, projection='3d')
            plot_point_cloud(coarse1, ax6, f'exp4 coarse (512pts)', color='orange', s=6)
            ax7 = fig2.add_subplot(133, projection='3d')
            plot_point_cloud(coarse2, ax7, f'exp5 coarse (512pts)', color='red', s=6)
            plt.suptitle(f'{taxonomy_name.upper()} #{count} - Coarse Comparison', fontsize=14)
            plt.tight_layout()
            plt.savefig(f"{prefix}_coarse_cmp.png", dpi=150, bbox_inches='tight')
            plt.close()

            print(f"[{taxonomy_name} #{count}] exp4 CD={cd1:.2f} | exp5 CD={cd2:.2f} | delta={cd2-cd1:+.2f}")
            collected[taxonomy_name] = count + 1

    print(f"\nDone! Output in {out_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt1', type=str, required=True, help='exp4 ckpt-best')
    parser.add_argument('--ckpt2', type=str, required=True, help='exp5 ckpt-best')
    parser.add_argument('--num_samples', type=int, default=2)
    args = parser.parse_args()
    main(args)
