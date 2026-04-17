"""
可视化 Lamp 和 Table 的 GT vs Fine 对比图
"""
import os
import sys
import torch
import numpy as np
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


def save_point_cloud(points, filename):
    np.savetxt(filename, points, fmt='%.6f', delimiter=' ')
    print(f"Saved: {filename}")


def plot_point_cloud(points, ax, title, color='blue', s=0.5):
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, s=s, alpha=0.6)
    ax.set_title(title, fontsize=12)
    ax.set_axis_off()
    # 统一视角
    ax.view_init(elev=20, azim=45)
    # 统一比例
    max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                          points[:, 1].max() - points[:, 1].min(),
                          points[:, 2].max() - points[:, 2].min()]).max() / 2.0
    mid = np.array([points[:, 0].mean(), points[:, 1].mean(), points[:, 2].mean()])
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)


def main(args):
    out_dir = './visualize_output'
    os.makedirs(out_dir, exist_ok=True)

    # 加载配置
    config = cfg_from_yaml_file(args.config)
    config.model.NAME = 'AdaPoinTr'

    # 加载模型
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.checkpoint)
    base_model.cuda()
    base_model.eval()

    # 加载测试集
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
        '04256520': 'sofa',
        '03001627': 'chair',
        '02958343': 'car',
        '04530566': 'watercraft',
        '04379243': 'table',
        '02691156': 'airplane',
        '02933112': 'cabinet',
        '03636649': 'lamp'
    }

    target_cats = {'lamp': 0, 'table': 0}
    max_samples = args.num_samples

    chamfer_fn = ChamferDistanceL1()

    print(f"开始收集 lamp/table 样本 (每类 {max_samples} 个)...")

    with torch.no_grad():
        for idx, (taxonomy_id, model_id, data) in enumerate(val_loader):
            if all(v >= max_samples for v in target_cats.values()):
                break

            taxonomy_name = synset_to_name.get(taxonomy_id[0], taxonomy_id[0])
            if taxonomy_name not in target_cats or target_cats[taxonomy_name] >= max_samples:
                continue

            partial, gt = data
            partial = partial.cuda()
            gt = gt.cuda()

            ret = base_model(partial)
            coarse_pred = ret[0]
            fine_pred = ret[1]

            cd_l1 = chamfer_fn(fine_pred, gt).item() * 1000

            partial_np = partial[0].cpu().numpy()
            gt_np = gt[0].cpu().numpy()
            fine_np = fine_pred[0].cpu().numpy()
            coarse_np = coarse_pred[0].cpu().numpy()

            si = target_cats[taxonomy_name]
            prefix = f"{out_dir}/{taxonomy_name}_{si}"

            save_point_cloud(gt_np, f"{prefix}_gt.xyz")
            save_point_cloud(partial_np, f"{prefix}_partial.xyz")
            save_point_cloud(fine_np, f"{prefix}_fine.xyz")
            save_point_cloud(coarse_np, f"{prefix}_coarse.xyz")

            # 绘制对比图
            fig = plt.figure(figsize=(18, 5))

            ax1 = fig.add_subplot(141, projection='3d')
            plot_point_cloud(partial_np, ax1, 'Partial Input', color='gray', s=0.3)

            ax2 = fig.add_subplot(142, projection='3d')
            plot_point_cloud(gt_np, ax2, 'Ground Truth', color='green', s=0.3)

            ax3 = fig.add_subplot(143, projection='3d')
            plot_point_cloud(fine_np, ax3, f'Prediction\nCD-L1: {cd_l1:.2f}', color='blue', s=0.3)

            ax4 = fig.add_subplot(144, projection='3d')
            plot_point_cloud(coarse_np, ax4, f'Coarse (512 pts)', color='red', s=8)

            plt.suptitle(f'{taxonomy_name.upper()} #{si}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{prefix}_compare.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {prefix}_compare.png")

            # GT vs Fine 并排对比图
            fig2 = plt.figure(figsize=(12, 5))
            ax5 = fig2.add_subplot(121, projection='3d')
            plot_point_cloud(gt_np, ax5, 'Ground Truth', color='green', s=0.3)
            ax6 = fig2.add_subplot(122, projection='3d')
            plot_point_cloud(fine_np, ax6, f'Prediction (CD-L1: {cd_l1:.2f})', color='blue', s=0.3)
            plt.suptitle(f'{taxonomy_name.upper()} #{si} - GT vs Prediction', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{out_dir}/compare_{taxonomy_name}_{si}_gt_vs_{taxonomy_name}_{si}_fine.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: compare_{taxonomy_name}_{si}_gt_vs_{taxonomy_name}_{si}_fine.png")

            print(f"  [{taxonomy_name} #{si}] CD-L1: {cd_l1:.2f}")
            target_cats[taxonomy_name] += 1

    print("\n完成！")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='cfgs/PCN_models/AdaPoinTr_pgst_mssa.yaml')
    parser.add_argument('--checkpoint', type=str,
                        default='./experiments/AdaPoinTr_pgst_mssa/PCN_models/exp4_mssa_adaptive_gate/ckpt-best.pth')
    parser.add_argument('--num_samples', type=int, default=2)
    args = parser.parse_args()
    main(args)
