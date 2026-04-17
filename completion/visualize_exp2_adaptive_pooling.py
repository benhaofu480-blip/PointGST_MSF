#!/usr/bin/env python
"""
Adaptive Pooling (exp2) 可视化：固定 5 个类别各 1 个样本，
展示 Partial / GT / 预测 / 骨架(coarse) 及叠加。
固定类别：airplane, chair, table, lamp, cabinet（基于 seed=20260411 扩展）

用法（在 completion 目录下）:
  CUDA_VISIBLE_DEVICES=0 /path/to/pgst/python visualize_exp2_adaptive_pooling.py
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from easydict import EasyDict
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from datasets.build import build_dataset_from_cfg
from extensions.chamfer_dist import ChamferDistanceL1
from tools import builder
from utils.config import cfg_from_yaml_file

SYNSET_TO_NAME = {
    "04256520": "sofa",
    "03001627": "chair",
    "02958343": "car",
    "04530566": "watercraft",
    "04379243": "table",
    "02691156": "airplane",
    "02933112": "cabinet",
    "03636649": "lamp",
}

# 固定 5 类（与之前 seed=20260411 实验一致，新增 cabinet）
FIXED_CATEGORIES = ['airplane', 'chair', 'table', 'lamp', 'cabinet']
FIXED_SEED = 20260411


def _axis_limits_from_points(*arrays):
    pts = np.concatenate([a for a in arrays if a is not None and len(a)], axis=0)
    max_range = (
        np.array([
            pts[:, 0].max() - pts[:, 0].min(),
            pts[:, 1].max() - pts[:, 1].min(),
            pts[:, 2].max() - pts[:, 2].min(),
        ]).max() / 2.0
    )
    mid = np.array([pts[:, 0].mean(), pts[:, 1].mean(), pts[:, 2].mean()])
    return mid, max_range


def plot_point_cloud(points, ax, title, color="blue", s=0.5, alpha=0.55):
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, s=s, alpha=alpha)
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    ax.view_init(elev=20, azim=45)
    mid, max_range = _axis_limits_from_points(points)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)


def plot_overlay_dense_and_coarse(dense_pts, dense_color, coarse_pts, coarse_color, ax, title):
    ax.scatter(dense_pts[:, 0], dense_pts[:, 1], dense_pts[:, 2],
               c=dense_color, s=0.25, alpha=0.35, label="dense")
    ax.scatter(coarse_pts[:, 0], coarse_pts[:, 1], coarse_pts[:, 2],
               c=coarse_color, s=10, alpha=0.95, label="skeleton")
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    ax.view_init(elev=20, azim=45)
    mid, max_range = _axis_limits_from_points(dense_pts, coarse_pts)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)


def save_xyz(path, pts):
    np.savetxt(path, pts, fmt="%.6f", delimiter=" ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="cfgs/PCN_models/AdaPoinTr_core_new.yaml")
    parser.add_argument("--checkpoint", type=str,
                        default="./experiments/AdaPoinTr_core_new/PCN_models/exp2_adaptive_pooling/ckpt-best.pth")
    parser.add_argument("--out_dir", type=str,
                        default="./visualize_output/exp2_adaptive_pooling_vis")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    config = cfg_from_yaml_file(args.config)
    config.model.NAME = "AdaPoinTr_PGST"

    model = builder.build_model_from_cfg(config.model)
    builder.load_model(model, args.checkpoint)
    model.cuda()
    model.eval()

    test_cfg = EasyDict()
    if "_base_" in config.dataset.test:
        for k, v in config.dataset.test["_base_"].items():
            test_cfg[k] = v
    if "others" in config.dataset.test:
        for k, v in config.dataset.test["others"].items():
            test_cfg[k] = v
    test_cfg["NAME"] = test_cfg.get("NAME", "PCN")
    test_cfg["subset"] = test_cfg.get("subset", "test")

    val_dataset = build_dataset_from_cfg(test_cfg)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    chamfer_fn = ChamferDistanceL1()
    num_coarse = int(config.model.num_query)

    need = {c: 1 for c in FIXED_CATEGORIES}
    done = {c: 0 for c in FIXED_CATEGORIES}

    meta_path = os.path.join(args.out_dir, "sample_meta.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"seed={FIXED_SEED}\n")
        f.write(f"fixed_categories={FIXED_CATEGORIES}\n")
        f.write(f"checkpoint={os.path.abspath(args.checkpoint)}\n")

    with torch.no_grad():
        for _, (taxonomy_id, model_id, data) in enumerate(val_loader):
            if all(done[c] >= need[c] for c in FIXED_CATEGORIES):
                break

            tax = SYNSET_TO_NAME.get(taxonomy_id[0], taxonomy_id[0])
            if tax not in need or done[tax] >= need[tax]:
                continue

            partial, gt = data
            partial = partial.cuda()
            gt = gt.cuda()
            ret = model(partial)
            coarse_t, fine_t = ret[0], ret[1]
            cd_l1 = chamfer_fn(fine_t, gt).item() * 1000

            partial_np = partial[0].cpu().numpy()
            gt_np = gt[0].cpu().numpy()
            fine_np = fine_t[0].cpu().numpy()
            coarse_np = coarse_t[0].cpu().numpy()

            si = done[tax]
            mid_short = str(model_id[0])[:12] if model_id is not None else str(si)
            prefix = os.path.join(args.out_dir, f"{tax}_{si}_{mid_short}")

            save_xyz(f"{prefix}_partial.xyz", partial_np)
            save_xyz(f"{prefix}_gt.xyz", gt_np)
            save_xyz(f"{prefix}_fine.xyz", fine_np)
            save_xyz(f"{prefix}_coarse.xyz", coarse_np)

            # Panel 图: Partial / GT / Pred / Skeleton / Overlay
            fig = plt.figure(figsize=(22, 4.2))
            ax1 = fig.add_subplot(151, projection="3d")
            plot_point_cloud(partial_np, ax1, "Partial", color="gray", s=0.35)
            ax2 = fig.add_subplot(152, projection="3d")
            plot_point_cloud(gt_np, ax2, "Ground Truth", color="green", s=0.3)
            ax3 = fig.add_subplot(153, projection="3d")
            plot_point_cloud(fine_np, ax3, f"Prediction\nCD-L1={cd_l1:.2f}",
                             color="royalblue", s=0.3)
            ax4 = fig.add_subplot(154, projection="3d")
            plot_point_cloud(coarse_np, ax4,
                             f"Skeleton (coarse)\n{num_coarse} pts", color="darkorange", s=8)
            ax5 = fig.add_subplot(155, projection="3d")
            plot_overlay_dense_and_coarse(fine_np, "royalblue", coarse_np,
                                          "darkorange", ax5, "Pred + skeleton")

            plt.suptitle(
                f"{tax.upper()}  sample#{si}  (exp2 Adaptive Pooling ckpt-best)",
                fontsize=12, fontweight="bold")
            plt.tight_layout()
            plt.savefig(f"{prefix}_panel.png", dpi=160, bbox_inches="tight")
            plt.close()

            # GT vs Pred 叠加图
            fig2 = plt.figure(figsize=(14, 5))
            bx1 = fig2.add_subplot(121, projection="3d")
            plot_overlay_dense_and_coarse(gt_np, "green", coarse_np, "darkorange",
                                         bx1, "GT + skeleton")
            bx2 = fig2.add_subplot(122, projection="3d")
            plot_overlay_dense_and_coarse(fine_np, "royalblue", coarse_np, "darkorange",
                                         bx2, f"Prediction + skeleton (CD-L1={cd_l1:.2f})")
            plt.suptitle(f"{tax.upper()} — GT vs Pred（均叠骨架）",
                         fontsize=13, fontweight="bold")
            plt.tight_layout()
            plt.savefig(f"{prefix}_gt_vs_pred_overlay.png", dpi=160, bbox_inches="tight")
            plt.close()

            with open(meta_path, "a", encoding="utf-8") as f:
                f.write(f"{tax}\t{si}\t{model_id[0]}\tCD_L1_mm={cd_l1:.4f}\t{prefix}\n")

            print(f"[{tax} #{si}] CD-L1={cd_l1:.2f} -> {prefix}_*.png")
            done[tax] += 1

    print(f"\n完成。输出目录: {os.path.abspath(args.out_dir)}")
    print(f"固定类别: {FIXED_CATEGORIES}")


if __name__ == "__main__":
    main()
