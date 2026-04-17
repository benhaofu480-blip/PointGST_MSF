#!/usr/bin/env python
"""
exp_abs_coord_hole_decoupled 可视化：4 个类别各 1 个样本，
展示 Partial / GT / 预测 / Local骨架 / Global骨架 / GT+双骨架 / 预测+双骨架。

local (384) 和 global (128) 骨架用不同颜色区分。
"""
import argparse
import os
import random
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
PCN_CATEGORIES = list(sorted(set(SYNSET_TO_NAME.values())))

NUM_LOCAL = 384  # local points
NUM_GLOBAL = 128  # global points


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
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()
    ax.view_init(elev=20, azim=45)
    mid, max_range = _axis_limits_from_points(points)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)


def plot_overlay_triple(dense_pts, dense_color, local_pts, global_pts, ax, title):
    """Overlay dense + local (cyan) + global (red) skeleton."""
    ax.scatter(dense_pts[:, 0], dense_pts[:, 1], dense_pts[:, 2],
               c=dense_color, s=0.25, alpha=0.3, label="dense")
    ax.scatter(local_pts[:, 0], local_pts[:, 1], local_pts[:, 2],
               c="cyan", s=6, alpha=0.9, label=f"local({NUM_LOCAL})")
    ax.scatter(global_pts[:, 0], global_pts[:, 1], global_pts[:, 2],
               c="red", s=10, alpha=0.95, label=f"global({NUM_GLOBAL})")
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()
    ax.view_init(elev=20, azim=45)
    mid, max_range = _axis_limits_from_points(dense_pts, local_pts, global_pts)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)


def plot_skeleton_split(local_pts, global_pts, ax, title):
    """Show local (cyan) and global (red) skeletons side by side in one axes."""
    ax.scatter(local_pts[:, 0], local_pts[:, 1], local_pts[:, 2],
               c="cyan", s=6, alpha=0.85, label=f"local({NUM_LOCAL})")
    ax.scatter(global_pts[:, 0], global_pts[:, 1], global_pts[:, 2],
               c="red", s=12, alpha=0.95, label=f"global({NUM_GLOBAL})")
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()
    ax.view_init(elev=20, azim=45)
    mid, max_range = _axis_limits_from_points(local_pts, global_pts)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)


def save_xyz(path, pts):
    np.savetxt(path, pts, fmt="%.6f", delimiter=" ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default="cfgs/PCN_models/pertoken_core_gpu1_ratio384.yaml")
    parser.add_argument("--checkpoint", type=str,
                        default="./experiments/pertoken_core_gpu1_ratio384/PCN_models/exp_abs_coord_hole_decoupled/ckpt-best.pth")
    parser.add_argument("--out_dir", type=str,
                        default="./visualize_output/exp_abs_coord_hole_decoupled")
    parser.add_argument("--seed", type=int, default=20260411)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rng = random.Random(args.seed)
    chosen = PCN_CATEGORIES.copy()
    rng.shuffle(chosen)
    target_list = chosen[:4]
    need = {c: 1 for c in target_list}
    done = {c: 0 for c in target_list}

    meta_path = os.path.join(args.out_dir, "sample_meta.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"seed={args.seed}\n")
        f.write(f"chosen_categories={target_list}\n")
        f.write(f"checkpoint={os.path.abspath(args.checkpoint)}\n")

    config = cfg_from_yaml_file(args.config)
    config.model.NAME = "AdaPoinTr_PGST"

    model = builder.model_builder(config.model)
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

    with torch.no_grad():
        for _, (taxonomy_id, model_id, data) in enumerate(val_loader):
            if all(done[c] >= need[c] for c in target_list):
                break

            tax = SYNSET_TO_NAME.get(taxonomy_id[0], taxonomy_id[0])
            if tax not in need or done[tax] >= need[tax]:
                continue

            partial, gt = data
            partial = partial.cuda()
            gt = gt.cuda()
            ret = model(partial)
            # eval mode: ret = (coarse_point_cloud, rebuild_points)
            coarse_t, fine_t = ret[0], ret[1]
            cd_l1 = chamfer_fn(fine_t, gt).item() * 1000

            coarse_np = coarse_t[0].cpu().numpy()
            local_np = coarse_np[:NUM_LOCAL]   # first 384 = local
            global_np = coarse_np[NUM_LOCAL:]   # last 128 = global
            partial_np = partial[0].cpu().numpy()
            gt_np = gt[0].cpu().numpy()
            fine_np = fine_t[0].cpu().numpy()

            si = done[tax]
            mid_short = str(model_id[0])[:12] if model_id is not None else str(si)
            prefix = os.path.join(args.out_dir, f"{tax}_{si}_{mid_short}")

            save_xyz(f"{prefix}_partial.xyz", partial_np)
            save_xyz(f"{prefix}_gt.xyz", gt_np)
            save_xyz(f"{prefix}_fine.xyz", fine_np)
            save_xyz(f"{prefix}_local.xyz", local_np)
            save_xyz(f"{prefix}_global.xyz", global_np)

            # --- Panel: 7 subplots ---
            fig = plt.figure(figsize=(30, 4.5))
            # 1) Partial
            ax1 = fig.add_subplot(171, projection="3d")
            plot_point_cloud(partial_np, ax1, "Partial", color="gray", s=0.35)
            # 2) GT
            ax2 = fig.add_subplot(172, projection="3d")
            plot_point_cloud(gt_np, ax2, "Ground Truth", color="green", s=0.3)
            # 3) Prediction
            ax3 = fig.add_subplot(173, projection="3d")
            plot_point_cloud(fine_np, ax3, f"Prediction\nCD-L1={cd_l1:.2f}",
                             color="royalblue", s=0.3)
            # 4) Local skeleton only (384)
            ax4 = fig.add_subplot(174, projection="3d")
            plot_point_cloud(local_np, ax4, f"Local skeleton\n{NUM_LOCAL} pts",
                             color="cyan", s=6, alpha=0.85)
            # 5) Global skeleton only (128)
            ax5 = fig.add_subplot(175, projection="3d")
            plot_point_cloud(global_np, ax5, f"Global skeleton\n{NUM_GLOBAL} pts",
                             color="red", s=12, alpha=0.95)
            # 6) GT + both skeletons
            ax6 = fig.add_subplot(176, projection="3d")
            plot_overlay_triple(gt_np, "green", local_np, global_np, ax6,
                                "GT + local(cyan)\n+ global(red)")
            # 7) Pred + both skeletons
            ax7 = fig.add_subplot(177, projection="3d")
            plot_overlay_triple(fine_np, "royalblue", local_np, global_np, ax7,
                                f"Pred + local(cyan)\n+ global(red)")

            plt.suptitle(
                f"{tax.upper()}  sample#{si}  (exp_abs_coord_hole_decoupled ckpt-best)",
                fontsize=12, fontweight="bold")
            plt.tight_layout()
            plt.savefig(f"{prefix}_panel.png", dpi=160, bbox_inches="tight")
            plt.close()

            # --- Second figure: GT+split vs Pred+split (larger) ---
            fig2 = plt.figure(figsize=(20, 5.5))
            bx1 = fig2.add_subplot(131, projection="3d")
            plot_overlay_triple(gt_np, "green", local_np, global_np, bx1,
                                "GT + local(cyan) + global(red)")
            bx2 = fig2.add_subplot(132, projection="3d")
            plot_overlay_triple(fine_np, "royalblue", local_np, global_np, bx2,
                                f"Pred + local(cyan) + global(red)\nCD-L1={cd_l1:.2f}")
            bx3 = fig2.add_subplot(133, projection="3d")
            plot_skeleton_split(local_np, global_np, bx3,
                                "Skeleton split\nlocal=cyan, global=red")
            plt.suptitle(
                f"{tax.upper()} — GT vs Pred (skeleton decoupled)",
                fontsize=13, fontweight="bold")
            plt.tight_layout()
            plt.savefig(f"{prefix}_gt_vs_pred_split.png", dpi=160, bbox_inches="tight")
            plt.close()

            with open(meta_path, "a", encoding="utf-8") as f:
                f.write(f"{tax}\t{si}\t{model_id[0]}\tCD_L1_mm={cd_l1:.4f}\t{prefix}\n")

            print(f"[{tax} #{si}] CD-L1={cd_l1:.2f} -> {prefix}_*.png")
            done[tax] += 1

    print(f"\n完成。输出目录: {os.path.abspath(args.out_dir)}")
    print(f"随机类别 (seed={args.seed}): {target_list}")


if __name__ == "__main__":
    main()
