#!/usr/bin/env python
"""
exp_unshuffle_grouped_v3_full_pc 可视化脚本
展示: Partial / GT / 预测 / FPS采样点 / Local骨架(384) / Global骨架(128) / 混合骨架

用法:
  python visualize_exp_skeletons_full_pc.py
  python visualize_exp_skeletons_full_pc.py --seed 42 --num_samples 4
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


def plot_skeleton_comparison(coor, coarse_local, coarse_global, coarse_mixed, ax, title):
    """在同一张图上展示 FPS点、Local骨架、Global骨架、混合骨架"""
    # FPS 采样点 (灰色，小点)
    ax.scatter(coor[:, 0], coor[:, 1], coor[:, 2],
               c='lightgray', s=3, alpha=0.4, label='FPS (384)')
    # Local 骨架 (青色，Flow Matching 偏移后)
    ax.scatter(coarse_local[:, 0], coarse_local[:, 1], coarse_local[:, 2],
               c='cyan', s=12, alpha=0.85, label='Local (384)')
    # Global 骨架 (洋红色，Global头预测)
    ax.scatter(coarse_global[:, 0], coarse_global[:, 1], coarse_global[:, 2],
               c='magenta', s=18, alpha=0.9, label='Global (128)')
    # 混合骨架 (橙色，排序后)
    ax.scatter(coarse_mixed[:, 0], coarse_mixed[:, 1], coarse_mixed[:, 2],
               c='darkorange', s=6, alpha=0.7, label='Mixed (512)')
    
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()
    ax.view_init(elev=20, azim=45)
    mid, max_range = _axis_limits_from_points(coor, coarse_local, coarse_global, coarse_mixed)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    ax.legend(fontsize=6, loc='upper left')


def plot_gt_vs_pred_with_skeletons(gt, pred, coarse_mixed, ax_gt, ax_pred, cd_l1):
    """展示GT和预测结果，都叠加骨架"""
    # GT + 骨架
    ax_gt.scatter(gt[:, 0], gt[:, 1], gt[:, 2], c='green', s=0.3, alpha=0.4, label='GT')
    ax_gt.scatter(coarse_mixed[:, 0], coarse_mixed[:, 1], coarse_mixed[:, 2],
                  c='darkorange', s=8, alpha=0.9, label='Skeleton')
    ax_gt.set_title("GT + Skeleton", fontsize=10)
    ax_gt.set_axis_off()
    ax_gt.view_init(elev=20, azim=45)
    ax_gt.legend(fontsize=7, loc='upper left')
    
    # Pred + 骨架
    ax_pred.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c='royalblue', s=0.3, alpha=0.4, label='Pred')
    ax_pred.scatter(coarse_mixed[:, 0], coarse_mixed[:, 1], coarse_mixed[:, 2],
                    c='darkorange', s=8, alpha=0.9, label='Skeleton')
    ax_pred.set_title(f"Pred + Skeleton\nCD-L1={cd_l1:.2f}", fontsize=10)
    ax_pred.set_axis_off()
    ax_pred.view_init(elev=20, azim=45)
    ax_pred.legend(fontsize=7, loc='upper left')
    
    # 统一坐标范围
    mid, max_range = _axis_limits_from_points(gt, pred, coarse_mixed)
    for ax in [ax_gt, ax_pred]:
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)


def save_xyz(path, pts):
    np.savetxt(path, pts, fmt="%.6f", delimiter=" ")


class SkeletonExtractor:
    """包装模型以提取 Local 和 Global 骨架"""
    def __init__(self, model):
        self.model = model
        self.base_model = model.base_model if hasattr(model, 'base_model') else model
        self._hook_handles = []
        self.intermediate_results = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """注册前向钩子来捕获中间结果"""
        def make_hook(name):
            def hook(module, input, output):
                self.intermediate_results[name] = output
            return hook
        
        # 尝试找到关键模块并注册钩子
        if hasattr(self.base_model, 'grouper'):
            self._hook_handles.append(
                self.base_model.grouper.register_forward_hook(make_hook('grouper'))
            )
    
    def remove_hooks(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
    
    @torch.no_grad()
    def extract_skeletons(self, xyz):
        """提取所有骨架点"""
        bs = xyz.size(0)
        device = xyz.device
        
        # 获取 grouper 输出 (FPS 采样点和特征)
        if hasattr(self.base_model, 'grouper'):
            coor, f = self.base_model.grouper(xyz, self.base_model.center_num)
        else:
            raise ValueError("Cannot find grouper module")
        
        pe = self.base_model.pos_embed(coor)
        x = self.base_model.input_proj(f)
        
        B, G, _ = coor.shape
        c = coor * 100
        
        # 使用模型内部函数
        from models.PGST import xyz2key, get_basis, sort
        key = xyz2key(c[:, :, 1], c[:, :, 0], c[:, :, 2])
        _, idx0 = torch.sort(key)
        sub_center = sort(coor, idx0)
        sub_U0, _ = get_basis(sub_center.reshape(B * (G // 16), 16, 3))
        sub_U0 = sub_U0.reshape(B, G // 16, 16, 16)
        sub_U1, _ = get_basis(sub_center.reshape(B * (G // 32), 32, 3))
        sub_U1 = sub_U1.reshape(B, G // 32, 32, 32)
        
        x = self.base_model.encoder(x + pe, coor, [sub_U0, sub_U1], [idx0, idx0])
        token_features = self.base_model.increase_dim(x)
        global_feature = torch.max(token_features, dim=1)[0]
        
        # 提取 Local 骨架 (Flow Matching)
        fm_cond = torch.cat([token_features, coor], dim=-1)
        if hasattr(self.base_model.flow_offset, 'sample'):
            coarse_local = self.base_model.flow_offset.sample(coor, fm_cond, num_steps=10)
        else:
            coarse_local = coor  # fallback
        
        # 提取 Global 骨架
        coarse_global = self.base_model.global_coarse_pred(global_feature).reshape(
            B, self.base_model.num_global_points, 3)
        
        # 混合骨架 (未排序)
        coarse_mixed = torch.cat([coarse_local, coarse_global], dim=1)
        
        return {
            'coor': coor[0].cpu().numpy(),  # FPS 采样点 (384点)
            'coarse_local': coarse_local[0].cpu().numpy(),  # Local 骨架 (384点)
            'coarse_global': coarse_global[0].cpu().numpy(),  # Global 骨架 (128点)
            'coarse_mixed': coarse_mixed[0].cpu().numpy(),  # 混合骨架 (512点)
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="cfgs/PCN_models/pertoken_core_gpu1_ratio384_full.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./experiments/pertoken_core_gpu1_ratio384_full/PCN_models/exp_unshuffle_grouped_v3_full_pc/ckpt-best.pth",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./visualize_output/exp_unshuffle_grouped_v3_full_pc_skeletons",
    )
    parser.add_argument("--seed", type=int, default=20260407)
    parser.add_argument("--num_samples", type=int, default=4, help="随机选择多少个样本")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)
    
    config = cfg_from_yaml_file(args.config)
    config.model.NAME = "AdaPoinTr_PGST"

    model = builder.model_builder(config.model)
    builder.load_model(model, args.checkpoint)
    model.cuda()
    model.eval()

    # 创建骨架提取器
    skeleton_extractor = SkeletonExtractor(model)

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

    # 收集所有样本索引
    all_samples = list(enumerate(val_loader))
    rng.shuffle(all_samples)
    selected_samples = all_samples[:args.num_samples]

    meta_path = os.path.join(args.out_dir, "sample_meta.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"seed={args.seed}\n")
        f.write(f"num_samples={args.num_samples}\n")
        f.write(f"checkpoint={os.path.abspath(args.checkpoint)}\n")
        f.write(f"config={os.path.abspath(args.config)}\n\n")

    with torch.no_grad():
        for idx, (batch_idx, (taxonomy_id, model_id, data)) in enumerate(selected_samples):
            tax = SYNSET_TO_NAME.get(taxonomy_id[0], taxonomy_id[0])

            partial, gt = data
            partial = partial.cuda()
            gt = gt.cuda()
            
            # 获取模型预测
            ret = model(partial)
            coarse_t, fine_t = ret[0], ret[1]
            cd_l1 = chamfer_fn(fine_t, gt).item() * 1000

            # 提取骨架点
            skeletons = skeleton_extractor.extract_skeletons(partial)
            
            partial_np = partial[0].cpu().numpy()
            gt_np = gt[0].cpu().numpy()
            fine_np = fine_t[0].cpu().numpy()
            coarse_np = coarse_t[0].cpu().numpy()
            
            coor_np = skeletons['coor']
            coarse_local_np = skeletons['coarse_local']
            coarse_global_np = skeletons['coarse_global']
            coarse_mixed_np = skeletons['coarse_mixed']

            mid_short = str(model_id[0])[:12] if model_id is not None else str(idx)
            prefix = os.path.join(args.out_dir, f"{tax}_{idx}_{mid_short}")

            # 保存点云数据
            save_xyz(f"{prefix}_partial.xyz", partial_np)
            save_xyz(f"{prefix}_gt.xyz", gt_np)
            save_xyz(f"{prefix}_fine.xyz", fine_np)
            save_xyz(f"{prefix}_coarse_final.xyz", coarse_np)
            save_xyz(f"{prefix}_coor_fps.xyz", coor_np)
            save_xyz(f"{prefix}_coarse_local.xyz", coarse_local_np)
            save_xyz(f"{prefix}_coarse_global.xyz", coarse_global_np)
            save_xyz(f"{prefix}_coarse_mixed.xyz", coarse_mixed_np)

            # ===== 图1: 完整6视图面板 =====
            fig = plt.figure(figsize=(24, 4))
            
            ax1 = fig.add_subplot(161, projection="3d")
            plot_point_cloud(partial_np, ax1, "Partial", color="gray", s=0.35)
            
            ax2 = fig.add_subplot(162, projection="3d")
            plot_point_cloud(gt_np, ax2, "Ground Truth", color="green", s=0.3)
            
            ax3 = fig.add_subplot(163, projection="3d")
            plot_point_cloud(fine_np, ax3, f"Prediction\nCD-L1={cd_l1:.2f}", color="royalblue", s=0.3)
            
            ax4 = fig.add_subplot(164, projection="3d")
            plot_point_cloud(coor_np, ax4, f"FPS Centers\n384 pts", color="lightgray", s=5)
            
            ax5 = fig.add_subplot(165, projection="3d")
            plot_skeleton_comparison(coor_np, coarse_local_np, coarse_global_np, coarse_mixed_np, ax5,
                                    "Skeleton Components\n(Local+Global+Mixed)")
            
            ax6 = fig.add_subplot(166, projection="3d")
            plot_point_cloud(coarse_np, ax6, f"Final Skeleton\n512 pts (sorted)", color="darkorange", s=8)

            plt.suptitle(
                f"{tax.upper()}  sample#{idx}  (exp_unshuffle_grouped_v3_full_pc ckpt-best)",
                fontsize=12, fontweight="bold",
            )
            plt.tight_layout()
            plt.savefig(f"{prefix}_full_panel.png", dpi=160, bbox_inches="tight")
            plt.close()

            # ===== 图2: GT vs Pred 对比（都叠加骨架） =====
            fig2 = plt.figure(figsize=(14, 5))
            ax_gt = fig2.add_subplot(121, projection="3d")
            ax_pred = fig2.add_subplot(122, projection="3d")
            plot_gt_vs_pred_with_skeletons(gt_np, fine_np, coarse_np, ax_gt, ax_pred, cd_l1)
            plt.suptitle(
                f"{tax.upper()} — GT vs Pred (both with skeletons)",
                fontsize=13, fontweight="bold",
            )
            plt.tight_layout()
            plt.savefig(f"{prefix}_gt_vs_pred.png", dpi=160, bbox_inches="tight")
            plt.close()

            # ===== 图3: 骨架点详细分解 =====
            fig3, axes = plt.subplots(2, 3, figsize=(15, 9), subplot_kw={'projection': '3d'})
            
            # FPS 采样点
            ax = axes[0, 0]
            ax.scatter(coor_np[:, 0], coor_np[:, 1], coor_np[:, 2], c='gray', s=8, alpha=0.6)
            ax.set_title("FPS Sampling Points (384)", fontsize=10)
            ax.set_axis_off()
            ax.view_init(elev=20, azim=45)
            
            # Local 骨架
            ax = axes[0, 1]
            ax.scatter(coarse_local_np[:, 0], coarse_local_np[:, 1], coarse_local_np[:, 2], 
                      c='cyan', s=12, alpha=0.85)
            ax.set_title("Local Skeleton (384)\nFlow Matching Offset", fontsize=10)
            ax.set_axis_off()
            ax.view_init(elev=20, azim=45)
            
            # Global 骨架
            ax = axes[0, 2]
            ax.scatter(coarse_global_np[:, 0], coarse_global_np[:, 1], coarse_global_np[:, 2], 
                      c='magenta', s=20, alpha=0.9)
            ax.set_title("Global Skeleton (128)\nGlobal Head Prediction", fontsize=10)
            ax.set_axis_off()
            ax.view_init(elev=20, azim=45)
            
            # Local + Global 叠加
            ax = axes[1, 0]
            ax.scatter(coarse_local_np[:, 0], coarse_local_np[:, 1], coarse_local_np[:, 2], 
                      c='cyan', s=12, alpha=0.7, label='Local (384)')
            ax.scatter(coarse_global_np[:, 0], coarse_global_np[:, 1], coarse_global_np[:, 2], 
                      c='magenta', s=20, alpha=0.9, label='Global (128)')
            ax.set_title("Local + Global (unsorted)", fontsize=10)
            ax.set_axis_off()
            ax.view_init(elev=20, azim=45)
            ax.legend(fontsize=7)
            
            # 混合骨架 (排序前)
            ax = axes[1, 1]
            ax.scatter(coarse_mixed_np[:, 0], coarse_mixed_np[:, 1], coarse_mixed_np[:, 2], 
                      c='darkorange', s=6, alpha=0.7)
            ax.set_title("Mixed Skeleton (512)\nBefore query_ranking", fontsize=10)
            ax.set_axis_off()
            ax.view_init(elev=20, azim=45)
            
            # 最终骨架 (排序后)
            ax = axes[1, 2]
            ax.scatter(coarse_np[:, 0], coarse_np[:, 1], coarse_np[:, 2], 
                      c='darkorange', s=8, alpha=0.85)
            ax.set_title("Final Skeleton (512)\nAfter query_ranking", fontsize=10)
            ax.set_axis_off()
            ax.view_init(elev=20, azim=45)
            
            # 统一坐标范围
            for ax in axes.flat:
                mid, max_range = _axis_limits_from_points(
                    coor_np, coarse_local_np, coarse_global_np, coarse_mixed_np, coarse_np)
                ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
                ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
                ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
            
            plt.suptitle(
                f"{tax.upper()} — Skeleton Decomposition\n(FPS → Local+Global → Mixed → Final)",
                fontsize=13, fontweight="bold",
            )
            plt.tight_layout()
            plt.savefig(f"{prefix}_skeleton_decomposition.png", dpi=160, bbox_inches="tight")
            plt.close()

            with open(meta_path, "a", encoding="utf-8") as f:
                f.write(f"{idx}\t{tax}\t{model_id[0]}\tCD_L1_mm={cd_l1:.4f}\t{prefix}\n")

            print(f"[{idx}] {tax} CD-L1={cd_l1:.2f}")
            print(f"     FPS: {coor_np.shape}, Local: {coarse_local_np.shape}, "
                  f"Global: {coarse_global_np.shape}, Final: {coarse_np.shape}")
            print(f"     Saved to {prefix}_*.png")

    # 清理钩子
    skeleton_extractor.remove_hooks()
    
    print(f"\n完成！输出目录: {os.path.abspath(args.out_dir)}")
    print(f"随机种子: seed={args.seed}")


if __name__ == "__main__":
    main()
