#!/usr/bin/env python
"""
Baseline点云补全可视化脚本 — 4个固定物体，展示骨架点(local/global不同颜色)

固定4个样本 (每个类别1个，来自PCN test集):
  - airplane (02691156): 2d7aff5577ae7af0d8ff6111270336a9
  - car     (02958343): e011a97bdaf8aac4bbb53e58fdcb6353
  - chair   (03001627): 49918114029ce6a63db5e7f805103dd
  - lamp    (03636649): ec0979097f7c811922a520e8315099fb

用法:
  cd completion/
  python visualize_baseline_skeleton.py
  python visualize_baseline_skeleton.py --out_dir ./visualize_output/baseline_skeleton
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
from utils import misc

# ─── 固定样本定义 ───────────────────────────────────────────
FIXED_SAMPLES = [
    {"taxonomy_id": "02691156", "model_id": "2d7aff5577ae7af0d8ff6111270336a9", "name": "airplane"},
    {"taxonomy_id": "02958343", "model_id": "e011a97bdaf8aac4bbb53e58fdcb6353", "name": "car"},
    {"taxonomy_id": "03001627", "model_id": "49918114029ce6a63db5e7f805103dd", "name": "chair"},
    {"taxonomy_id": "03636649", "model_id": "ec0979097f7c811922a520e8315099fb", "name": "lamp"},
]


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


@torch.no_grad()
def extract_skeletons_with_source(model, xyz):
    """提取骨架点并追踪来源(global_pred vs local_fps)

    Baseline PCTransformer.forward() 流程:
      coarse = coarse_pred(global_feature).reshape(B, num_query, 3)  # 512点, global
      coarse_inp = fps(xyz, num_query//2)  # 256点, local (来自输入表面)
      coarse = cat([coarse, coarse_inp], dim=1)  # 768点
      idx = argsort(query_ranking, descending=True)
      coarse = gather(coarse, idx[:,:num_query])  # 512点 (混合来源)

    我们复制这段逻辑来追踪每个选中点的来源。
    """
    base_model = model.base_model
    bs = xyz.size(0)

    # === 复制 PCTransformer.forward 的推理逻辑 ===
    coor, f = base_model.grouper(xyz, base_model.center_num)
    pe = base_model.pos_embed(coor)
    x = base_model.input_proj(f)

    B, G, _ = coor.shape
    c = coor * 100
    from models.PGST import xyz2key, get_basis, sort
    key = xyz2key(c[:, :, 1], c[:, :, 0], c[:, :, 2])
    _, idx0 = torch.sort(key)
    _, idx1 = torch.sort(idx0)
    sub_center = sort(coor, idx0)
    sub_U0 = get_basis(sub_center.reshape(B * (G // 16), 16, 3)).reshape(B, G // 16, 16, 16)
    sub_U1 = get_basis(sub_center.reshape(B * (G // 32), 32, 3)).reshape(B, G // 32, 32, 32)

    x = base_model.encoder(x + pe, coor, [sub_U0, sub_U1], [idx0, idx1])
    token_features = base_model.increase_dim(x)  # B, G, 1024
    global_feature = torch.max(token_features, dim=1)[0]  # B, 1024

    # Global骨架: coarse_pred生成
    num_query = base_model.num_query
    coarse_pred = base_model.coarse_pred(global_feature).reshape(bs, -1, 3)  # (B, num_query, 3)

    # Local骨架: FPS从输入采样
    coarse_inp = misc.fps(xyz, num_query // 2)  # (B, num_query//2, 3)

    # 拼接
    coarse_all = torch.cat([coarse_pred, coarse_inp], dim=1)  # (B, num_query + num_query//2, 3)

    # 追踪来源: 0=global_pred, 1=local_fps
    n_global = coarse_pred.size(1)
    n_local = coarse_inp.size(1)
    source_labels = torch.cat([
        torch.zeros(n_global, device=xyz.device, dtype=torch.long),
        torch.ones(n_local, device=xyz.device, dtype=torch.long),
    ]).unsqueeze(0).expand(bs, -1)  # (B, n_global+n_local)

    # Query ranking
    query_ranking = base_model.query_ranking(coarse_all)  # (B, N, 1)
    idx = torch.argsort(query_ranking, dim=1, descending=True)  # (B, N, 1)
    selected_idx = idx[:, :num_query, 0]  # (B, num_query)

    # 按ranking选出骨架点
    coarse_selected = torch.gather(coarse_all, 1, selected_idx.unsqueeze(-1).expand(-1, -1, 3))
    source_selected = torch.gather(source_labels, 1, selected_idx)  # (B, num_query)

    # 分离 global 和 local 点
    global_mask = (source_selected[0] == 0)  # num_query
    local_mask = (source_selected[0] == 1)    # num_query

    coarse_np = coarse_selected[0].cpu().numpy()
    global_points = coarse_np[global_mask.cpu().numpy()]
    local_points = coarse_np[local_mask.cpu().numpy()]

    # 获取完整模型推理结果
    ret = model(xyz)
    coarse_final = ret[0][0].cpu().numpy()
    fine_points = ret[1][0].cpu().numpy()

    return {
        'coarse_all': coarse_np,         # query_ranking排序后的512个骨架点
        'global_points': global_points,   # 来自coarse_pred的部分
        'local_points': local_points,     # 来自fps的部分
        'coarse_final': coarse_final,     # 模型原始返回的骨架点
        'fine_points': fine_points,       # 最终精细点云
        'fps_centers': coor[0].cpu().numpy(),  # FPS中心点
        'n_global': global_points.shape[0],
        'n_local': local_points.shape[0],
    }


def find_sample_in_loader(val_loader, taxonomy_id, model_id):
    """在dataloader中查找指定样本"""
    for idx, (tax_ids, model_ids, data) in enumerate(val_loader):
        if tax_ids[0] == taxonomy_id and model_ids[0] == model_id:
            return idx, data
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="cfgs/PCN_models/pertoken_core_gpu1_ratio384.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./experiments/pertoken_core_gpu1_ratio384/PCN_models/exp_baseline_original/ckpt-best.pth",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./visualize_output/baseline_skeleton",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    config = cfg_from_yaml_file(args.config)
    config.model.NAME = "AdaPoinTr_PGST"

    model = builder.model_builder(config.model)
    builder.load_model(model, args.checkpoint)
    model.cuda()
    model.eval()

    # 构建测试数据集
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

    # 保存固定样本信息
    meta_path = os.path.join(args.out_dir, "sample_meta.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("Fixed 4 samples for baseline visualization\n")
        f.write(f"checkpoint={os.path.abspath(args.checkpoint)}\n")
        f.write(f"config={os.path.abspath(args.config)}\n\n")
        for s in FIXED_SAMPLES:
            f.write(f"{s['name']}\t{s['taxonomy_id']}\t{s['model_id']}\n")

    with torch.no_grad():
        for sample_info in FIXED_SAMPLES:
            tax_id = sample_info["taxonomy_id"]
            model_id = sample_info["model_id"]
            cat_name = sample_info["name"]

            print(f"Searching for {cat_name} ({model_id})...")
            _, data = find_sample_in_loader(val_loader, tax_id, model_id)
            if data is None:
                print(f"  WARNING: Sample not found, skipping!")
                continue

            partial, gt = data
            partial = partial.cuda()
            gt = gt.cuda()

            # 提取骨架点和来源
            result = extract_skeletons_with_source(model, partial)

            # 计算CD
            ret = model(partial)
            cd_l1 = chamfer_fn(ret[1], gt).item() * 1000

            partial_np = partial[0].cpu().numpy()
            gt_np = gt[0].cpu().numpy()
            fine_np = result['fine_points']
            global_pts = result['global_points']
            local_pts = result['local_points']
            fps_centers = result['fps_centers']

            prefix = os.path.join(args.out_dir, f"{cat_name}")

            print(f"  {cat_name}: CD-L1={cd_l1:.2f}, "
                  f"Global={result['n_global']}, Local={result['n_local']}, "
                  f"FPS={fps_centers.shape[0]}")

            # 保存xyz数据
            np.savetxt(f"{prefix}_partial.xyz", partial_np, fmt="%.6f", delimiter=" ")
            np.savetxt(f"{prefix}_gt.xyz", gt_np, fmt="%.6f", delimiter=" ")
            np.savetxt(f"{prefix}_fine.xyz", fine_np, fmt="%.6f", delimiter=" ")
            np.savetxt(f"{prefix}_skeleton_global.xyz", global_pts, fmt="%.6f", delimiter=" ")
            np.savetxt(f"{prefix}_skeleton_local.xyz", local_pts, fmt="%.6f", delimiter=" ")

            # ===== 图1: 完整6视图面板 =====
            fig = plt.figure(figsize=(24, 4))

            ax1 = fig.add_subplot(161, projection="3d")
            plot_point_cloud(partial_np, ax1, "Partial Input", color="gray", s=0.35)

            ax2 = fig.add_subplot(162, projection="3d")
            plot_point_cloud(gt_np, ax2, "Ground Truth", color="green", s=0.3)

            ax3 = fig.add_subplot(163, projection="3d")
            plot_point_cloud(fine_np, ax3, f"Prediction\nCD-L1={cd_l1:.2f}", color="royalblue", s=0.3)

            ax4 = fig.add_subplot(164, projection="3d")
            plot_point_cloud(fps_centers, ax4, f"FPS Centers\n{fps_centers.shape[0]} pts", color="lightgray", s=5)

            # 图5: 骨架点 local/global 分色
            ax5 = fig.add_subplot(165, projection="3d")
            ax5.scatter(local_pts[:, 0], local_pts[:, 1], local_pts[:, 2],
                        c='#2196F3', s=14, alpha=0.85, label=f'Local/FPS ({result["n_local"]})')
            ax5.scatter(global_pts[:, 0], global_pts[:, 1], global_pts[:, 2],
                        c='#FF5722', s=18, alpha=0.9, label=f'Global/Pred ({result["n_global"]})')
            ax5.set_title("Skeleton Points\n(Local vs Global)", fontsize=10)
            ax5.set_axis_off()
            ax5.view_init(elev=20, azim=45)
            mid, max_range = _axis_limits_from_points(local_pts, global_pts)
            ax5.set_xlim(mid[0] - max_range, mid[0] + max_range)
            ax5.set_ylim(mid[1] - max_range, mid[1] + max_range)
            ax5.set_zlim(mid[2] - max_range, mid[2] + max_range)
            ax5.legend(fontsize=7, loc='upper left')

            # 图6: 预测+骨架叠加
            ax6 = fig.add_subplot(166, projection="3d")
            ax6.scatter(fine_np[:, 0], fine_np[:, 1], fine_np[:, 2],
                        c='royalblue', s=0.3, alpha=0.3, label='Prediction')
            ax6.scatter(local_pts[:, 0], local_pts[:, 1], local_pts[:, 2],
                        c='#2196F3', s=12, alpha=0.9, label=f'Local ({result["n_local"]})')
            ax6.scatter(global_pts[:, 0], global_pts[:, 1], global_pts[:, 2],
                        c='#FF5722', s=16, alpha=0.9, label=f'Global ({result["n_global"]})')
            ax6.set_title("Pred + Skeleton", fontsize=10)
            ax6.set_axis_off()
            ax6.view_init(elev=20, azim=45)
            mid, max_range = _axis_limits_from_points(fine_np, local_pts, global_pts)
            ax6.set_xlim(mid[0] - max_range, mid[0] + max_range)
            ax6.set_ylim(mid[1] - max_range, mid[1] + max_range)
            ax6.set_zlim(mid[2] - max_range, mid[2] + max_range)
            ax6.legend(fontsize=6, loc='upper left')

            plt.suptitle(
                f"{cat_name.upper()} — Baseline Skeleton Visualization",
                fontsize=13, fontweight="bold",
            )
            plt.tight_layout()
            plt.savefig(f"{prefix}_full_panel.png", dpi=160, bbox_inches="tight")
            plt.close()

            # ===== 图2: 骨架点详细分解 =====
            fig2, axes = plt.subplots(1, 4, figsize=(20, 5), subplot_kw={'projection': '3d'})

            # FPS中心
            ax = axes[0]
            ax.scatter(fps_centers[:, 0], fps_centers[:, 1], fps_centers[:, 2],
                       c='gray', s=8, alpha=0.6)
            ax.set_title(f"FPS Centers ({fps_centers.shape[0]})", fontsize=10)
            ax.set_axis_off()
            ax.view_init(elev=20, azim=45)

            # Local骨架 (来自FPS采样)
            ax = axes[1]
            ax.scatter(local_pts[:, 0], local_pts[:, 1], local_pts[:, 2],
                       c='#2196F3', s=14, alpha=0.85)
            ax.set_title(f"Local/FPS Points ({result['n_local']})\nfrom input surface", fontsize=10)
            ax.set_axis_off()
            ax.view_init(elev=20, azim=45)

            # Global骨架 (来自MLP预测)
            ax = axes[2]
            ax.scatter(global_pts[:, 0], global_pts[:, 1], global_pts[:, 2],
                       c='#FF5722', s=18, alpha=0.9)
            ax.set_title(f"Global/Pred Points ({result['n_global']})\nfrom coarse_pred MLP", fontsize=10)
            ax.set_axis_off()
            ax.view_init(elev=20, azim=45)

            # Local + Global 叠加
            ax = axes[3]
            ax.scatter(local_pts[:, 0], local_pts[:, 1], local_pts[:, 2],
                       c='#2196F3', s=12, alpha=0.7, label=f'Local ({result["n_local"]})')
            ax.scatter(global_pts[:, 0], global_pts[:, 1], global_pts[:, 2],
                       c='#FF5722', s=18, alpha=0.9, label=f'Global ({result["n_global"]})')
            ax.set_title("Local + Global\n(All Skeleton)", fontsize=10)
            ax.set_axis_off()
            ax.view_init(elev=20, azim=45)
            ax.legend(fontsize=7)

            # 统一坐标范围
            for ax in axes.flat:
                mid, max_range = _axis_limits_from_points(fps_centers, local_pts, global_pts)
                ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
                ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
                ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

            plt.suptitle(
                f"{cat_name.upper()} — Skeleton Decomposition",
                fontsize=13, fontweight="bold",
            )
            plt.tight_layout()
            plt.savefig(f"{prefix}_skeleton_decomposition.png", dpi=160, bbox_inches="tight")
            plt.close()

            # ===== 图3: GT vs Pred 对比(叠加骨架) =====
            fig3 = plt.figure(figsize=(14, 5))
            ax_gt = fig3.add_subplot(121, projection="3d")
            ax_pred = fig3.add_subplot(122, projection="3d")

            # GT + 骨架
            ax_gt.scatter(gt_np[:, 0], gt_np[:, 1], gt_np[:, 2],
                          c='green', s=0.3, alpha=0.4, label='GT')
            ax_gt.scatter(local_pts[:, 0], local_pts[:, 1], local_pts[:, 2],
                          c='#2196F3', s=12, alpha=0.9, label=f'Local')
            ax_gt.scatter(global_pts[:, 0], global_pts[:, 1], global_pts[:, 2],
                          c='#FF5722', s=16, alpha=0.9, label=f'Global')
            ax_gt.set_title("GT + Skeleton", fontsize=10)
            ax_gt.set_axis_off()
            ax_gt.view_init(elev=20, azim=45)
            ax_gt.legend(fontsize=7, loc='upper left')

            # Pred + 骨架
            ax_pred.scatter(fine_np[:, 0], fine_np[:, 1], fine_np[:, 2],
                            c='royalblue', s=0.3, alpha=0.4, label='Pred')
            ax_pred.scatter(local_pts[:, 0], local_pts[:, 1], local_pts[:, 2],
                            c='#2196F3', s=12, alpha=0.9, label=f'Local')
            ax_pred.scatter(global_pts[:, 0], global_pts[:, 1], global_pts[:, 2],
                            c='#FF5722', s=16, alpha=0.9, label=f'Global')
            ax_pred.set_title(f"Pred + Skeleton\nCD-L1={cd_l1:.2f}", fontsize=10)
            ax_pred.set_axis_off()
            ax_pred.view_init(elev=20, azim=45)
            ax_pred.legend(fontsize=7, loc='upper left')

            mid, max_range = _axis_limits_from_points(gt_np, fine_np, local_pts, global_pts)
            for ax in [ax_gt, ax_pred]:
                ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
                ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
                ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

            plt.suptitle(
                f"{cat_name.upper()} — GT vs Pred with Skeletons",
                fontsize=13, fontweight="bold",
            )
            plt.tight_layout()
            plt.savefig(f"{prefix}_gt_vs_pred.png", dpi=160, bbox_inches="tight")
            plt.close()

            print(f"  Saved to {prefix}_*.png")

    print(f"\nDone! Output: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
