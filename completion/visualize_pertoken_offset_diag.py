#!/usr/bin/env python
"""
Diagnostic visualization for exp_pertoken_global_offset.
Compares with exp_unshuffle_grouped_v3 baseline.

Key diagnostic questions:
  1. Are the FPS-selected anchors (sel_coor) biased toward partial regions?
  2. Are global offsets too large / misdirected?
  3. Do global points actually fill holes or just cluster with local points?
  4. How does local vs global coverage compare to baseline?

Usage:
  python visualize_pertoken_offset_diag.py
  python visualize_pertoken_offset_diag.py --seed 42 --num_samples 8
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
    "04256520": "sofa", "03001627": "chair", "02958343": "car",
    "04530566": "watercraft", "04379243": "table", "02691156": "airplane",
    "02933112": "cabinet", "03636649": "lamp",
}


def _axis_limits(*arrays):
    pts = np.concatenate([a for a in arrays if a is not None and len(a)], axis=0)
    mx = np.array([pts[:, i].max() - pts[:, i].min() for i in range(3)]).max() / 2.0
    mid = pts.mean(axis=0)
    return mid, mx


def setup_ax(ax, mid, mx, elev=20, azim=45):
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(mid[0] - mx, mid[0] + mx)
    ax.set_ylim(mid[1] - mx, mid[1] + mx)
    ax.set_zlim(mid[2] - mx, mid[2] + mx)


class PertokenOffsetCapture:
    """Capture intermediate data for the pertoken_global_offset architecture."""
    def __init__(self):
        self.grouper_coor = None
        self.increase_dim_out = None
        self.global_offset_out = None
        self._handles = []

    def install(self, base_model):
        def hook_grouper(module, inp, out):
            self.grouper_coor = out[0].detach()
        def hook_increase_dim(module, inp, out):
            self.increase_dim_out = out.detach()
        def hook_global_offset(module, inp, out):
            self.global_offset_out = out.detach()

        self._handles.append(
            base_model.grouper.register_forward_hook(hook_grouper))
        self._handles.append(
            base_model.increase_dim.register_forward_hook(hook_increase_dim))
        self._handles.append(
            base_model.global_offset_net.register_forward_hook(hook_global_offset))
        self._base_model = base_model

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def get_decomposition(self, num_global):
        """Reconstruct local/global decomposition from captured intermediate data."""
        from pointnet2_ops import pointnet2_utils

        coor = self.grouper_coor          # (B, 384, 3)
        token_features = self.increase_dim_out  # (B, 384, D)
        B = coor.size(0)

        # Local: flow matching
        fm_cond = torch.cat([token_features, coor], dim=-1)
        with torch.no_grad():
            coarse_local = self._base_model.flow_offset.sample(coor, fm_cond, num_steps=10)

        # Global: FPS select + offset
        sel_idx = pointnet2_utils.furthest_point_sample(coor, num_global).long()
        sel_coor = torch.gather(coor, 1, sel_idx.unsqueeze(-1).expand(-1, -1, 3))
        sel_feat = torch.gather(token_features, 1, sel_idx.unsqueeze(-1).expand(-1, -1, token_features.size(-1)))
        global_offset = self.global_offset_out  # (B, num_global, 3) — captured from hook
        coarse_global = sel_coor + global_offset

        return coor, coarse_local, coarse_global, sel_coor, global_offset


class BaselineCapture:
    """Capture intermediate data for the unshuffle_grouped_v3 baseline.
    
    Handles both old architecture (global_coarse_pred) and new (global_offset_net).
    When loading old checkpoint with new code, global_offset_net weights are randomly
    initialized — we fall back to just using the coarse output directly.
    """
    def __init__(self):
        self.grouper_coor = None
        self.global_pred_flat = None
        self.increase_dim_out = None
        self._handles = []
        self._use_fm = False
        self._use_global_coarse_pred = False

    def install(self, base_model):
        self._use_fm = hasattr(base_model, 'flow_offset')
        self._use_global_coarse_pred = hasattr(base_model, 'global_coarse_pred')

        def hook_global_pred(module, inp, out):
            self.global_pred_flat = out.detach()
        def hook_grouper(module, inp, out):
            self.grouper_coor = out[0].detach()
        def hook_increase_dim(module, inp, out):
            self.increase_dim_out = out.detach()

        self._handles.append(
            base_model.grouper.register_forward_hook(hook_grouper))
        self._handles.append(
            base_model.increase_dim.register_forward_hook(hook_increase_dim))

        if self._use_global_coarse_pred:
            self._handles.append(
                base_model.global_coarse_pred.register_forward_hook(hook_global_pred))

        self._base_model = base_model

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def get_decomposition(self, num_global):
        coor = self.grouper_coor
        B = coor.size(0)

        if self._use_fm:
            token_features = self.increase_dim_out
            fm_cond = torch.cat([token_features, coor], dim=-1)
            with torch.no_grad():
                coarse_local = self._base_model.flow_offset.sample(coor, fm_cond, num_steps=10)
        else:
            coarse_local = coor  # fallback

        if self._use_global_coarse_pred and self.global_pred_flat is not None:
            coarse_global = self.global_pred_flat.reshape(B, num_global, 3)
        else:
            # Fallback: split coarse output into local + global portions
            coarse_global = coarse_local[B-1:]  # dummy; will be overridden below
            coarse_global = torch.zeros(B, num_global, 3, device=coor.device)

        return coor, coarse_local, coarse_global, None, None


def compute_nn_distance(points, ref):
    """Average nearest-neighbor distance from points to ref (numpy)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(ref)
    dists, _ = tree.query(points)
    return dists


def draw_arrows(ax, origins, targets, color='orange', alpha=0.4, lw=0.8, step=1):
    """Draw arrows from origins to targets on a 3D axes."""
    for i in range(0, len(origins), step):
        ax.plot([origins[i, 0], targets[i, 0]],
                [origins[i, 1], targets[i, 1]],
                [origins[i, 2], targets[i, 2]],
                c=color, alpha=alpha, linewidth=lw)


def visualize_sample(
    partial_np, gt_np, fine_np, coarse_np,
    local_np, global_np, coor_np,
    sel_coor_np=None, global_offset_np=None,
    prefix="", tax="", sample_idx=0, exp_name=""
):
    """Generate diagnostic panels for one sample."""
    chamfer_fn = ChamferDistanceL1()
    mid_all, mx_all = _axis_limits(gt_np, coarse_np, local_np, global_np, fine_np)

    # === FIG 1: 8-panel full diagnostic ===
    fig = plt.figure(figsize=(36, 4.5))

    # 1. Partial
    ax = fig.add_subplot(181, projection="3d")
    ax.scatter(*partial_np.T, c="gray", s=0.3, alpha=0.5)
    ax.set_title("Partial", fontsize=9)
    setup_ax(ax, mid_all, mx_all)

    # 2. GT
    ax = fig.add_subplot(182, projection="3d")
    ax.scatter(*gt_np.T, c="green", s=0.15, alpha=0.35)
    ax.set_title("Ground Truth", fontsize=9)
    setup_ax(ax, mid_all, mx_all)

    # 3. Local on GT
    cd_local = chamfer_fn(torch.from_numpy(local_np).unsqueeze(0).cuda(),
                          torch.from_numpy(gt_np).unsqueeze(0).cuda()).item() * 1000
    ax = fig.add_subplot(183, projection="3d")
    ax.scatter(*gt_np.T, c="lightgreen", s=0.1, alpha=0.12)
    ax.scatter(*local_np.T, c="red", s=4, alpha=0.85)
    ax.set_title(f"Local (384) on GT\nCD={cd_local:.2f}", fontsize=8)
    setup_ax(ax, mid_all, mx_all)

    # 4. Global on GT
    cd_global = chamfer_fn(torch.from_numpy(global_np).unsqueeze(0).cuda(),
                           torch.from_numpy(gt_np).unsqueeze(0).cuda()).item() * 1000
    ax = fig.add_subplot(184, projection="3d")
    ax.scatter(*gt_np.T, c="lightgreen", s=0.1, alpha=0.12)
    ax.scatter(*global_np.T, c="blue", s=14, alpha=0.9)
    ax.set_title(f"Global (128) on GT\nCD={cd_global:.2f}", fontsize=8)
    setup_ax(ax, mid_all, mx_all)

    # 5. Anchor + Offset arrows (pertoken model only)
    ax = fig.add_subplot(185, projection="3d")
    ax.scatter(*gt_np.T, c="lightgreen", s=0.08, alpha=0.1)
    if sel_coor_np is not None:
        ax.scatter(*sel_coor_np.T, c="yellow", s=10, alpha=0.9, marker="^", label="Anchor (FPS)")
        ax.scatter(*global_np.T, c="blue", s=12, alpha=0.85, label="Global (offset)")
        offset_norms = np.linalg.norm(global_offset_np, axis=1) if global_offset_np is not None else np.zeros(len(global_np))
        draw_arrows(ax, sel_coor_np, global_np, color='orange', alpha=0.35, lw=0.6, step=max(1, len(sel_coor_np)//40))
        ax.set_title(f"Anchor→Global offset\nmean={offset_norms.mean():.3f} max={offset_norms.max():.3f}", fontsize=8)
        ax.legend(fontsize=6, loc="upper right")
    else:
        ax.scatter(*global_np.T, c="magenta", s=14, alpha=0.9)
        ax.set_title(f"Global (128)\nDirect prediction", fontsize=8)
    setup_ax(ax, mid_all, mx_all)

    # 6. Local + Global combined on GT
    cd_coarse = chamfer_fn(torch.from_numpy(coarse_np).unsqueeze(0).cuda(),
                           torch.from_numpy(gt_np).unsqueeze(0).cuda()).item() * 1000
    ax = fig.add_subplot(186, projection="3d")
    ax.scatter(*gt_np.T, c="lightgreen", s=0.1, alpha=0.12)
    ax.scatter(*local_np.T, c="red", s=3, alpha=0.7, label="Local (384)")
    ax.scatter(*global_np.T, c="blue", s=10, alpha=0.9, label="Global (128)")
    ax.set_title(f"Local+Global (512)\nCD={cd_coarse:.2f}", fontsize=8)
    ax.legend(fontsize=6, loc="upper right")
    setup_ax(ax, mid_all, mx_all)

    # 7. Final coarse (after query_ranking)
    cd_fine = chamfer_fn(torch.from_numpy(fine_np).unsqueeze(0).cuda(),
                         torch.from_numpy(gt_np).unsqueeze(0).cuda()).item() * 1000
    ax = fig.add_subplot(187, projection="3d")
    ax.scatter(*coarse_np.T, c="darkorange", s=6, alpha=0.8)
    ax.set_title(f"Final Skeleton (sorted)\nCD={cd_coarse:.2f}", fontsize=8)
    setup_ax(ax, mid_all, mx_all)

    # 8. Fine prediction
    ax = fig.add_subplot(188, projection="3d")
    ax.scatter(*fine_np.T, c="royalblue", s=0.15, alpha=0.35)
    ax.set_title(f"Fine (16384)\nCD={cd_fine:.2f}", fontsize=8)
    setup_ax(ax, mid_all, mx_all)

    plt.suptitle(f"{tax.upper()} #{sample_idx} — {exp_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{prefix}_full_diag.png", dpi=160, bbox_inches="tight")
    plt.close()

    # === FIG 2: Multi-angle offset arrows ===
    if sel_coor_np is not None:
        angles = [(20, 45), (20, 135), (20, 225), (80, 45)]
        fig, axes = plt.subplots(2, 4, figsize=(24, 11), subplot_kw={"projection": "3d"})

        for j, (elev, azim) in enumerate(angles):
            # Top row: anchors + global + arrows on GT
            ax = axes[0, j]
            ax.scatter(*gt_np.T, c="lightgreen", s=0.06, alpha=0.08)
            ax.scatter(*partial_np.T, c="gray", s=0.15, alpha=0.25)
            ax.scatter(*sel_coor_np.T, c="yellow", s=8, alpha=0.9, marker="^")
            ax.scatter(*global_np.T, c="blue", s=10, alpha=0.85)
            draw_arrows(ax, sel_coor_np, global_np, color='orange', alpha=0.4, lw=0.7, step=max(1, len(sel_coor_np)//40))
            ax.set_title(f"Anchor→Global (e={elev} a={azim})", fontsize=7)
            setup_ax(ax, mid_all, mx_all, elev=elev, azim=azim)

            # Bottom row: local vs global coverage heat
            ax = axes[1, j]
            ax.scatter(*gt_np.T, c="lightgreen", s=0.06, alpha=0.08)
            ax.scatter(*local_np.T, c="red", s=3, alpha=0.7)
            ax.scatter(*global_np.T, c="blue", s=10, alpha=0.85)
            ax.scatter(*coor_np.T, c="lightgray", s=1.5, alpha=0.3)
            ax.set_title("Local(red) vs Global(blue)", fontsize=7)
            setup_ax(ax, mid_all, mx_all, elev=elev, azim=azim)

        offset_norms = np.linalg.norm(global_offset_np, axis=1) if global_offset_np is not None else np.zeros(1)
        plt.suptitle(f"{tax.upper()} #{sample_idx} — Offset Analysis\n"
                     f"offset: mean={offset_norms.mean():.4f}, std={offset_norms.std():.4f}, "
                     f"max={offset_norms.max():.4f}, median={np.median(offset_norms):.4f}",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{prefix}_offset_multiangle.png", dpi=140, bbox_inches="tight")
        plt.close()

    # === FIG 3: Offset histogram ===
    if sel_coor_np is not None and global_offset_np is not None:
        offset_norms = np.linalg.norm(global_offset_np, axis=1)

        fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

        # 3a: offset magnitude histogram
        ax = axes[0]
        ax.hist(offset_norms, bins=50, color="tomato", alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.axvline(offset_norms.mean(), color="red", ls="--", label=f"mean={offset_norms.mean():.4f}")
        ax.axvline(np.median(offset_norms), color="blue", ls="--", label=f"median={np.median(offset_norms):.4f}")
        ax.set_xlabel("Offset L2 norm")
        ax.set_ylabel("Count")
        ax.set_title(f"{tax} — Global offset magnitude")
        ax.legend(fontsize=8)

        # 3b: per-axis offset
        ax = axes[1]
        colors_ax = ["red", "green", "blue"]
        for i, (name, c) in enumerate(zip(["X", "Y", "Z"], colors_ax)):
            ax.hist(global_offset_np[:, i], bins=50, alpha=0.5, color=c, label=name)
        ax.set_xlabel("Offset value")
        ax.set_ylabel("Count")
        ax.set_title("Per-axis global offset")
        ax.legend(fontsize=8)

        # 3c: anchor distance to GT vs global point distance to GT
        ax = axes[2]
        try:
            from scipy.spatial import cKDTree
            gt_tree = cKDTree(gt_np)
            anchor_nn = np.array([gt_tree.query(p)[0] for p in sel_coor_np])
            global_nn = np.array([gt_tree.query(p)[0] for p in global_np])
            ax.hist(anchor_nn, bins=50, alpha=0.6, color="yellow", edgecolor="orange", label="Anchor→GT")
            ax.hist(global_nn, bins=50, alpha=0.6, color="blue", edgecolor="navy", label="Global→GT")
            ax.axvline(anchor_nn.mean(), color="orange", ls="--", label=f"Anchor mean={anchor_nn.mean():.4f}")
            ax.axvline(global_nn.mean(), color="navy", ls="--", label=f"Global mean={global_nn.mean():.4f}")
        except ImportError:
            ax.text(0.5, 0.5, "scipy not available", ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel("NN distance to GT")
        ax.set_ylabel("Count")
        ax.set_title("Anchor vs Global proximity to GT")
        ax.legend(fontsize=7)

        plt.tight_layout()
        plt.savefig(f"{prefix}_offset_hist.png", dpi=140, bbox_inches="tight")
        plt.close()

    # === FIG 4: Coverage analysis (projected 2D) ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Top-left: XY projection
    ax = axes[0, 0]
    ax.scatter(gt_np[:, 0], gt_np[:, 1], c="lightgreen", s=0.3, alpha=0.3, label="GT")
    ax.scatter(partial_np[:, 0], partial_np[:, 1], c="gray", s=0.3, alpha=0.4, label="Partial")
    ax.scatter(local_np[:, 0], local_np[:, 1], c="red", s=3, alpha=0.7, label="Local (384)")
    ax.scatter(global_np[:, 0], global_np[:, 1], c="blue", s=8, alpha=0.9, label="Global (128)")
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title(f"{tax} XY projection")
    ax.legend(fontsize=7)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    # Top-right: XZ projection
    ax = axes[0, 1]
    ax.scatter(gt_np[:, 0], gt_np[:, 2], c="lightgreen", s=0.3, alpha=0.3, label="GT")
    ax.scatter(partial_np[:, 0], partial_np[:, 2], c="gray", s=0.3, alpha=0.4, label="Partial")
    ax.scatter(local_np[:, 0], local_np[:, 2], c="red", s=3, alpha=0.7, label="Local (384)")
    ax.scatter(global_np[:, 0], global_np[:, 2], c="blue", s=8, alpha=0.9, label="Global (128)")
    if sel_coor_np is not None:
        ax.scatter(sel_coor_np[:, 0], sel_coor_np[:, 2], c="yellow", s=6, alpha=0.8, marker="^", label="Anchor")
    ax.set_xlabel("X"); ax.set_ylabel("Z")
    ax.set_title(f"{tax} XZ projection")
    ax.legend(fontsize=7)
    ax.set_aspect("equal")

    # Bottom-left: YZ projection
    ax = axes[1, 0]
    ax.scatter(gt_np[:, 1], gt_np[:, 2], c="lightgreen", s=0.3, alpha=0.3, label="GT")
    ax.scatter(partial_np[:, 1], partial_np[:, 2], c="gray", s=0.3, alpha=0.4, label="Partial")
    ax.scatter(local_np[:, 1], local_np[:, 2], c="red", s=3, alpha=0.7, label="Local (384)")
    ax.scatter(global_np[:, 1], global_np[:, 2], c="blue", s=8, alpha=0.9, label="Global (128)")
    ax.set_xlabel("Y"); ax.set_ylabel("Z")
    ax.set_title(f"{tax} YZ projection")
    ax.legend(fontsize=7)
    ax.set_aspect("equal")

    # Bottom-right: offset magnitude color-coded on XY
    if sel_coor_np is not None and global_offset_np is not None:
        ax = axes[1, 1]
        offset_norms = np.linalg.norm(global_offset_np, axis=1)
        sc = ax.scatter(sel_coor_np[:, 0], sel_coor_np[:, 1],
                        c=offset_norms, cmap="hot", s=15, alpha=0.9,
                        edgecolors="black", linewidth=0.3)
        plt.colorbar(sc, ax=ax, label="Offset L2 norm")
        ax.scatter(gt_np[:, 0], gt_np[:, 1], c="lightgreen", s=0.2, alpha=0.2)
        ax.set_xlabel("X"); ax.set_ylabel("Y")
        ax.set_title(f"{tax} Anchor position + offset magnitude")
        ax.set_aspect("equal")
        ax.invert_yaxis()

    plt.suptitle(f"{tax.upper()} #{sample_idx} — Coverage Analysis ({exp_name})", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{prefix}_coverage_2d.png", dpi=140, bbox_inches="tight")
    plt.close()

    return {
        'cd_fine': cd_fine, 'cd_coarse': cd_coarse,
        'cd_local': cd_local, 'cd_global': cd_global,
    }


def load_model(config_path, ckpt_path):
    config = cfg_from_yaml_file(config_path)
    config.model.NAME = "AdaPoinTr_PGST"
    model = builder.model_builder(config.model)
    builder.load_model(model, ckpt_path)
    model.cuda().eval()
    return model, config


def get_val_loader(config):
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
    return val_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        default="cfgs/PCN_models/pertoken_core_gpu1_ratio384.yaml")
    parser.add_argument("--checkpoint",
                        default="./experiments/pertoken_core_gpu1_ratio384/PCN_models/"
                                "exp_pertoken_global_offset/ckpt-best.pth")
    parser.add_argument("--out_dir", default="./visualize_output/pertoken_offset_diag")
    parser.add_argument("--seed", type=int, default=20260409)
    parser.add_argument("--num_categories", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    # Load model
    print("Loading pertoken_global_offset model...")
    model, config = load_model(args.config, args.checkpoint)

    # Install hooks
    cap = PertokenOffsetCapture()
    cap.install(model.base_model)

    num_local = model.base_model.center_num[-1]    # 384
    num_global = model.base_model.num_global_points  # 128
    print(f"Architecture: {num_local} local + {num_global} global = {num_local + num_global} coarse points")

    val_loader = get_val_loader(config)
    chamfer_fn = ChamferDistanceL1()

    # Collect samples per category
    all_cats = sorted(set(SYNSET_TO_NAME.values()))
    chosen = all_cats.copy()
    rng.shuffle(chosen)
    targets = chosen[:args.num_categories]
    need = {c: 1 for c in targets}
    done = {c: 0 for c in targets}

    summary_lines = []
    summary_lines.append(f"=== Diagnostic: pertoken_global_offset (ckpt-best) ===\n")
    summary_lines.append(f"{'Cat':12s} | {'CD_fine':>10s} {'CD_coarse':>12s} {'CD_local':>12s} {'CD_global':>13s}")
    summary_lines.append("-" * 70)

    with torch.no_grad():
        for _, (taxonomy_id, model_id, data) in enumerate(val_loader):
            if all(done[c] >= need[c] for c in targets):
                break
            tax = SYNSET_TO_NAME.get(taxonomy_id[0], taxonomy_id[0])
            if tax not in need or done[tax] >= need[tax]:
                continue

            partial, gt = data
            partial, gt = partial.cuda(), gt.cuda()

            ret = model(partial)
            coarse, fine = ret[0], ret[1]
            coor_np, local_np, global_np, sel_coor_np, offset_np = cap.get_decomposition(num_global)

            mid_short = str(model_id[0])[:12] if model_id is not None else str(done[tax])
            prefix = os.path.join(args.out_dir, f"{tax}_{mid_short}")

            m = visualize_sample(
                partial[0].cpu().numpy(), gt[0].cpu().numpy(),
                fine[0].cpu().numpy(), coarse[0].cpu().numpy(),
                local_np[0].cpu().numpy(), global_np[0].cpu().numpy(),
                coor_np[0].cpu().numpy(),
                sel_coor_np=sel_coor_np[0].cpu().numpy() if sel_coor_np is not None else None,
                global_offset_np=offset_np[0].cpu().numpy() if offset_np is not None else None,
                prefix=prefix, tax=tax, sample_idx=done[tax],
                exp_name="pertoken_global_offset"
            )
            done[tax] += 1

            line = (f"{tax:12s} | {m['cd_fine']:10.2f} {m['cd_coarse']:12.2f} "
                    f"{m['cd_local']:12.2f} {m['cd_global']:13.2f}")
            summary_lines.append(line)
            print(line)

    cap.remove()

    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"\nDone. Output: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
