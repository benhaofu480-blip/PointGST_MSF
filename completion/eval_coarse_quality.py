#!/usr/bin/env python
"""
Evaluate coarse (skeleton) point quality for exp_ratio_384_128_v2.
Uses forward hooks to decompose 384 local + 128 global coarse points,
computes per-component CD metrics, and generates multi-angle visualizations.
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
from utils import misc

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
    ax.set_xlim(mid[0]-mx, mid[0]+mx)
    ax.set_ylim(mid[1]-mx, mid[1]+mx)
    ax.set_zlim(mid[2]-mx, mid[2]+mx)


class IntermediateCapture:
    """Use forward hooks to capture coarse decomposition (supports both MLP and FM models)."""
    def __init__(self):
        self.coarse_offset = None
        self.global_pred_flat = None
        self.grouper_coor = None
        self.increase_dim_out = None
        self._handles = []
        self._use_fm = False

    def install(self, base_model):
        self._use_fm = hasattr(base_model, 'flow_offset')

        def hook_global_pred(module, inp, out):
            self.global_pred_flat = out.detach()
        def hook_grouper(module, inp, out):
            self.grouper_coor = out[0].detach()
        def hook_increase_dim(module, inp, out):
            self.increase_dim_out = out.detach()

        self._handles.append(
            base_model.global_coarse_pred.register_forward_hook(hook_global_pred))
        self._handles.append(
            base_model.grouper.register_forward_hook(hook_grouper))
        self._handles.append(
            base_model.increase_dim.register_forward_hook(hook_increase_dim))

        if not self._use_fm:
            def hook_coarse_pred(module, inp, out):
                self.coarse_offset = out.detach()
            self._handles.append(
                base_model.coarse_pred.register_forward_hook(hook_coarse_pred))

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
            coarse_local = coor + self.coarse_offset

        coarse_global = self.global_pred_flat.reshape(B, num_global, 3)
        return coor, coarse_local, coarse_global


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="cfgs/PCN_models/pertoken_core_gpu1_ratio384.yaml")
    parser.add_argument("--checkpoint",
                        default="./experiments/pertoken_core_gpu1_ratio384/PCN_models/"
                                "exp_ratio_384_128_v2/ckpt-best.pth")
    parser.add_argument("--out_dir", default="./visualize_output/coarse_eval_v2")
    parser.add_argument("--exp_name", default="v2 best")
    parser.add_argument("--seed", type=int, default=20260405)
    parser.add_argument("--num_categories", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rng = random.Random(args.seed)
    all_cats = sorted(set(SYNSET_TO_NAME.values()))
    chosen = all_cats.copy()
    rng.shuffle(chosen)
    targets = chosen[:args.num_categories]
    need = {c: 1 for c in targets}
    done = {c: 0 for c in targets}

    config = cfg_from_yaml_file(args.config)
    config.model.NAME = "AdaPoinTr_PGST"
    model = builder.model_builder(config.model)
    builder.load_model(model, args.checkpoint)
    model.cuda().eval()

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
    cap = IntermediateCapture()
    cap.install(model.base_model)

    num_local = model.base_model.center_num[-1]    # 384
    num_global = model.base_model.num_global_points  # 128
    print(f"Architecture: {num_local} local + {num_global} global = "
          f"{num_local + num_global} coarse points\n")

    summary_lines = []

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
            coarse_final, fine = ret[0], ret[1]

            coor, coarse_local, coarse_global = cap.get_decomposition(num_global)

            # numpy conversions
            partial_np = partial[0].cpu().numpy()
            gt_np = gt[0].cpu().numpy()
            fine_np = fine[0].cpu().numpy()
            coarse_final_np = coarse_final[0].cpu().numpy()
            local_np = coarse_local[0].cpu().numpy()
            global_np = coarse_global[0].cpu().numpy()
            coor_np = coor[0].cpu().numpy()

            # === Metrics ===
            cd_fine = chamfer_fn(fine, gt).item() * 1000
            cd_coarse = chamfer_fn(coarse_final, gt).item() * 1000
            cd_local = chamfer_fn(coarse_local, gt).item() * 1000
            cd_global = chamfer_fn(coarse_global, gt).item() * 1000

            gt_fps = misc.fps(gt, num_local + num_global)
            cd_coarse_vs_fps = chamfer_fn(coarse_final, gt_fps).item() * 1000

            offsets = local_np - coor_np
            offset_norms = np.linalg.norm(offsets, axis=1)

            mid_short = str(model_id[0])[:12] if model_id is not None else str(done[tax])
            prefix = os.path.join(args.out_dir, f"{tax}_{mid_short}")

            line = (f"{tax:12s} | CD_fine={cd_fine:7.2f} | CD_coarse={cd_coarse:7.2f} | "
                    f"CD_local384={cd_local:7.2f} | CD_global128={cd_global:7.2f} | "
                    f"CD_vs_FPS_GT={cd_coarse_vs_fps:7.2f} | "
                    f"offset: mean={offset_norms.mean():.4f} max={offset_norms.max():.4f} "
                    f"std={offset_norms.std():.4f}")
            print(line)
            summary_lines.append(line)

            mid_all, mx_all = _axis_limits(gt_np, coarse_final_np, local_np, global_np)

            # ====== FIG 1: 6-panel overview ======
            fig = plt.figure(figsize=(30, 4.5))

            ax = fig.add_subplot(161, projection="3d")
            ax.scatter(*partial_np.T, c="gray", s=0.3, alpha=0.5)
            ax.set_title("Partial", fontsize=10)
            setup_ax(ax, mid_all, mx_all)

            ax = fig.add_subplot(162, projection="3d")
            ax.scatter(*gt_np.T, c="green", s=0.2, alpha=0.4)
            ax.set_title("Ground Truth", fontsize=10)
            setup_ax(ax, mid_all, mx_all)

            ax = fig.add_subplot(163, projection="3d")
            ax.scatter(*gt_np.T, c="lightgreen", s=0.1, alpha=0.15)
            ax.scatter(*local_np.T, c="red", s=5, alpha=0.85)
            ax.set_title(f"Local ({num_local}) on GT\nCD={cd_local:.2f}", fontsize=9)
            setup_ax(ax, mid_all, mx_all)

            ax = fig.add_subplot(164, projection="3d")
            ax.scatter(*gt_np.T, c="lightgreen", s=0.1, alpha=0.15)
            ax.scatter(*global_np.T, c="blue", s=12, alpha=0.9)
            ax.set_title(f"Global ({num_global}) on GT\nCD={cd_global:.2f}", fontsize=9)
            setup_ax(ax, mid_all, mx_all)

            ax = fig.add_subplot(165, projection="3d")
            ax.scatter(*gt_np.T, c="lightgreen", s=0.1, alpha=0.15)
            ax.scatter(*local_np.T, c="red", s=3, alpha=0.75, label=f"local({num_local})")
            ax.scatter(*global_np.T, c="blue", s=10, alpha=0.9, label=f"global({num_global})")
            ax.set_title(f"All {num_local+num_global} on GT\nCD={cd_coarse:.2f}", fontsize=9)
            ax.legend(fontsize=7, loc="upper right")
            setup_ax(ax, mid_all, mx_all)

            ax = fig.add_subplot(166, projection="3d")
            ax.scatter(*fine_np.T, c="royalblue", s=0.2, alpha=0.4)
            ax.set_title(f"Fine ({fine_np.shape[0]})\nCD={cd_fine:.2f}", fontsize=9)
            setup_ax(ax, mid_all, mx_all)

            plt.suptitle(f"{tax.upper()} — Coarse Decomposition ({args.exp_name})",
                         fontsize=12, fontweight="bold")
            plt.tight_layout()
            plt.savefig(f"{prefix}_coarse_eval.png", dpi=160, bbox_inches="tight")
            plt.close()

            # ====== FIG 2: Multi-angle + offset arrows ======
            angles = [(20, 45), (20, 135), (20, 225), (80, 45)]
            fig, axes = plt.subplots(2, 4, figsize=(22, 11),
                                      subplot_kw={"projection": "3d"})

            for j, (elev, azim) in enumerate(angles):
                ax = axes[0, j]
                ax.scatter(*gt_np.T, c="lightgreen", s=0.08, alpha=0.12)
                ax.scatter(*local_np.T, c="red", s=3, alpha=0.8)
                ax.scatter(*global_np.T, c="blue", s=10, alpha=0.9)
                ax.set_title(f"elev={elev} azim={azim}", fontsize=8)
                setup_ax(ax, mid_all, mx_all, elev=elev, azim=azim)

                ax = axes[1, j]
                ax.scatter(*coor_np.T, c="gray", s=2, alpha=0.4, label="center")
                ax.scatter(*local_np.T, c="red", s=3, alpha=0.6, label="local")
                step = max(1, len(coor_np) // 50)
                for k in range(0, len(coor_np), step):
                    ax.plot([coor_np[k,0], local_np[k,0]],
                            [coor_np[k,1], local_np[k,1]],
                            [coor_np[k,2], local_np[k,2]],
                            c="orange", alpha=0.35, linewidth=0.5)
                ax.set_title("Center → Local offset", fontsize=8)
                setup_ax(ax, mid_all, mx_all, elev=elev, azim=azim)

            plt.suptitle(f"{tax.upper()} — Multi-angle (offset mean={offset_norms.mean():.4f}, "
                         f"max={offset_norms.max():.4f})",
                         fontsize=11, fontweight="bold")
            plt.tight_layout()
            plt.savefig(f"{prefix}_multiangle.png", dpi=140, bbox_inches="tight")
            plt.close()

            # ====== FIG 3: Offset histogram ======
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.hist(offset_norms, bins=50, color="tomato", alpha=0.7,
                     edgecolor="black", linewidth=0.5)
            ax1.axvline(offset_norms.mean(), color="red", ls="--",
                        label=f"mean={offset_norms.mean():.4f}")
            ax1.axvline(np.median(offset_norms), color="blue", ls="--",
                        label=f"median={np.median(offset_norms):.4f}")
            ax1.set_xlabel("Offset L2 norm")
            ax1.set_ylabel("Count")
            ax1.set_title(f"{tax} — Local offset magnitude")
            ax1.legend()

            for i, (name, c) in enumerate(zip(["X","Y","Z"], ["red","green","blue"])):
                ax2.hist(offsets[:,i], bins=50, alpha=0.5, color=c, label=name)
            ax2.set_xlabel("Offset value")
            ax2.set_ylabel("Count")
            ax2.set_title("Per-axis offset")
            ax2.legend()

            plt.tight_layout()
            plt.savefig(f"{prefix}_offset_hist.png", dpi=140, bbox_inches="tight")
            plt.close()

            done[tax] += 1

    cap.remove()

    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(f"=== Coarse Point Quality Evaluation ({args.exp_name}) ===\n")
        f.write(f"Architecture: {num_local} local + {num_global} global\n\n")
        for line in summary_lines:
            f.write(line + "\n")

    print(f"\nDone. Output: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
