#!/usr/bin/env python
"""
对比 exp_ratio_384_128_v2 (baseline) 与 exp_surface_fix_v1 (tanh 约束)
1. 画 CDL1 / F-Score 训练曲线
2. 对同 4 类物体（seed 相同）做 panel 可视化对比
"""
import os, sys, random
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
from mpl_toolkits.mplot3d import Axes3D

SYNSET_TO_NAME = {
    "04256520": "sofa", "03001627": "chair", "02958343": "car",
    "04530566": "watercraft", "04379243": "table", "02691156": "airplane",
    "02933112": "cabinet", "03636649": "lamp",
}


def _axis_equal(ax, *arrays):
    pts = np.concatenate([a for a in arrays if a is not None and len(a)], axis=0)
    mr = np.array([pts[:,i].max()-pts[:,i].min() for i in range(3)]).max()/2.0
    mid = pts.mean(axis=0)
    for i, s in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
        s(mid[i]-mr, mid[i]+mr)


def plot_pc(pts, ax, title, color="blue", s=0.3, alpha=0.55):
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], c=color, s=s, alpha=alpha)
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()
    ax.view_init(elev=20, azim=45)
    _axis_equal(ax, pts)


def plot_overlay(dense, dc, coarse, cc, ax, title):
    ax.scatter(dense[:,0], dense[:,1], dense[:,2], c=dc, s=0.2, alpha=0.3)
    ax.scatter(coarse[:,0], coarse[:,1], coarse[:,2], c=cc, s=10, alpha=0.9)
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()
    ax.view_init(elev=20, azim=45)
    _axis_equal(ax, dense, coarse)


def main():
    out_dir = "./visualize_output/compare_surface_fix"
    os.makedirs(out_dir, exist_ok=True)

    # ── Part 1: 训练曲线对比 ──
    epochs_old = list(range(0, 151, 10))
    cdl1_old = [14.682, 9.452, 9.321, 8.707, 8.542, 8.384, 8.683,
                8.207, 8.299, 8.483, 8.183, 8.153, 8.246, 8.226, 8.130, 8.097]
    fscore_old = [0.4943, 0.6875, 0.7150, 0.7318, 0.7455, 0.7491, 0.7464,
                  0.7586, 0.7590, 0.7538, 0.7679, 0.7663, 0.7652, 0.7683, 0.7723, 0.7744]

    epochs_new = list(range(0, 151, 10))
    cdl1_new = [19.412, 13.217, 11.749, 10.862, 10.217, 10.000, 9.727,
                9.657, 9.407, 9.461, 9.259, 9.130, 9.084, 9.100, 8.974, 8.903]
    fscore_new = [0.5483, 0.6059, 0.6312, 0.6499, 0.6678, 0.6800, 0.6936,
                  0.6955, 0.7054, 0.7053, 0.7126, 0.7183, 0.7231, 0.7237, 0.7280, 0.7301]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs_old, cdl1_old, "o-", label="v2 baseline (no constraint)", color="tab:blue")
    ax1.plot(epochs_new, cdl1_new, "s-", label="surface_fix_v1 (tanh constraint)", color="tab:orange")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("CDL1 (mm)")
    ax1.set_title("CDL1 Convergence"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax1.axhline(y=8.097, ls="--", color="tab:blue", alpha=0.4, label="v2 best=8.097")

    ax2.plot(epochs_old, fscore_old, "o-", label="v2 baseline", color="tab:blue")
    ax2.plot(epochs_new, fscore_new, "s-", label="surface_fix_v1", color="tab:orange")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("F-Score")
    ax2.set_title("F-Score Convergence"); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{out_dir}/training_curves.png", dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_dir}/training_curves.png")

    # ── Part 2: per-category CDL1 对比（两个实验的 epoch 150） ──
    cats = ["airplane","car","cabinet","chair","lamp","sofa","table","watercraft"]
    cdl1_old_cat = [4.578, 10.375, 8.782, 9.586, 7.219, 8.656, 10.152, 8.429]
    cdl1_new_cat = [4.471, 11.027, 9.733, 11.052, 7.500, 9.099, 10.188, 8.158]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(cats))
    w = 0.35
    ax.bar(x - w/2, cdl1_old_cat, w, label="v2 baseline", color="tab:blue", alpha=0.7)
    ax.bar(x + w/2, cdl1_new_cat, w, label="surface_fix_v1", color="tab:orange", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(cats, rotation=30, ha="right")
    ax.set_ylabel("CDL1 (mm)"); ax.set_title("Per-category CDL1 @ Epoch 150")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for i in range(len(cats)):
        delta = cdl1_new_cat[i] - cdl1_old_cat[i]
        color = "red" if delta > 0 else "green"
        ax.text(x[i]+w/2, cdl1_new_cat[i]+0.1, f"{delta:+.2f}", ha="center", fontsize=7, color=color)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/per_category_cdl1.png", dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_dir}/per_category_cdl1.png")

    # ── Part 3: 可视化对比（同 4 类样本） ──
    config = cfg_from_yaml_file("cfgs/PCN_models/pertoken_core_gpu1_ratio384.yaml")
    config.model.NAME = "AdaPoinTr_PGST"

    ckpt_old = "./experiments/pertoken_core_gpu1_ratio384/PCN_models/exp_ratio_384_128_v2/ckpt-best.pth"
    ckpt_new = "./experiments/pertoken_core_gpu1_ratio384/PCN_models/exp_surface_fix_v1/ckpt-best.pth"

    print("Loading v2 baseline model...")
    model_old = builder.model_builder(config.model)
    builder.load_model(model_old, ckpt_old)
    model_old.cuda().eval()

    print("Loading surface_fix_v1 model...")
    model_new = builder.model_builder(config.model)
    builder.load_model(model_new, ckpt_new)
    model_new.cuda().eval()

    test_cfg = EasyDict()
    if "_base_" in config.dataset.test:
        for k, v in config.dataset.test["_base_"].items(): test_cfg[k] = v
    if "others" in config.dataset.test:
        for k, v in config.dataset.test["others"].items(): test_cfg[k] = v
    test_cfg["NAME"] = test_cfg.get("NAME", "PCN")
    test_cfg["subset"] = test_cfg.get("subset", "test")
    val_dataset = build_dataset_from_cfg(test_cfg)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    chamfer_fn = ChamferDistanceL1()
    rng = random.Random(20260404)
    all_cats = sorted(set(SYNSET_TO_NAME.values()))
    rng.shuffle(all_cats)
    target_list = all_cats[:4]
    done = {c: 0 for c in target_list}

    with torch.no_grad():
        for _, (tid, mid, data) in enumerate(val_loader):
            if all(done[c] >= 1 for c in target_list):
                break
            tax = SYNSET_TO_NAME.get(tid[0], tid[0])
            if tax not in done or done[tax] >= 1:
                continue

            partial, gt = data
            partial, gt = partial.cuda(), gt.cuda()

            ret_old = model_old(partial)
            coarse_old, fine_old = ret_old[0][0].cpu().numpy(), ret_old[1][0].cpu().numpy()
            cd_old = chamfer_fn(ret_old[1], gt).item() * 1000

            ret_new = model_new(partial)
            coarse_new, fine_new = ret_new[0][0].cpu().numpy(), ret_new[1][0].cpu().numpy()
            cd_new = chamfer_fn(ret_new[1], gt).item() * 1000

            gt_np = gt[0].cpu().numpy()
            partial_np = partial[0].cpu().numpy()

            # 7 列: partial | GT | v2_coarse | v2_fine | fix_coarse | fix_fine | GT+两者骨架叠加
            fig = plt.figure(figsize=(28, 4.2))
            ax0 = fig.add_subplot(171, projection="3d")
            plot_pc(partial_np, ax0, "Partial", "gray", 0.3)
            ax1 = fig.add_subplot(172, projection="3d")
            plot_pc(gt_np, ax1, "Ground Truth", "green", 0.3)
            ax2 = fig.add_subplot(173, projection="3d")
            plot_pc(coarse_old, ax2, f"v2 skeleton\n(no constraint)", "darkorange", 8)
            ax3 = fig.add_subplot(174, projection="3d")
            plot_pc(fine_old, ax3, f"v2 fine\nCD={cd_old:.2f}", "royalblue", 0.3)
            ax4 = fig.add_subplot(175, projection="3d")
            plot_pc(coarse_new, ax4, f"fix skeleton\n(tanh constrained)", "red", 8)
            ax5 = fig.add_subplot(176, projection="3d")
            plot_pc(fine_new, ax5, f"fix fine\nCD={cd_new:.2f}", "purple", 0.3)
            ax6 = fig.add_subplot(177, projection="3d")
            plot_overlay(gt_np, "green", coarse_old, "darkorange", ax6, "GT + v2 skel")

            delta = cd_new - cd_old
            sym = "+" if delta > 0 else ""
            plt.suptitle(f"{tax.upper()}  v2={cd_old:.2f}  fix={cd_new:.2f}  (delta={sym}{delta:.2f})",
                         fontsize=12, fontweight="bold")
            plt.tight_layout()
            plt.savefig(f"{out_dir}/{tax}_compare.png", dpi=160, bbox_inches="tight")
            plt.close()
            print(f"[{tax}] v2 CD={cd_old:.2f} | fix CD={cd_new:.2f} | delta={delta:+.2f}")
            done[tax] += 1

    print(f"\nDone! Output in {out_dir}/")


if __name__ == "__main__":
    main()
