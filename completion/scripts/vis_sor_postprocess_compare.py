#!/usr/bin/env python3
"""Compare dense pred before/after statistical outlier removal (test-time SOR)."""
import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from easydict import EasyDict
from utils.config import get_config, cfg_from_yaml_file
from tools import builder
from utils.open3d_postprocess import (
    statistical_outlier_filter_numpy,
    postprocess_dense_points,
)
from extensions.chamfer_dist import ChamferDistanceL1


def chamfer_x1e3(pred, gt):
    cd = ChamferDistanceL1()(pred, gt)
    return float(cd.item() * 1000.0)


def subsample(pts, n, seed):
    if pts.shape[0] <= n:
        return pts
    rng = np.random.default_rng(seed)
    idx = rng.choice(pts.shape[0], n, replace=False)
    return pts[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default="experiments/AdaPoinTr_MSF_Pure_Group_sigmoid/PCN_models/exp_MSF_Pure_Group_sigmoid/ckpt-best.pth",
    )
    parser.add_argument(
        "--cfg",
        default="cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid_test_o3d.yaml",
    )
    parser.add_argument("--taxonomy", default="02933112", help="cabinet (worst delta)")
    parser.add_argument("--sample-idx", type=int, default=0, help="index within taxonomy in test set")
    parser.add_argument("--nb", type=int, default=20)
    parser.add_argument("--std", type=float, default=2.0)
    parser.add_argument(
        "--out-dir",
        default="vis_sor_postprocess_compare",
    )
    args = parser.parse_args()

    os.chdir(ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = cfg_from_yaml_file(args.cfg)
    cfg.model.NAME = "AdaPoinTr_PGST"
    model = builder.model_builder(cfg.model)
    builder.load_model(model, args.ckpt, logger=None)
    model.to(device).eval()

    from tools import builder as ds_builder
    from utils.parser import get_args as _ga

    class A:
        pass

    a = A()
    a.config = args.cfg
    a.launcher = "none"
    a.local_rank = 0
    a.num_workers = 0
    a.seed = 0
    a.deterministic = False
    a.sync_bn = False
    a.exp_name = "vis_sor"
    a.start_ckpts = None
    a.ckpts = args.ckpt
    a.val_freq = 1
    a.model = "pgst"
    a.resume = False
    a.test = True
    a.mode = None
    a.experiment_path = os.path.join(ROOT, "tmp_vis_sor")
    a.tfboard_path = os.path.join(a.experiment_path, "TFBoard")
    a.log_name = "vis_sor"
    a.use_gpu = True
    a.distributed = False

    cfg_ds = get_config(a, logger=None)
    _, test_loader = ds_builder.dataset_builder(a, cfg_ds.dataset.test)

    targets = []
    for idx, (taxonomy_ids, model_ids, data) in enumerate(test_loader):
        tid = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
        if tid != args.taxonomy:
            continue
        if len(targets) < args.sample_idx + 1:
            targets.append((idx, tid, model_ids[0], data))
        if len(targets) > args.sample_idx:
            break

    if not targets:
        raise SystemExit(f"No sample for taxonomy {args.taxonomy}")

    idx, tid, mid, data = targets[args.sample_idx]
    partial = data[0].cuda()
    gt = data[1].cuda()

    with torch.no_grad():
        out = model(partial)
        pred_raw = out[-1].clone()

    pts_np = pred_raw[0].detach().cpu().numpy()
    filt_np = statistical_outlier_filter_numpy(
        pts_np, nb_neighbors=args.nb, std_ratio=args.std, backend="torch_knn",
    )
    keep_ratio = filt_np.shape[0] / pts_np.shape[0]

    pred_pp = postprocess_dense_points(pred_raw, cfg_ds, logger=None)

    cd_raw = chamfer_x1e3(pred_raw, gt)
    cd_pp = chamfer_x1e3(pred_pp, gt)

    os.makedirs(args.out_dir, exist_ok=True)

    # 2x2: raw / filtered-before-fps / after full pipeline / gt
    fig = plt.figure(figsize=(12, 10))
    titles = [
        f"Raw pred  CD×1e3={cd_raw:.2f}",
        f"After SOR only ({keep_ratio*100:.1f}% kept, n={filt_np.shape[0]})",
        f"SOR + FPS→16384  CD×1e3={cd_pp:.2f}",
        "GT",
    ]
    clouds = [
        subsample(pts_np, 6000, 1),
        subsample(filt_np, 6000, 2),
        subsample(pred_pp[0].detach().cpu().numpy(), 6000, 3),
        subsample(gt[0].detach().cpu().numpy(), 6000, 4),
    ]
    all_pts = np.concatenate(clouds, axis=0)
    c = all_pts.mean(0)
    s = (all_pts.max(0) - all_pts.min(0)).max() * 0.55
    lim = (c - s, c + s)

    for i, (pts, title) in enumerate(zip(clouds, titles)):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="C0", s=0.4, alpha=0.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(lim[0][0], lim[1][0])
        ax.set_ylim(lim[0][1], lim[1][1])
        ax.set_zlim(lim[0][2], lim[1][2])
        ax.set_axis_off()

    fig.suptitle(
        f"SOR postprocess  tax={tid}  model={mid}  nb={args.nb} std={args.std}\n"
        f"ΔCD = {cd_pp - cd_raw:+.2f} (×1e3)",
        fontsize=12,
    )
    fig.tight_layout()
    png = os.path.join(args.out_dir, f"{tid}_{mid[:8]}_sor_compare.png")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {png}")
    print(f"keep_ratio_after_sor={keep_ratio:.3f}  cd_raw={cd_raw:.3f}  cd_pp={cd_pp:.3f}")


if __name__ == "__main__":
    main()
