"""
KITTI 可视化：一阶段 MSF Sigmoid（PCN 完整数据集 ckpt-best）。
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_COMPLETION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _COMPLETION_ROOT not in sys.path:
    sys.path.insert(0, _COMPLETION_ROOT)

from tools import builder
from utils import misc
from utils.config import get_config

CFG = "cfgs/KITTI_models/AdaPoinTr_MSF_Pure_Group_sigmoid_complete.yaml"
CKPT = (
    "experiments/AdaPoinTr_MSF_Pure_Group_sigmoid_complete/PCN_models/"
    "exp_MSF_Pure_Group_sigmoid_complete_seed42/ckpt-best.pth"
)
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "kitti_msf_complete")
DEFAULT_INDICES = [0, 120, 240, 360, 480, 600, 720, 840, 960, 1080, 1200, 1320]

COLORS = {"input": "#4C72B0", "pred": "#C44E52"}
SIZES = {"input": 3.0, "pred": 2.5}


class VisArgs:
    launcher = "none"
    local_rank = 0
    distributed = False
    use_gpu = True
    num_workers = 0
    test = True
    model = "pgst"
    ckpts = ""
    start_ckpts = ""
    resume = False
    exp_name = "kitti_msf_complete_vis"
    experiment_path = ""
    tfboard_path = ""
    log_name = "vis_kitti_msf_complete"
    seed = 42
    deterministic = False
    sync_bn = False
    mode = None
    config = ""
    val_freq = 10


def _to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _pred_for_vis(dense: np.ndarray) -> np.ndarray:
    if dense.shape[0] > 2048:
        return dense[:-2048]
    return dense


def _subsample(pts: np.ndarray, n: int, seed: int) -> np.ndarray:
    if pts.shape[0] <= n:
        return pts
    rng = np.random.RandomState(seed)
    return pts[rng.choice(pts.shape[0], n, replace=False)]


def _limits(pts_list):
    all_pts = np.concatenate(pts_list, axis=0)
    c = all_pts.mean(0)
    s = float((all_pts.max(0) - all_pts.min(0)).max())
    pad = max(s * 0.55, 1e-3)
    return c - pad, c + pad


def _plot_pc(ax, pts, color, size, elev, azim, lim_min, lim_max, title=""):
    ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2],
        c=color, s=size, alpha=0.92, depthshade=False, linewidths=0,
    )
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(lim_min[0], lim_max[0])
    ax.set_ylim(lim_min[1], lim_max[1])
    ax.set_zlim(lim_min[2], lim_max[2])
    ax.set_axis_off()
    ax.grid(False)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=2)


def _build_model_and_loader(root: str, device: torch.device):
    args = VisArgs()
    args.config = CFG
    args.ckpts = CKPT
    args.experiment_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "_tmp_kitti_msf_vis")
    args.tfboard_path = os.path.join(args.experiment_path, "TFBoard")
    os.makedirs(args.experiment_path, exist_ok=True)

    cfg = get_config(args, logger=None)
    cfg.model.NAME = "AdaPoinTr_PGST"
    cfg.model.encoder_config.adapter_mode = "msf_pure_group_sigmoid"
    cfg.model.encoder_config.pop("use_msf", None)
    cfg.model.msf_route_mode = "none"
    cfg.model.use_msf_route_to_decoder = False

    model = builder.model_builder(cfg.model)
    builder.load_model(model, os.path.join(root, CKPT), logger=None)
    model.to(device)
    model.eval()

    _, loader = builder.dataset_builder(args, cfg.dataset.test)
    return model, loader


@torch.no_grad()
def _infer_sample(model, partial, device):
    partial = partial.unsqueeze(0).to(device)
    ret = model(partial)
    return _to_np(ret[-1][0])


def _maybe_denorm(pts, meta, denorm: bool):
    if not denorm or meta is None:
        return pts
    return _to_np(misc.denormalize_kitti_point_cloud(pts, meta))


def _save_pair(out_dir: str, model_id: str, partial_vis, pred_vis, elev, azim):
    sample_dir = os.path.join(out_dir, "samples", model_id)
    os.makedirs(sample_dir, exist_ok=True)
    np.save(os.path.join(sample_dir, "input.npy"), partial_vis)
    np.save(os.path.join(sample_dir, "pred.npy"), pred_vis)

    fig = plt.figure(figsize=(12, 5), facecolor="white")
    lim_min, lim_max = _limits([partial_vis, pred_vis])
    for i, (pts, title, key) in enumerate(
        ((partial_vis, "Input (partial)", "input"), (pred_vis, "MSF prediction", "pred"))
    ):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        _plot_pc(ax, pts, COLORS[key], SIZES[key], elev, azim, lim_min, lim_max, title)
    fig.savefig(os.path.join(sample_dir, "pair.png"), dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _draw_gallery(records, out_path: str, elev: float, azim: float, dpi: int):
    n = len(records)
    fig = plt.figure(figsize=(10.0, 3.2 * n), facecolor="white")
    gs = GridSpec(n, 2, figure=fig, hspace=0.06, wspace=0.04)

    for row, rec in enumerate(records):
        partial = _subsample(rec["partial"], 2048, 11 + row)
        pred = _subsample(rec["pred"], 8000, 23 + row)
        lim_min, lim_max = _limits([partial, pred])

        for col, (pts, title, key) in enumerate(
            ((partial, "Input", "input"), (pred, "MSF pred", "pred"))
        ):
            ax = fig.add_subplot(gs[row, col], projection="3d")
            _plot_pc(ax, pts, COLORS[key], SIZES[key], elev, azim, lim_min, lim_max, title)
            if col == 0:
                ax.text2D(
                    0.02, 0.92, rec["model_id"], transform=ax.transAxes, fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="none"),
                )

    fig.suptitle(
        "KITTI — MSF Sigmoid Stage-1 (PCN complete, ckpt-best)",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _parse_indices(cli, ds_len: int) -> list[int]:
    if cli.indices:
        return list(cli.indices)
    if cli.num <= 0:
        return [i for i in DEFAULT_INDICES if i < ds_len]
    step = max(1, ds_len // cli.num)
    return list(range(0, ds_len, step))[: cli.num]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=12)
    ap.add_argument("--indices", type=int, nargs="*", default=None)
    ap.add_argument("--denorm", action="store_true")
    ap.add_argument("--elev", type=float, default=20.0)
    ap.add_argument("--azim", type=float, default=-120.0)
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument("--out", type=str, default=DEFAULT_OUT)
    ap.add_argument("--device", type=str, default="cuda:0")
    cli = ap.parse_args()

    os.makedirs(cli.out, exist_ok=True)
    device = torch.device(cli.device if torch.cuda.is_available() else "cpu")

    print(f"loading {CKPT}")
    model, loader = _build_model_and_loader(_COMPLETION_ROOT, device)
    dataset = loader.dataset
    indices = _parse_indices(cli, len(dataset))
    print(f"KITTI samples: {len(dataset)}, visualize indices: {indices}")

    records = []
    for idx in indices:
        if idx < 0 or idx >= len(dataset):
            print(f"skip invalid index {idx}")
            continue
        taxonomy_id, model_id, data = dataset[idx]
        if isinstance(data, (tuple, list)):
            partial_t, meta = data[0], data[1] if len(data) > 1 else None
        else:
            partial_t, meta = data, None

        dense_np = _infer_sample(model, partial_t, device)
        pred_np = _pred_for_vis(dense_np)
        partial_vis = _maybe_denorm(_to_np(partial_t), meta, cli.denorm)
        pred_vis = _maybe_denorm(pred_np, meta, cli.denorm)

        rec = {
            "idx": idx,
            "taxonomy_id": str(taxonomy_id),
            "model_id": str(model_id),
            "partial": partial_vis,
            "pred": pred_vis,
        }
        records.append(rec)
        _save_pair(cli.out, rec["model_id"], partial_vis, pred_vis, cli.elev, cli.azim)
        print(f"  [{idx:4d}] {model_id}  input={partial_vis.shape[0]} pred={pred_vis.shape[0]}")

    if not records:
        raise SystemExit("no samples rendered")

    gallery_path = os.path.join(cli.out, f"gallery_{len(records)}x2.png")
    _draw_gallery(records, gallery_path, cli.elev, cli.azim, cli.dpi)

    meta_path = os.path.join(cli.out, "samples_list.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("# idx\tmodel_id\n")
        for rec in records:
            f.write(f"{rec['idx']}\t{rec['model_id']}\n")

    print(f"gallery -> {gallery_path}")
    print(f"per-sample -> {os.path.join(cli.out, 'samples')}")
    print(f"index -> {meta_path}")


if __name__ == "__main__":
    main()
