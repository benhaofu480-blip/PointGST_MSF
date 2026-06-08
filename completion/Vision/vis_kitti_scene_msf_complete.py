"""
KITTI 场景级可视化：同一 frame 内多辆车分别补全，反归一化后拼回场景坐标。

权重：一阶段 MSF Sigmoid（PCN complete, ckpt-best）

用法:
  python vis_kitti_scene_msf_complete.py --frames frame_0 frame_100
  python vis_kitti_scene_msf_complete.py --frames frame_0 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_COMPLETION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _COMPLETION_ROOT not in sys.path:
    sys.path.insert(0, _COMPLETION_ROOT)

from datasets import data_transforms
from datasets.io import IO
from tools import builder
from utils import misc
from utils.config import get_config

CFG = "cfgs/KITTI_models/AdaPoinTr_MSF_Pure_Group_sigmoid_complete.yaml"
CKPT = (
    "experiments/AdaPoinTr_MSF_Pure_Group_sigmoid_complete/PCN_models/"
    "exp_MSF_Pure_Group_sigmoid_complete_seed42/ckpt-best.pth"
)
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "kitti_msf_scenes")
FRAME_RE = re.compile(r"^(frame_\d+)_car_(\d+)$")

KITTI_TRANSFORMS = data_transforms.Compose([
    {
        "callback": "NormalizeObjectPose",
        "parameters": {
            "input_keys": {"ptcloud": "partial_cloud", "bbox": "bounding_box"},
            "unit_sphere": True,
            "return_meta": True,
            "meta_key": "normalize_meta",
        },
        "objects": ["partial_cloud", "bounding_box"],
    },
    {
        "callback": "RandomSamplePoints",
        "parameters": {"n_points": 2048},
        "objects": ["partial_cloud"],
    },
    {
        "callback": "ToTensor",
        "objects": ["partial_cloud", "bounding_box"],
    },
])


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
    exp_name = "kitti_msf_scene_vis"
    experiment_path = ""
    tfboard_path = ""
    log_name = "vis_kitti_scene"
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
    pad = max(s * 0.12, 1e-3)
    return c - pad, c + pad


def _plot_scene(ax, pts, colors, elev, azim, lim_min, lim_max, title=""):
    for arr, c in zip(pts, colors):
        if arr.shape[0] == 0:
            continue
        ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], c=c, s=1.8, alpha=0.9, depthshade=False, linewidths=0)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(lim_min[0], lim_max[0])
    ax.set_ylim(lim_min[1], lim_max[1])
    ax.set_zlim(lim_min[2], lim_max[2])
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=2)


def _load_kitti_index(root: str) -> list[str]:
    with open(os.path.join(root, "data/KITTI/KITTI.json"), "r", encoding="utf-8") as f:
        return json.load(f)[0]["test"]


def _group_by_frame(samples: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for sid in samples:
        m = FRAME_RE.match(sid)
        if not m:
            continue
        frame = m.group(1)
        groups.setdefault(frame, []).append(sid)
    for frame in groups:
        groups[frame].sort(key=lambda s: int(FRAME_RE.match(s).group(2)))
    return groups


def _load_car_sample(root: str, sample_id: str):
    cloud_path = os.path.join(root, "data/KITTI/cars", f"{sample_id}.pcd")
    bbox_path = os.path.join(root, "data/KITTI/bboxes", f"{sample_id}.txt")
    data = {
        "partial_cloud": IO.get(cloud_path).astype(np.float32),
        "bounding_box": IO.get(bbox_path).astype(np.float32),
    }
    data = KITTI_TRANSFORMS(data)
    meta = data.pop("normalize_meta", None)
    partial = data["partial_cloud"]
    return partial, meta


def _build_model(root: str, device: torch.device):
    args = VisArgs()
    args.config = CFG
    args.ckpts = CKPT
    args.experiment_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "_tmp_kitti_scene_vis")
    os.makedirs(args.experiment_path, exist_ok=True)

    cfg = get_config(args, logger=None)
    cfg.model.NAME = "AdaPoinTr_PGST"
    cfg.model.encoder_config.adapter_mode = "msf_pure_group_sigmoid"

    model = builder.model_builder(cfg.model)
    builder.load_model(model, os.path.join(root, CKPT), logger=None)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def _infer(model, partial_t, device):
    out = model(partial_t.unsqueeze(0).to(device))
    return _pred_for_vis(_to_np(out[-1][0]))


def _denorm(pts, meta):
    return _to_np(misc.denormalize_kitti_point_cloud(pts, meta))


def _process_frame(model, root, frame_id, car_ids, device, out_dir, elev, azim, dpi):
    palette = plt.cm.tab10(np.linspace(0, 1, max(10, len(car_ids))))

    scene_input = []
    scene_pred = []
    car_records = []

    for i, sid in enumerate(car_ids):
        partial_t, meta = _load_car_sample(root, sid)
        pred_norm = _infer(model, partial_t, device)
        partial_world = _denorm(_to_np(partial_t), meta)
        pred_world = _denorm(pred_norm, meta)
        color = palette[i % len(palette)]

        scene_input.append(partial_world)
        scene_pred.append(pred_world)
        car_records.append({
            "sample_id": sid,
            "color": color,
            "partial": partial_world,
            "pred": pred_world,
        })

        car_dir = os.path.join(out_dir, frame_id, sid)
        os.makedirs(car_dir, exist_ok=True)
        np.save(os.path.join(car_dir, "input_world.npy"), partial_world)
        np.save(os.path.join(car_dir, "pred_world.npy"), pred_world)

    lim_min, lim_max = _limits(scene_input + scene_pred)

    fig = plt.figure(figsize=(12, 5.5), facecolor="white")
    ax0 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1 = fig.add_subplot(1, 2, 2, projection="3d")
    colors = [r["color"] for r in car_records]
    _plot_scene(ax0, scene_input, colors, elev, azim, lim_min, lim_max, f"{frame_id} — partial (scene)")
    _plot_scene(ax1, scene_pred, colors, elev, azim, lim_min, lim_max, f"{frame_id} — MSF completion (scene)")
    fig.suptitle(
        f"KITTI scene | {len(car_ids)} cars | MSF Sigmoid Stage-1 (PCN complete)",
        fontsize=13, fontweight="bold",
    )
    png_path = os.path.join(out_dir, f"{frame_id}_scene.png")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    merged_in = np.concatenate(scene_input, axis=0)
    merged_pred = np.concatenate(scene_pred, axis=0)
    np.save(os.path.join(out_dir, f"{frame_id}_scene_input_world.npy"), merged_in)
    np.save(os.path.join(out_dir, f"{frame_id}_scene_pred_world.npy"), merged_pred)

    with open(os.path.join(out_dir, f"{frame_id}_cars.txt"), "w", encoding="utf-8") as f:
        for sid in car_ids:
            f.write(sid + "\n")

    print(f"  {frame_id}: {len(car_ids)} cars -> {png_path}")
    return png_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", nargs="+", default=["frame_0"], help="如 frame_0 frame_100")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--elev", type=float, default=25.0)
    ap.add_argument("--azim", type=float, default=-115.0)
    ap.add_argument("--dpi", type=int, default=180)
    cli = ap.parse_args()

    os.makedirs(cli.out, exist_ok=True)
    device = torch.device(cli.device if torch.cuda.is_available() else "cpu")

    samples = _load_kitti_index(_COMPLETION_ROOT)
    groups = _group_by_frame(samples)

    print(f"loading MSF complete ckpt on {device}")
    model = _build_model(_COMPLETION_ROOT, device)

    done = []
    for frame_id in cli.frames:
        car_ids = groups.get(frame_id)
        if not car_ids:
            print(f"skip unknown/empty frame: {frame_id}")
            continue
        done.append(_process_frame(
            model, _COMPLETION_ROOT, frame_id, car_ids, device,
            cli.out, cli.elev, cli.azim, cli.dpi,
        ))

    if not done:
        raise SystemExit("no scene rendered")
    print(f"done. outputs under {cli.out}")


if __name__ == "__main__":
    main()
