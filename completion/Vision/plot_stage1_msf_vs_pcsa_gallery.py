"""
一阶段 MSF vs PCSA：多样本画廊，每个物体单独一张图（4 列：输入 / PCSA / MSF / G.T.）
不计算、不显示 CD。默认排除此前 8+8 张对比图用过的样本。

用法:
  cd completion
  python Vision/plot_stage1_msf_vs_pcsa_gallery.py
  python Vision/plot_stage1_msf_vs_pcsa_gallery.py --per-category 4 --elev 12 --azim -120
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch
_COMPLETION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _COMPLETION_ROOT not in sys.path:
    sys.path.insert(0, _COMPLETION_ROOT)

from datasets.io import IO
from tools import builder
from utils.config import get_config

VISION_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "stage1_gallery")

ALL_TAXONOMIES = [
    "02691156",
    "02933112",
    "02958343",
    "03001627",
    "03636649",
    "04256520",
    "04379243",
    "04530566",
]

TAXONOMY_CN = {
    "02691156": "飞机",
    "02933112": "橱柜",
    "02958343": "汽车",
    "03001627": "椅子",
    "03636649": "台灯",
    "04256520": "沙发",
    "04379243": "桌子",
    "04530566": "船只",
}

CFG_MSF = "cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid_complete.yaml"
CFG_PCSA = "cfgs/PCN_models/AdaPoinTr_PCSA_complete.yaml"
CKPT_MSF = (
    "experiments/AdaPoinTr_MSF_Pure_Group_sigmoid_complete/PCN_models/"
    "exp_MSF_Pure_Group_sigmoid_complete_seed42/ckpt-best.pth"
)
CKPT_PCSA = (
    "experiments/AdaPoinTr_PCSA_complete/PCN_models/"
    "exp_PCSA_complete_seed42/ckpt-best.pth"
)

BANNED_LISTS = [
    "data/stage1_complete_vis_8cat.txt",
    "data/stage1_complete_vis_8cat_alt.txt",
    "data/stage1_complete_vis_8cat_rescan.txt",
]

# 与 vis_stage1_sigmoid_vs_pcsa_paper 第一行完全一致
VIEWS = [(18, 45)]
LIMIT_PAD_RATIO = 0.55

COLORS = {"input": "#7f7f7f", "pcsa": "#1f77b4", "msf": "#d62728", "gt": "#2ca02c"}
SIZES = {"input": 2.5, "pcsa": 4.5, "msf": 4.5, "gt": 2.0}
COLUMN_TITLE_FONTSIZE = 26
PANEL_FIGSIZE = (5.0, 5.4)
PANEL_DPI = 96
PANEL_AX_BOX = [0.03, 0.04, 0.94, 0.80]
PANEL_TITLE_Y = 0.87
COL_GAP_PX = 36
STITCH_TOP_PAD_PX = 10


class VisArgs:
    launcher = "none"
    local_rank = 0
    distributed = False
    use_gpu = True
    num_workers = 4
    test = True
    model = "pgst"
    ckpts = ""
    start_ckpts = ""
    resume = False
    exp_name = ""
    experiment_path = ""
    tfboard_path = ""
    log_name = "vis_stage1_gallery"
    seed = 0
    deterministic = False
    sync_bn = False
    mode = None
    config = ""
    val_freq = 10


def _set_chinese_font():
    path = "/usr/share/fonts/truetype/arphic/uming.ttc"
    if os.path.exists(path):
        plt.rcParams["font.sans-serif"] = ["AR PL UMing CN", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False


def _build_model(cfg_path: str, ckpt_path: str, exp_root: str, adapter_mode: str):
    args = VisArgs()
    args.config = cfg_path
    args.ckpts = ckpt_path
    args.experiment_path = exp_root
    args.tfboard_path = os.path.join(exp_root, "TFBoard")
    os.makedirs(args.experiment_path, exist_ok=True)
    cfg = get_config(args, logger=None)
    cfg.model.NAME = "AdaPoinTr_PGST"
    cfg.model.encoder_config.adapter_mode = adapter_mode
    cfg.model.encoder_config.pop("use_msf", None)
    cfg.model.msf_route_mode = "none"
    cfg.model.use_msf_route_to_decoder = False
    model = builder.model_builder(cfg.model)
    builder.load_model(model, ckpt_path, logger=None)
    return model, cfg, args


def _to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _load_gt_for_vis(root: str, tax: str, model_id: str) -> np.ndarray:
    """从原始 complete PCD 读 GT，避免子集列表下 idx 与 cache_pcn 错位。"""
    path = os.path.join(root, "data", "PCN", "test", "complete", tax, f"{model_id}.pcd")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"GT PCD missing: {path}")
    return IO.get(path).astype(np.float32)


def _load_partial_for_vis(root: str, tax: str, model_id: str, view_id: int = 0) -> np.ndarray:
    """test 集固定视角 00（与 PCNDataset test rand_idx=0 一致）。"""
    path = os.path.join(
        root, "data", "PCN", "test", "partial", tax, model_id, f"{view_id:02d}.pcd"
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"partial PCD missing: {path}")
    return IO.get(path).astype(np.float32)


def _to_tax(v):
    if torch.is_tensor(v):
        return str(int(v.item())) if v.dtype == torch.int64 else str(v.item())
    return str(v)


def _subsample(pts, n, seed):
    if pts.shape[0] <= n:
        return pts
    rng = np.random.RandomState(seed)
    return pts[rng.choice(pts.shape[0], n, replace=False)]


def _limits(pts_list, pad_ratio: float = LIMIT_PAD_RATIO):
    all_pts = np.concatenate(pts_list, axis=0)
    c = all_pts.mean(0)
    s = float((all_pts.max(0) - all_pts.min(0)).max())
    pad = max(s * pad_ratio, 1e-3)
    return c - pad, c + pad


def _crop_white_border(img: np.ndarray, thresh: int = 252) -> np.ndarray:
    """裁掉 3D 画布四周白边（仅点云区域，不含标题）。"""
    mask = np.any(img < thresh, axis=2)
    if not mask.any():
        return img
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return img[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _plot_pc(
    ax,
    pts,
    color,
    size,
    elev,
    azim,
    lim_min,
    lim_max,
    title="",
    title_fontsize: float = COLUMN_TITLE_FONTSIZE,
):
    """与 vis_stage1_sigmoid_vs_pcsa_paper._plot_pc 一致，列标题加大。"""
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
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass


def _load_picks_from_file(path: str) -> list[tuple[str, str]]:
    picks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tax, mid = line.split("/", 1)
            picks.append((tax, mid))
    return picks


def _load_banned(root: str) -> set[str]:
    banned: set[str] = set()
    for rel in BANNED_LISTS:
        path = os.path.join(root, rel)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                _, mid = line.split("/", 1)
                banned.add(mid)
    return banned


def build_gallery_picks(root: str, per_category: int, banned: set[str]) -> list[tuple[str, str]]:
    pcn_json = os.path.join(root, "data", "PCN", "PCN.json")
    with open(pcn_json, "r", encoding="utf-8") as f:
        taxonomies = json.load(f)

    by_id = {t["taxonomy_id"]: t for t in taxonomies}
    picks: list[tuple[str, str]] = []

    for tax in ALL_TAXONOMIES:
        test_ids = list(by_id[tax]["test"])
        candidates = [m for m in test_ids if m not in banned]
        if len(candidates) < per_category:
            raise RuntimeError(
                f"{tax}: only {len(candidates)} candidates after ban ({per_category} needed)"
            )
        n = len(candidates)
        # 在 test 集上均匀取点，避免都挤在同一难度段
        indices = [
            int(n * (i + 1) / (per_category + 1))
            for i in range(per_category)
        ]
        for idx in indices:
            picks.append((tax, candidates[idx]))
    return picks


def _save_picks_file(root: str, picks: list[tuple[str, str]], rel_path: str):
    path = os.path.join(root, rel_path)
    lines = ["# gallery picks (exclude previous 8cat figures)"]
    for tax, mid in picks:
        lines.append(f"{tax}/{mid}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


@torch.no_grad()
def _infer_one(model_msf, model_pcsa, partial, gt_t, gt, tax, model_id):
    pred_m = model_msf(partial)[-1][0]
    pred_p = model_pcsa(partial)[-1][0]
    return {
        "tax": tax,
        "model_id": str(model_id),
        "partial": _to_np(partial[0]),
        "gt": gt,
        "pred_msf": _to_np(pred_m),
        "pred_pcsa": _to_np(pred_p),
    }


def _render_square_panel(
    pts: np.ndarray,
    color: str,
    size: float,
    elev: float,
    azim: float,
    lim_min: np.ndarray,
    lim_max: np.ndarray,
    title: str,
    panel_dpi: int = PANEL_DPI,
    title_fontsize: float = COLUMN_TITLE_FONTSIZE,
) -> np.ndarray:
    """单图：标题紧贴 3D 区上方，固定画布尺寸。"""
    _set_chinese_font()
    fig = plt.figure(figsize=PANEL_FIGSIZE, facecolor="white", dpi=panel_dpi)
    if title:
        fig.text(
            0.5, PANEL_TITLE_Y, title,
            ha="center", va="bottom",
            fontsize=title_fontsize, fontweight="bold", color="black",
        )
    ax = fig.add_axes(PANEL_AX_BOX, projection="3d")
    _plot_pc(ax, pts, color, size, elev, azim, lim_min, lim_max)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((h, w, 4))
    plt.close(fig)
    return buf[:, :, :3].copy()


def draw_one(
    sample: dict,
    out_path: str,
    dpi: int = 220,
    views: list[tuple[float, float]] | None = None,
    pad_ratio: float = LIMIT_PAD_RATIO,
    title_fontsize: float = COLUMN_TITLE_FONTSIZE,
    col_gap: int = COL_GAP_PX,
):
    """4 列横拼：无 model_id，大列标题，裁白边后紧凑拼接。"""
    views = views or VIEWS
    seed = hash(sample["model_id"]) % 10000
    partial = _subsample(sample["partial"], 2048, seed)
    gt = _subsample(sample["gt"], 6000, seed + 1)
    pred_p = _subsample(sample["pred_pcsa"], 6000, seed + 2)
    pred_m = _subsample(sample["pred_msf"], 6000, seed + 3)
    lim_min, lim_max = _limits([partial, gt, pred_p, pred_m], pad_ratio=pad_ratio)

    elev, azim = views[0]

    cols = [
        ("输入", partial, COLORS["input"], SIZES["input"]),
        ("PCSA", pred_p, COLORS["pcsa"], SIZES["pcsa"]),
        ("MSF", pred_m, COLORS["msf"], SIZES["msf"]),
        ("G.T.", gt, COLORS["gt"], SIZES["gt"]),
    ]

    panels = [
        _render_square_panel(
            pts, color, size, elev, azim, lim_min, lim_max, title,
            title_fontsize=title_fontsize,
        )
        for title, pts, color, size in cols
    ]
    h_max = max(p.shape[0] for p in panels)
    gap = np.ones((h_max, col_gap, 3), dtype=np.uint8) * 255
    padded = []
    for i, p in enumerate(panels):
        if p.shape[0] < h_max:
            pad = np.ones((h_max - p.shape[0], p.shape[1], 3), dtype=np.uint8) * 255
            p = np.concatenate([p, pad], axis=0)
        padded.append(p)
        if i < len(panels) - 1:
            padded.append(gap)
    stitched = np.concatenate(padded, axis=1)
    if STITCH_TOP_PAD_PX > 0:
        top = np.ones((STITCH_TOP_PAD_PX, stitched.shape[1], 3), dtype=np.uint8) * 255
        stitched = np.concatenate([top, stitched], axis=0)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig = plt.figure(
        figsize=(stitched.shape[1] / dpi, stitched.shape[0] / dpi),
        dpi=dpi,
        facecolor="white",
    )
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(stitched.astype(np.uint8))
    ax.axis("off")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-category", type=int, default=3, help="每类生成几张（默认 3，共 24 张）")
    parser.add_argument(
        "--picks-file",
        type=str,
        default="",
        help="沿用已有列表（仅重绘视角时用），默认自动生成 data/stage1_complete_vis_gallery.txt",
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--title-fontsize",
        type=float,
        default=COLUMN_TITLE_FONTSIZE,
        help="列标题字号（输入/PCSA/MSF/G.T.）",
    )
    parser.add_argument(
        "--col-gap",
        type=int,
        default=COL_GAP_PX,
        help="列与列之间的空白宽度（像素）",
    )
    parser.add_argument(
        "--pad-ratio",
        type=float,
        default=LIMIT_PAD_RATIO,
        help="坐标轴留白比例，越小物体越大（vis_exp2≈0.06, 旧画廊误用效果≈0.55+扁画布）",
    )
    parser.add_argument(
        "--out-dir",
        default=VISION_OUT,
    )
    cli = parser.parse_args()

    _set_chinese_font()
    root = _COMPLETION_ROOT

    if cli.picks_file:
        picks_rel = cli.picks_file
        picks = _load_picks_from_file(os.path.join(root, picks_rel))
        picks_path = os.path.join(root, picks_rel)
        print(f"reuse picks ({len(picks)}): {picks_path}")
    else:
        banned = _load_banned(root)
        print(f"excluded {len(banned)} model_ids from previous 8cat figures")
        picks = build_gallery_picks(root, cli.per_category, banned)
        picks_rel = "data/stage1_complete_vis_gallery.txt"
        picks_path = _save_picks_file(root, picks, picks_rel)
        print(f"gallery picks: {len(picks)} -> {picks_path}")

    ckpt_msf = os.path.join(root, CKPT_MSF)
    ckpt_pcsa = os.path.join(root, CKPT_PCSA)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  views={VIEWS}  pad_ratio={cli.pad_ratio}")

    print("Loading MSF...")
    model_msf, cfg, args = _build_model(
        os.path.join(root, CFG_MSF), ckpt_msf,
        os.path.join(root, "tmp_vis_gallery_msf"),
        "msf_pure_group_sigmoid",
    )
    time.sleep(0.3)
    print("Loading PCSA...")
    model_pcsa, _, _ = _build_model(
        os.path.join(root, CFG_PCSA), ckpt_pcsa,
        os.path.join(root, "tmp_vis_gallery_pcsa"),
        "pcsa",
    )
    model_msf = model_msf.to(device).eval()
    model_pcsa = model_pcsa.to(device).eval()

    out_dir = cli.out_dir
    os.makedirs(out_dir, exist_ok=True)
    index_lines = ["# tax\tname\tmodel_id\tpng"]

    for i, (tax, model_id) in enumerate(picks):
        partial_np = _load_partial_for_vis(root, tax, model_id, view_id=0)
        partial = torch.from_numpy(partial_np).unsqueeze(0).float().to(device)
        gt = _load_gt_for_vis(root, tax, model_id)
        rec = _infer_one(model_msf, model_pcsa, partial, None, gt, tax, model_id)

        name = TAXONOMY_CN.get(tax, tax)
        fname = f"{tax}_{name}_{model_id}.png"
        out_png = os.path.join(out_dir, fname)
        draw_one(
            rec, out_png, dpi=cli.dpi, pad_ratio=cli.pad_ratio,
            title_fontsize=cli.title_fontsize, col_gap=cli.col_gap,
        )
        index_lines.append(f"{tax}\t{name}\t{model_id}\t{fname}")
        print(f"  [{i + 1}/{len(picks)}] {name}  {model_id}  -> {fname}")

    index_path = os.path.join(out_dir, "INDEX.txt")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("\n".join(index_lines) + "\n")
    print(f"Done. {len(picks)} figures in {os.path.abspath(out_dir)}")
    print(f"index: {index_path}")


if __name__ == "__main__":
    main()
