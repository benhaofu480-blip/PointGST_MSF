"""
PCN 数据集示意图（图1）：1 个完整点云 + 8 个残缺视角。
绘制逻辑与 vis_stage1_sigmoid_vs_pcsa_paper / vis_hard_ft_before_after 一致：
  3D scatter、_limits(pad=0.55)、subsample、depthshade=False。

用法:
  cd completion && python Vision/plot_pcn_dataset_fig1.py
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from matplotlib.gridspec import GridSpec

_COMPLETION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _COMPLETION_ROOT not in sys.path:
    sys.path.insert(0, _COMPLETION_ROOT)

PCN_ROOT = os.path.join(_COMPLETION_ROOT, "data", "PCN")
VISION_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# 与 vis_stage1 输入灰 / 论文 G.T. 橙黄
COLOR_GT = "#E8A020"
COLOR_PARTIAL = "#7f7f7f"

# 与 vis_stage1 / vis_compare_sigmoid_vs_pcsa 相同
VIEW_ELEV = 18
VIEW_AZIM = 45

CATEGORY_CN = {
    "02691156": "飞机",
    "02933112": "橱柜",
    "02958343": "汽车",
    "03001627": "椅子",
    "03636649": "台灯",
    "04256520": "沙发",
    "04379243": "桌子",
    "04530566": "船只",
}

DEFAULT_TAXONOMY = "03001627"
DEFAULT_MODEL = "1015e71a0d21b127de03ab2a27ba7531"

# 与 Vision/README 图3 顶栏一致（子图标题按比例略小）
TITLE_FONT_SIZE = 24
FONT_WEIGHT = "bold"
SUPTITLE_FONT_SIZE = 32


def _set_chinese_font(preferred: str | None = None) -> str:
    candidates = [
        preferred,
        "AR PL UMing CN",
        "SimSun",
        "Noto Serif CJK SC",
        "WenQuanYi Micro Hei",
        "Noto Sans CJK SC",
    ]
    candidates = [c for c in candidates if c]
    for p in (
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    ):
        if os.path.exists(p):
            try:
                fm.fontManager.addfont(p)
            except Exception:
                pass
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Serif"]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["axes.unicode_minus"] = False
    return "default"


def _read_pcd(path: str) -> np.ndarray:
    pc = o3d.io.read_point_cloud(path)
    return np.asarray(pc.points, dtype=np.float32)


def _subsample(points: np.ndarray, max_num: int, seed: int) -> np.ndarray:
    if points.shape[0] <= max_num:
        return points
    rng = np.random.RandomState(seed)
    idx = rng.choice(points.shape[0], max_num, replace=False)
    return points[idx]


def _limits(pts_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """与 vis_stage1_sigmoid_vs_pcsa_paper._limits 相同。"""
    all_pts = np.concatenate(pts_list, axis=0)
    c = all_pts.mean(0)
    s = float((all_pts.max(0) - all_pts.min(0)).max())
    pad = max(s * 0.55, 1e-3)
    return c - pad, c + pad


def _plot_pc(
    ax,
    pts: np.ndarray,
    color: str,
    size: float,
    elev: float,
    azim: float,
    lim_min: np.ndarray,
    lim_max: np.ndarray,
    title: str = "",
):
    """与 vis_stage1._plot_pc / vis_hard_ft._plot_ax 相同。"""
    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        c=color,
        s=size,
        alpha=0.92,
        depthshade=False,
        linewidths=0,
    )
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(lim_min[0], lim_max[0])
    ax.set_ylim(lim_min[1], lim_max[1])
    ax.set_zlim(lim_min[2], lim_max[2])
    ax.set_axis_off()
    ax.grid(False)
    if title:
        ax.set_title(
            title,
            fontsize=TITLE_FONT_SIZE,
            fontweight=FONT_WEIGHT,
            pad=6,
        )


def load_sample(split: str, taxonomy_id: str, model_id: str, n_views: int = 8):
    base = os.path.join(PCN_ROOT, split)
    gt_path = os.path.join(base, "complete", taxonomy_id, f"{model_id}.pcd")
    partial_paths = [
        os.path.join(base, "partial", taxonomy_id, model_id, f"{i:02d}.pcd")
        for i in range(n_views)
    ]
    if not os.path.exists(gt_path):
        raise FileNotFoundError(gt_path)
    gt = _read_pcd(gt_path)
    partials = []
    for p in partial_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        partials.append(_read_pcd(p))
    return gt, partials


def draw_figure(
    gt: np.ndarray,
    partials: list[np.ndarray],
    category_cn: str,
    save_path: str,
    elev: float = VIEW_ELEV,
    azim: float = VIEW_AZIM,
    dpi: int = 220,
):
    # subsample 与 stage1 一致：GT 6000，partial 最多 2048（训练输入规模）
    gt_vis = _subsample(gt, 6000, seed=0)
    partial_vis = [_subsample(p, 2048, seed=10 + i) for i, p in enumerate(partials)]
    lim_min, lim_max = _limits([gt_vis] + partial_vis)

    fig = plt.figure(figsize=(16.0, 10.0), facecolor="white")
    gs = GridSpec(
        3,
        4,
        figure=fig,
        height_ratios=[1.35, 1.0, 1.0],
        hspace=0.04,
        wspace=0.05,
    )

    ax_gt = fig.add_subplot(gs[0, :], projection="3d")
    _plot_pc(
        ax_gt,
        gt_vis,
        COLOR_GT,
        size=2.0,
        elev=elev,
        azim=azim,
        lim_min=lim_min,
        lim_max=lim_max,
        title="完整点云  G.T.",
    )

    for i in range(8):
        row = 1 + i // 4
        col = i % 4
        ax = fig.add_subplot(gs[row, col], projection="3d")
        _plot_pc(
            ax,
            partial_vis[i],
            COLOR_PARTIAL,
            size=4.0,
            elev=elev,
            azim=azim,
            lim_min=lim_min,
            lim_max=lim_max,
            title=f"残缺视角 {i + 1}",
        )

    fig.suptitle(
        f"PCN 数据集示例：{category_cn}",
        fontsize=SUPTITLE_FONT_SIZE,
        fontweight=FONT_WEIGHT,
        y=0.98,
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="PCN 图1（3D scatter，与项目 vis 脚本一致）")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--taxonomy_id", type=str, default=DEFAULT_TAXONOMY)
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(VISION_OUT, "pcn_fig1_complete_and_8partials.png"),
    )
    parser.add_argument("--font", type=str, default="AR PL UMing CN")
    parser.add_argument("--elev", type=float, default=VIEW_ELEV)
    parser.add_argument("--azim", type=float, default=VIEW_AZIM)
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    font_name = _set_chinese_font(preferred=args.font or None)
    category_cn = CATEGORY_CN.get(args.taxonomy_id, args.taxonomy_id)
    gt, partials = load_sample(args.split, args.taxonomy_id, args.model_id)
    draw_figure(
        gt,
        partials,
        category_cn,
        args.output,
        elev=args.elev,
        azim=args.azim,
        dpi=args.dpi,
    )
    print(f"字体: {font_name}")
    print(f"视角: elev={args.elev}, azim={args.azim}")
    print(f"绘制: 3D scatter  GT s=2.0  partial s=4.0  (同 vis_stage1/vis_hard_ft)")
    print(f"已保存: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
