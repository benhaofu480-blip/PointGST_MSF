"""
画廊图 + 全自动局部放大（第二行）

在 MSF 相对 PCSA 更接近 GT 的区域自动裁 3D 框，无需手标像素。

用法:
  cd completion
  python Vision/plot_stage1_gallery_auto_zoom.py \\
    --images Vision/output/stage1_gallery_18_45/03001627_椅子_91b8fe4616208bd4cf752e9bed38184f.png
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import cKDTree

_COMPLETION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _COMPLETION_ROOT not in sys.path:
    sys.path.insert(0, _COMPLETION_ROOT)

from Vision.plot_stage1_msf_vs_pcsa_gallery import (
    CFG_MSF,
    CFG_PCSA,
    CKPT_MSF,
    CKPT_PCSA,
    COLORS,
    COL_GAP_PX,
    COLUMN_TITLE_FONTSIZE,
    SIZES,
    STITCH_TOP_PAD_PX,
    VIEWS,
    VisArgs,
    _build_model,
    _infer_one,
    _limits,
    _load_gt_for_vis,
    _load_partial_for_vis,
    _render_square_panel,
    _set_chinese_font,
    _subsample,
)

DETAIL_PAD_RATIO = 0.12
DETAIL_SIZE_SCALE = 1.35
ROW_GAP_PX = 14
DETAIL_LABEL = "局部放大"


def _parse_gallery_png(path: str) -> tuple[str, str]:
    """03001627_椅子_xxx.png -> (taxonomy_id, model_id)"""
    base = os.path.basename(path).replace(".png", "")
    m = re.match(r"^(\d{8})_.+_([0-9a-f]{32})$", base)
    if not m:
        raise ValueError(f"cannot parse gallery filename: {path}")
    return m.group(1), m.group(2)


def _auto_zoom_box(
    gt: np.ndarray,
    pred_pcsa: np.ndarray,
    pred_msf: np.ndarray,
    top_ratio: float = 0.08,
    min_span: float = 0.12,
    expand: float = 1.35,
) -> tuple[np.ndarray, np.ndarray]:
    """
    在 GT 上找「PCSA 误差 - MSF 误差」最大的区域，返回 (lim_min, lim_max)。
    """
    n = 8192
    seed = 42
    gt_s = _subsample(gt, n, seed)
    p_s = _subsample(pred_pcsa, n, seed + 1)
    m_s = _subsample(pred_msf, n, seed + 2)

    tree_p = cKDTree(p_s)
    tree_m = cKDTree(m_s)
    err_p, _ = tree_p.query(gt_s, k=1)
    err_m, _ = tree_m.query(gt_s, k=1)
    advantage = err_p - err_m

    thr = np.percentile(advantage, 100 * (1 - top_ratio))
    mask = advantage >= max(thr, 0.0)
    if mask.sum() < 32:
        idx = np.argsort(advantage)[-max(32, int(len(gt_s) * top_ratio)) :]
        pts = gt_s[idx]
    else:
        pts = gt_s[mask]

    cmin = pts.min(0)
    cmax = pts.max(0)
    span = cmax - cmin
    span = np.maximum(span, min_span)
    center = (cmin + cmax) * 0.5
    half = span * 0.5 * expand
    lim_min = center - half
    lim_max = center + half
    return lim_min.astype(np.float32), lim_max.astype(np.float32)


def _stitch_row(
    panels: list[np.ndarray],
    col_gap: int,
    top_pad: int = STITCH_TOP_PAD_PX,
) -> np.ndarray:
    h_max = max(p.shape[0] for p in panels)
    gap = np.ones((h_max, col_gap, 3), dtype=np.uint8) * 255
    parts = []
    for i, p in enumerate(panels):
        if p.shape[0] < h_max:
            pad = np.ones((h_max - p.shape[0], p.shape[1], 3), dtype=np.uint8) * 255
            p = np.concatenate([p, pad], axis=0)
        parts.append(p)
        if i < len(panels) - 1:
            parts.append(gap)
    row = np.concatenate(parts, axis=1)
    if top_pad > 0:
        top = np.ones((top_pad, row.shape[1], 3), dtype=np.uint8) * 255
        row = np.concatenate([top, row], axis=0)
    return row


def draw_with_auto_zoom(
    sample: dict,
    out_path: str,
    dpi: int = 220,
    pad_ratio: float = 0.42,
    col_gap: int = COL_GAP_PX,
    title_fontsize: float = COLUMN_TITLE_FONTSIZE,
):
    seed = hash(sample["model_id"]) % 10000
    partial = _subsample(sample["partial"], 2048, seed)
    gt = _subsample(sample["gt"], 6000, seed + 1)
    pred_p = _subsample(sample["pred_pcsa"], 6000, seed + 2)
    pred_m = _subsample(sample["pred_msf"], 6000, seed + 3)

    lim_full_min, lim_full_max = _limits([partial, gt, pred_p, pred_m], pad_ratio=pad_ratio)
    lim_zoom_min, lim_zoom_max = _auto_zoom_box(
        sample["gt"], sample["pred_pcsa"], sample["pred_msf"]
    )

    elev, azim = VIEWS[0]
    sizes_detail = {k: v * DETAIL_SIZE_SCALE for k, v in SIZES.items()}

    def _cols(lim_min, lim_max, sizes, titles):
        data = [
            (titles[0], partial, COLORS["input"], sizes["input"]),
            (titles[1], pred_p, COLORS["pcsa"], sizes["pcsa"]),
            (titles[2], pred_m, COLORS["msf"], sizes["msf"]),
            (titles[3], gt, COLORS["gt"], sizes["gt"]),
        ]
        return [
            _render_square_panel(pts, color, sz, elev, azim, lim_min, lim_max, title, title_fontsize=title_fontsize)
            for title, pts, color, sz in data
        ]

    row_main = _stitch_row(
        _cols(lim_full_min, lim_full_max, SIZES, ("输入", "PCSA", "MSF", "G.T.")),
        col_gap,
    )
    row_detail = _stitch_row(
        _cols(lim_zoom_min, lim_zoom_max, sizes_detail, ("输入", "PCSA", "MSF", "G.T.")),
        col_gap,
        top_pad=0,
    )

    w = max(row_main.shape[1], row_detail.shape[1])
    label_h = 36
    label_band = np.ones((label_h, w, 3), dtype=np.uint8) * 255
    _set_chinese_font()
    fig_l = plt.figure(figsize=(w / 100, label_h / 100), dpi=100, facecolor="white")
    fig_l.text(0.02, 0.5, DETAIL_LABEL, ha="left", va="center", fontsize=18, fontweight="bold")
    fig_l.canvas.draw()
    lh, lw = fig_l.canvas.get_width_height()
    buf = np.asarray(fig_l.canvas.buffer_rgba(), dtype=np.uint8).reshape((lh, lw, 4))[:, :, :3]
    plt.close(fig_l)
    if buf.shape[1] != w:
        pad_r = w - buf.shape[1]
        buf = np.pad(buf, ((0, 0), (0, max(0, pad_r)), (0, 0)), constant_values=255)

    def _pad_width(row_img: np.ndarray) -> np.ndarray:
        if row_img.shape[1] >= w:
            return row_img[:, :w]
        pad = np.ones((row_img.shape[0], w - row_img.shape[1], 3), dtype=np.uint8) * 255
        return np.concatenate([row_img, pad], axis=1)

    row_main = _pad_width(row_main)
    row_detail = _pad_width(row_detail)
    sep = np.ones((ROW_GAP_PX, w, 3), dtype=np.uint8) * 255
    stitched = np.concatenate([row_main, sep, label_band, row_detail], axis=0)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig = plt.figure(figsize=(stitched.shape[1] / dpi, stitched.shape[0] / dpi), dpi=dpi, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(stitched.astype(np.uint8))
    ax.axis("off")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="画廊 PNG 路径（从文件名解析 tax / model_id）",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "output", "stage1_gallery_18_45_zoom"),
    )
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--pad-ratio", type=float, default=0.42)
    parser.add_argument("--col-gap", type=int, default=COL_GAP_PX)
    cli = parser.parse_args()

    _set_chinese_font()
    root = _COMPLETION_ROOT
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Loading MSF / PCSA...")
    model_msf, _, _ = _build_model(
        os.path.join(root, CFG_MSF), os.path.join(root, CKPT_MSF),
        os.path.join(root, "tmp_vis_zoom_msf"), "msf_pure_group_sigmoid",
    )
    time.sleep(0.3)
    model_pcsa, _, _ = _build_model(
        os.path.join(root, CFG_PCSA), os.path.join(root, CKPT_PCSA),
        os.path.join(root, "tmp_vis_zoom_pcsa"), "pcsa",
    )
    model_msf = model_msf.to(device).eval()
    model_pcsa = model_pcsa.to(device).eval()

    os.makedirs(cli.out_dir, exist_ok=True)

    for img_path in cli.images:
        tax, model_id = _parse_gallery_png(img_path)
        partial_np = _load_partial_for_vis(root, tax, model_id, 0)
        partial = torch.from_numpy(partial_np).unsqueeze(0).float().to(device)
        gt = _load_gt_for_vis(root, tax, model_id)
        rec = _infer_one(model_msf, model_pcsa, partial, None, gt, tax, model_id)

        base = os.path.basename(img_path).replace(".png", "")
        out_png = os.path.join(cli.out_dir, f"{base}_zoom.png")
        draw_with_auto_zoom(
            rec, out_png, dpi=cli.dpi, pad_ratio=cli.pad_ratio, col_gap=cli.col_gap,
        )
        print(f"  {base} -> {out_png}")

    print(f"Done -> {os.path.abspath(cli.out_dir)}")


if __name__ == "__main__":
    main()
