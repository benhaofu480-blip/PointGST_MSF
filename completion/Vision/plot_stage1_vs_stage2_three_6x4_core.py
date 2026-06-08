"""
PCN_core：指定 3 例 → 6 行×4 列（输入 / Stage-1 / Stage-2 / G.T.），布局同 pick_03 橱柜图。

用法:
  cd completion && conda activate pgst
  CUDA_VISIBLE_DEVICES=0 python -u Vision/plot_stage1_vs_stage2_three_6x4_core.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
from PIL import Image

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Vision.plot_stage1_vs_stage2_picks import (
    CFG_STAGE1,
    CFG_STAGE2,
    CKPT_STAGE1,
    CKPT_STAGE2,
    COLORS,
    SAVE_DPI,
    SIZES,
    _build_model,
    _limits,
    _load_gt_for_vis,
    _load_partial_for_vis,
    _render_header,
    _render_square_panel,
    _set_chinese_font,
    _stitch_row,
    _subsample,
)

# 固定 /tmp 实验目录，二次运行可少写 config、略快
TMP_VIS_ROOT = "/tmp/vis_three_6x4_pcn_core"
STITCH_TOP_PAD_PX = 10
ROW_GAP = 8  # 原 pick 用 12，略收紧
HEADER_GAP = 10

VIEWS_2 = [(18, 45), (10, 90)]
# 原 0.42，略放大
PAD_RATIO = 0.36
# 点略大（相对 gallery 默认）
SIZES_USE = {
    "input": 2.8,
    "msf": 4.8,
    "gt": 2.2,
}

PICKS = [
    ("椅子", "03001627", "4ea3d680127a9fe91360172b4b6205b1"),
    ("桌子", "04379243", "1dc7f7d076afd0ccf11c3739edd52fa3"),
    ("船只", "04530566", "390de3a1bd0191c881d9d9b1473043a2"),
]

OUT = os.path.join(
    _ROOT, "Vision", "output", "stage1_vs_stage2_pcn_core", "batch15",
    "three_chair_table_boat_6x4.png",
)


def _stitch_body(body_rows: list[np.ndarray]) -> np.ndarray:
    target_w = max(r.shape[1] for r in body_rows)
    padded = []
    for r in body_rows:
        if r.shape[1] < target_w:
            r = np.concatenate(
                [r, np.ones((r.shape[0], target_w - r.shape[1], 3), dtype=np.uint8) * 255],
                axis=1,
            )
        padded.append(r)
    body = padded[0]
    for r in padded[1:]:
        body = np.concatenate(
            [body, np.ones((ROW_GAP, target_w, 3), dtype=np.uint8) * 255, r], axis=0
        )
    w = body.shape[1]
    return np.concatenate(
        [
            np.ones((STITCH_TOP_PAD_PX, w, 3), dtype=np.uint8) * 255,
            _render_header(w),
            np.ones((HEADER_GAP, w, 3), dtype=np.uint8) * 255,
            body,
        ],
        axis=0,
    )


@torch.no_grad()
def _rows_for_sample(root, device, model_s1, model_s2, tax, model_id):
    partial_np = _load_partial_for_vis(root, tax, model_id, 0)
    partial = torch.from_numpy(partial_np).unsqueeze(0).float().to(device)
    gt = _load_gt_for_vis(root, tax, model_id)
    pred_s1 = model_s1(partial)[-1][0].detach().cpu().numpy()
    pred_s2 = model_s2(partial)[-1][0].detach().cpu().numpy()
    seed = hash(model_id) % 10000
    partial_s = _subsample(partial_np, 2048, seed)
    gt_s = _subsample(gt, 6000, seed + 1)
    pred_s1s = _subsample(pred_s1, 6000, seed + 2)
    pred_s2s = _subsample(pred_s2, 6000, seed + 3)
    lim_min, lim_max = _limits([partial_s, gt_s, pred_s1s, pred_s2s], pad_ratio=PAD_RATIO)
    data = [
        (partial_s, COLORS["input"], SIZES_USE["input"]),
        (pred_s1s, COLORS["stage1"], SIZES_USE["msf"]),
        (pred_s2s, COLORS["stage2"], SIZES_USE["msf"]),
        (gt_s, COLORS["gt"], SIZES_USE["gt"]),
    ]
    return [
        _stitch_row([
            _render_square_panel(pts, c, sz, elev, azim, lim_min, lim_max, "")
            for pts, c, sz in data
        ])
        for elev, azim in VIEWS_2
    ]


def main():
    _set_chinese_font()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tmp_s1 = os.path.join(TMP_VIS_ROOT, "s1")
    tmp_s2 = os.path.join(TMP_VIS_ROOT, "s2")
    os.makedirs(tmp_s1, exist_ok=True)
    os.makedirs(tmp_s2, exist_ok=True)
    print(f"tmp dirs: {TMP_VIS_ROOT}", flush=True)

    print("Loading Stage-1 PCN_core...", flush=True)
    model_s1, _, _ = _build_model(
        os.path.join(_ROOT, CFG_STAGE1),
        os.path.join(_ROOT, CKPT_STAGE1),
        tmp_s1,
        "msf_pure_group_sigmoid",
    )
    time.sleep(0.3)
    print("Loading Stage-2 PCN_core...", flush=True)
    model_s2, _, _ = _build_model(
        os.path.join(_ROOT, CFG_STAGE2),
        os.path.join(_ROOT, CKPT_STAGE2),
        tmp_s2,
        "msf_pure_group_sigmoid",
    )
    model_s1 = model_s1.to(device).eval()
    model_s2 = model_s2.to(device).eval()

    all_rows = []
    for i, (name, tax, mid) in enumerate(PICKS, 1):
        print(f"[{i}/3] {name} {mid[:12]}...", flush=True)
        all_rows.extend(_rows_for_sample(_ROOT, device, model_s1, model_s2, tax, mid))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    stitched = _stitch_body(all_rows)
    Image.fromarray(stitched).save(OUT, dpi=(SAVE_DPI, SAVE_DPI), optimize=True)
    print(f"saved {OUT}  {stitched.shape[1]}x{stitched.shape[0]}", flush=True)


if __name__ == "__main__":
    main()
