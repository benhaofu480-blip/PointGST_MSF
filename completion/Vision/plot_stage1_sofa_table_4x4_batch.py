"""
沙发 2fc5cf49... 固定 + 4 个下行候选，各 1 张 4×4（只推理+画图，不算 CD）。

用法:
  cd completion && python -u Vision/plot_stage1_sofa_table_4x4_batch.py
"""

from __future__ import annotations

import os
import sys
import time

# 必须先 Agg，避免无显示环境卡死
import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

_COMPLETION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _COMPLETION_ROOT not in sys.path:
    sys.path.insert(0, _COMPLETION_ROOT)

from Vision.plot_stage1_msf_vs_pcsa_gallery import (
    COL_GAP_PX,
    COLORS,
    SIZES,
    _build_model,
    _infer_one,
    _limits,
    _load_gt_for_vis,
    _load_partial_for_vis,
    _render_square_panel,
    _set_chinese_font,
    _subsample,
)
from Vision.plot_stage1_sofa_table_4x4 import (
    CFG_MSF,
    CFG_PCSA,
    CKPT_MSF,
    CKPT_PCSA,
    PAD_RATIO,
    VIEWS_2,
    _stitch_row,
)
from Vision.stitch_gallery_rows import (
    HEADER_FONT_SIZE,
    HEADER_H,
    HEADER_STROKE,
    ROW_GAP,
    _render_header,
)

STITCH_TOP_PAD_PX = 10
# 略降 DPI，减小 PNG 体积
OUT_DPI = 220
VISION_OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "output", "sofa_bottom_candidates"
)
_TMP_ROOT = "/tmp" if os.path.isdir("/tmp") else _COMPLETION_ROOT

SOFA = ("沙发", "04256520", "2fc5cf498b0fa6de1525e8c8552c3a9c")

BOTTOM_CANDIDATES = [
    ("台灯", "03636649", "e1fe4f81f074abc3e6597d391ab6fcc1", "lamp"),
    ("椅子", "03001627", "2a1124c7deb11176af42602f1636bd9", "chair"),
    ("橱柜", "02933112", "9b34b5983bd64409e08dea88cca8641e", "cabinet"),
    ("桌子", "04379243", "16ecdb0dcbd419ce30bbd4cddd04c77b", "table"),
]


@torch.no_grad()
def _render_pair(root, device, model_msf, model_pcsa, tax, model_id):
    partial_np = _load_partial_for_vis(root, tax, model_id, 0)
    partial = torch.from_numpy(partial_np).unsqueeze(0).float().to(device)
    gt = _load_gt_for_vis(root, tax, model_id)
    rec = _infer_one(model_msf, model_pcsa, partial, None, gt, tax, model_id)

    seed = hash(model_id) % 10000
    partial_s = _subsample(rec["partial"], 2048, seed)
    gt_s = _subsample(rec["gt"], 6000, seed + 1)
    pred_p = _subsample(rec["pred_pcsa"], 6000, seed + 2)
    pred_m = _subsample(rec["pred_msf"], 6000, seed + 3)
    lim_min, lim_max = _limits([partial_s, gt_s, pred_p, pred_m], pad_ratio=PAD_RATIO)
    data = [
        (partial_s, COLORS["input"], SIZES["input"]),
        (pred_p, COLORS["pcsa"], SIZES["pcsa"]),
        (pred_m, COLORS["msf"], SIZES["msf"]),
        (gt_s, COLORS["gt"], SIZES["gt"]),
    ]
    rows = []
    for elev, azim in VIEWS_2:
        panels = [
            _render_square_panel(pts, color, sz, elev, azim, lim_min, lim_max, "")
            for pts, color, sz in data
        ]
        rows.append(_stitch_row(panels, COL_GAP_PX))
    return rows


def _stitch_full(body_rows):
    body = body_rows[0]
    for r in body_rows[1:]:
        sep = np.ones((ROW_GAP, body.shape[1], 3), dtype=np.uint8) * 255
        body = np.concatenate([body, sep, r], axis=0)
    w = body.shape[1]
    header = _render_header(w, COL_GAP_PX, HEADER_FONT_SIZE, HEADER_STROKE, HEADER_H)
    top = np.ones((STITCH_TOP_PAD_PX, w, 3), dtype=np.uint8) * 255
    gap = np.ones((ROW_GAP, w, 3), dtype=np.uint8) * 255
    return np.concatenate([top, header, gap, body], axis=0)


@torch.no_grad()
def main():
    _set_chinese_font()
    root = _COMPLETION_ROOT
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(VISION_OUT, exist_ok=True)

    print("load MSF / PCSA ...", flush=True)
    model_msf, _, _ = _build_model(
        os.path.join(root, CFG_MSF),
        os.path.join(root, CKPT_MSF),
        os.path.join(_TMP_ROOT, "vis_4x4_batch_msf"),
        "msf_pure_group_sigmoid",
    )
    time.sleep(0.2)
    model_pcsa, _, _ = _build_model(
        os.path.join(root, CFG_PCSA),
        os.path.join(root, CKPT_PCSA),
        os.path.join(_TMP_ROOT, "vis_4x4_batch_pcsa"),
        "pcsa",
    )
    model_msf = model_msf.to(device).eval()
    model_pcsa = model_pcsa.to(device).eval()

    print("sofa 2fc5cf49...", flush=True)
    sofa_rows = _render_pair(
        root, device, model_msf, model_pcsa, SOFA[1], SOFA[2]
    )

    from PIL import Image

    for i, (name_cn, tax, model_id, tag) in enumerate(BOTTOM_CANDIDATES, 1):
        print(f"[{i}/4] {name_cn} ...", flush=True)
        bottom_rows = _render_pair(root, device, model_msf, model_pcsa, tax, model_id)
        stitched = _stitch_full(sofa_rows + bottom_rows)
        out = os.path.join(VISION_OUT, f"pick_{i:02d}_{name_cn}_{tag}.png")
        Image.fromarray(stitched).save(out, dpi=(OUT_DPI, OUT_DPI), optimize=True)
        print(f"  -> {out}", flush=True)

    print(f"done: {len(BOTTOM_CANDIDATES)} files in {VISION_OUT}", flush=True)


if __name__ == "__main__":
    main()
