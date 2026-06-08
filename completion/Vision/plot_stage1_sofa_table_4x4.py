"""
沙发 + 桌子：4 行 × 4 列（每样本 2 视角 × 4 列），顶栏与 README 图3 一致。
权重：PCN_core 一阶段（MSF Sigmoid ckpt-best + PCSA ckpt-epoch-150）。

用法:
  cd completion
  python Vision/plot_stage1_sofa_table_4x4.py
"""

from __future__ import annotations

import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch

_COMPLETION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _COMPLETION_ROOT not in sys.path:
    sys.path.insert(0, _COMPLETION_ROOT)

from Vision.plot_stage1_msf_vs_pcsa_gallery import (
    CKPT_MSF,
    CKPT_PCSA,
    CFG_MSF,
    CFG_PCSA,
    COLORS,
    COL_GAP_PX,
    SIZES,
    VIEWS,
    _build_model,
    _infer_one,
    _limits,
    _load_gt_for_vis,
    _load_partial_for_vis,
    _render_square_panel,
    _set_chinese_font,
    _subsample,
)
from Vision.stitch_gallery_rows import (
    HEADER_FONT_SIZE,
    HEADER_H,
    HEADER_STROKE,
    ROW_GAP,
    SAVE_DPI,
    _render_header,
)

STITCH_TOP_PAD_PX = 10

VISION_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# PCN_core 一阶段权重（与 vis_stage1_sigmoid_vs_pcsa_paper 一致）
CFG_MSF = "cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid.yaml"
CFG_PCSA = "cfgs/PCN_models/AdaPoinTr_MSF_B.yaml"
CKPT_MSF = (
    "experiments/AdaPoinTr_MSF_Pure_Group_sigmoid/PCN_models/"
    "exp_MSF_Pure_Group_sigmoid/ckpt-best.pth"
)
CKPT_PCSA = (
    "experiments/AdaPoinTr_MSF_B/PCN_models/"
    "exp_pksa_bug_author_impl/ckpt-epoch-150.pth"
)

SOFA = ("沙发", "04256520", "2fc5cf498b0fa6de1525e8c8552c3a9c")

# --picks 时下行 4 选 1（只画图，不算 CD）
PICK_BOTTOMS = [
    ("台灯", "03636649", "e1fe4f81f074abc3e6597d391ab6fcc1", "lamp"),
    ("椅子", "03001627", "2a1124c7deb11176af42602f1636bd9", "chair"),
    ("橱柜", "02933112", "9b34b5983bd64409e08dea88cca8641e", "cabinet"),
    ("桌子", "04379243", "16ecdb0dcbd419ce30bbd4cddd04c77b", "table"),
]

# 上：01 图沙发（2fc5cf49）；下：pick_01 台灯
SAMPLES = [
    SOFA,
    ("台灯", "03636649", "e1fe4f81f074abc3e6597d391ab6fcc1"),
]
PICK_OUT_DIR = os.path.join(VISION_OUT, "sofa_bottom_candidates")
PICK_DPI = 220
VIEWS_2 = [(18, 45), (10, 90)]
PAD_RATIO = 0.42


def _stitch_row(panels: list[np.ndarray], col_gap: int) -> np.ndarray:
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
    return np.concatenate(parts, axis=1)


@torch.no_grad()
def _rows_for_sample(root, device, model_msf, model_pcsa, tax, model_id):
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


def _stitch_body(body_rows: list[np.ndarray]) -> np.ndarray:
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
def main(picks_only: bool = False):
    _set_chinese_font()
    root = _COMPLETION_ROOT
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tmp_root = "/tmp" if os.path.isdir("/tmp") else root

    print("Loading MSF / PCSA (PCN_core stage-1)...", flush=True)
    model_msf, _, _ = _build_model(
        os.path.join(root, CFG_MSF), os.path.join(root, CKPT_MSF),
        os.path.join(tmp_root, "tmp_vis_4x4_msf"), "msf_pure_group_sigmoid",
    )
    time.sleep(0.3)
    model_pcsa, _, _ = _build_model(
        os.path.join(root, CFG_PCSA), os.path.join(root, CKPT_PCSA),
        os.path.join(tmp_root, "tmp_vis_4x4_pcsa"), "pcsa",
    )
    model_msf = model_msf.to(device).eval()
    model_pcsa = model_pcsa.to(device).eval()

    from PIL import Image

    if picks_only:
        os.makedirs(PICK_OUT_DIR, exist_ok=True)
        print("sofa 2fc5cf49...", flush=True)
        sofa_rows = _rows_for_sample(root, device, model_msf, model_pcsa, SOFA[1], SOFA[2])
        for i, (name_cn, tax, model_id, tag) in enumerate(PICK_BOTTOMS, 1):
            print(f"[{i}/4] {name_cn}", flush=True)
            bottom_rows = _rows_for_sample(
                root, device, model_msf, model_pcsa, tax, model_id
            )
            out = os.path.join(PICK_OUT_DIR, f"pick_{i:02d}_{name_cn}_{tag}.png")
            Image.fromarray(_stitch_body(sofa_rows + bottom_rows)).save(
                out, dpi=(PICK_DPI, PICK_DPI), optimize=True
            )
            print(f"  -> {out}", flush=True)
        print(f"done 4 picks in {PICK_OUT_DIR}", flush=True)
        return

    body_rows: list[np.ndarray] = []

    for name_cn, tax, model_id in SAMPLES:
        body_rows.extend(
            _rows_for_sample(root, device, model_msf, model_pcsa, tax, model_id)
        )
        print(f"  done {name_cn} {model_id[:12]}...", flush=True)

    out_path = os.path.join(VISION_OUT, "stage1_sofa_table_4x4.png")
    os.makedirs(VISION_OUT, exist_ok=True)
    stitched = _stitch_body(body_rows)
    Image.fromarray(stitched).save(out_path, dpi=(SAVE_DPI, SAVE_DPI))
    print(f"saved {out_path}  {stitched.shape[1]}x{stitched.shape[0]}", flush=True)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--picks",
        action="store_true",
        help="沙发固定 + 4 个下行候选各 1 张（pick_01..04，省磁盘）",
    )
    args = ap.parse_args()
    main(picks_only=args.picks)
