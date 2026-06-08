"""
PCN_core：Stage-1 MSF Sigmoid vs Stage-2 FeedPoinTrS-FT 可视化（4 列布局）

Stage-1: AdaPoinTr_MSF_Pure_Group_sigmoid / ckpt-best（与 MSF vs PCSA 一阶段一致）
Stage-2: feedpointrs_ft crop02_04 seed42 / ckpt-epoch-050（test F=0.8226 CDL1=7.0607）

默认 --scan：扫 PCN test，按 ΔCD + 缺失区 G2P 为每类挑 Stage-2 明显更优样本，再出 4×(2视角×4列) 图。
布局与 sofa_bottom_candidates/pick_01 一致：输入 / Stage-1 / Stage-2 / G.T.

用法:
  cd completion
  CUDA_VISIBLE_DEVICES=1 python -u Vision/plot_stage1_vs_stage2_picks.py --scan
  python -u Vision/plot_stage1_vs_stage2_picks.py --picks data/stage1_vs_stage2_vis_picks.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
from PIL import Image

_COMPLETION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _COMPLETION_ROOT not in sys.path:
    sys.path.insert(0, _COMPLETION_ROOT)

from extensions.chamfer_dist import ChamferDistanceL1, ChamferFunction
from tools import builder
from utils.metrics import Metrics
from Vision.plot_stage1_msf_vs_pcsa_gallery import (
    COL_GAP_PX,
    SIZES,
    _build_model,
    _limits,
    _load_gt_for_vis,
    _load_partial_for_vis,
    _render_square_panel,
    _set_chinese_font,
    _subsample,
    _to_tax,
)
from Vision.stitch_gallery_rows import HEADER_FONT_SIZE, HEADER_H, HEADER_STROKE, ROW_GAP

CFG_STAGE1 = "cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid.yaml"
CFG_STAGE2 = "cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid_feedpointrs_ft.yaml"
CKPT_STAGE1 = (
    "experiments/AdaPoinTr_MSF_Pure_Group_sigmoid/PCN_models/"
    "exp_MSF_Pure_Group_sigmoid/ckpt-best.pth"
)
CKPT_STAGE2 = (
    "experiments/AdaPoinTr_MSF_Pure_Group_sigmoid_feedpointrs_ft/PCN_models/"
    "exp_MSF_Pure_Group_sigmoid_feedpointrs_ft_crop02_04_seed42/ckpt-epoch-050.pth"
)

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

TAG_EN = {
    "02691156": "airplane",
    "02933112": "cabinet",
    "02958343": "car",
    "03001627": "chair",
    "03636649": "lamp",
    "04256520": "sofa",
    "04379243": "table",
    "04530566": "boat",
}

# 桌子/椅子/橱柜/沙发（台灯类 Stage-2 缺失区改善普遍偏小，不利于可视化）
DEFAULT_SCAN_TAX = ["04379243", "03001627", "02933112", "04256520"]

OLD_PICKS = {
    "e1fe4f81f074abc3e6597d391ab6fcc1",
    "2a1124c7deb11176af42602f1636bd9",
    "9b34b5983bd64409e08dea88cca8641e",
    "16ecdb0dcbd419ce30bbd4cddd04c77b",
    "f9b41c5ee5ce8b6fcb8d8c6d4df8143",
    "8cc8499cdf11e9fc1735ea0e092a805a",
}

COLORS = {
    "input": "#7f7f7f",
    "stage1": "#1f77b4",
    "stage2": "#d62728",
    "gt": "#2ca02c",
}

VIEWS_2 = [(18, 45), (10, 90), (18, 225)]
PAD_RATIO = 0.42
OUT_DIR = os.path.join(_COMPLETION_ROOT, "VIS", "vis_stage1_vs_stage2_picks")
PICKS_FILE = os.path.join(_COMPLETION_ROOT, "data", "stage1_vs_stage2_vis_picks.txt")
HEADER_COLUMNS = ("输入", "Stage-1", "Stage-2", "G.T.")
SAVE_DPI = 220
STITCH_TOP_PAD_PX = 10

_CD = ChamferDistanceL1()


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
    log_name = "vis_stage1_vs_stage2"
    seed = 0
    deterministic = False
    sync_bn = False
    mode = None
    config = ""
    val_freq = 10


def _cd_l1_x1e3(pred, gt, device):
    if not torch.is_tensor(pred):
        pred = torch.as_tensor(pred, dtype=torch.float32, device=device)
    if not torch.is_tensor(gt):
        gt = torch.as_tensor(gt, dtype=torch.float32, device=device)
    with torch.no_grad():
        return float(_CD(pred.unsqueeze(0).to(device), gt.unsqueeze(0).to(device)).item() * 1000.0)


def _fscore(pred, gt, device):
    if not torch.is_tensor(pred):
        pred = torch.as_tensor(pred, dtype=torch.float32, device=device)
    if not torch.is_tensor(gt):
        gt = torch.as_tensor(gt, dtype=torch.float32, device=device)
    with torch.no_grad():
        return float(Metrics._get_f_score(pred.unsqueeze(0).to(device), gt.unsqueeze(0).to(device)).item())


def _missing_g2p_x1e3(partial, pred, gt, device, miss_ratio=0.35):
    if not torch.is_tensor(partial):
        partial = torch.as_tensor(partial, dtype=torch.float32, device=device)
    if not torch.is_tensor(pred):
        pred = torch.as_tensor(pred, dtype=torch.float32, device=device)
    if not torch.is_tensor(gt):
        gt = torch.as_tensor(gt, dtype=torch.float32, device=device)
    partial = partial.unsqueeze(0).to(device)
    pred = pred.unsqueeze(0).to(device)
    gt = gt.unsqueeze(0).to(device)
    with torch.no_grad():
        _, d_partial_gt = ChamferFunction.apply(partial, gt)
        d_in = torch.sqrt(d_partial_gt[0].clamp(min=1e-8))
        k = max(1, int(d_in.numel() * float(miss_ratio)))
        thr = torch.topk(d_in, k=k, largest=True).values.min()
        miss_mask = d_in >= thr
        _, d_gt_pred = ChamferFunction.apply(gt, pred)
        d_g2p = torch.sqrt(d_gt_pred[0].clamp(min=1e-8))
        if miss_mask.any():
            return float(d_g2p[miss_mask].mean().item() * 1000.0)
        return float(d_g2p.mean().item() * 1000.0)


def _selection_score(rec):
    d_cd = rec["cd_s1"] - rec["cd_s2"]
    d_miss = rec["miss_s1"] - rec["miss_s2"]
    d_f = rec["f_s2"] - rec["f_s1"]
    rec["delta_cd"] = d_cd
    rec["delta_miss"] = d_miss
    rec["delta_f"] = d_f
    # 强调缺失区改善 + CD 下降；Stage-1 CD 适中时更易看出差别
    cd_s1 = rec["cd_s1"]
    mid_bonus = 0.0
    if 5.5 <= cd_s1 <= 11.0:
        mid_bonus = 0.15 * min(d_cd, 4.0)
    rec["score"] = 0.10 * d_cd + 0.75 * d_miss + 0.15 * (d_f * 100.0) + mid_bonus
    return rec["score"]


def _load_models(root, device, tmp_root):
    print("Loading Stage-1 PCN_core MSF Sigmoid...", flush=True)
    model_s1, cfg, args = _build_model(
        os.path.join(root, CFG_STAGE1),
        os.path.join(root, CKPT_STAGE1),
        os.path.join(tmp_root, "tmp_vis_s1_vs_s2_s1"),
        "msf_pure_group_sigmoid",
    )
    time.sleep(0.3)
    print("Loading Stage-2 PCN_core FeedPoinTrS-FT ep50...", flush=True)
    model_s2, _, _ = _build_model(
        os.path.join(root, CFG_STAGE2),
        os.path.join(root, CKPT_STAGE2),
        os.path.join(tmp_root, "tmp_vis_s1_vs_s2_s2"),
        "msf_pure_group_sigmoid",
    )
    return model_s1.to(device).eval(), model_s2.to(device).eval(), cfg, args


@torch.no_grad()
def _infer_metrics(model_s1, model_s2, partial, gt_t, gt, tax, model_id, device):
    pred_s1 = model_s1(partial)[-1]
    pred_s2 = model_s2(partial)[-1]
    if isinstance(pred_s1, (list, tuple)):
        pred_s1 = pred_s1[0]
    if isinstance(pred_s2, (list, tuple)):
        pred_s2 = pred_s2[0]
    rec = {
        "tax": tax,
        "model_id": str(model_id),
        "cd_s1": _cd_l1_x1e3(pred_s1[0], gt_t[0], device),
        "cd_s2": _cd_l1_x1e3(pred_s2[0], gt_t[0], device),
        "f_s1": _fscore(pred_s1[0], gt_t[0], device),
        "f_s2": _fscore(pred_s2[0], gt_t[0], device),
        "miss_s1": _missing_g2p_x1e3(partial[0], pred_s1[0], gt_t[0], device),
        "miss_s2": _missing_g2p_x1e3(partial[0], pred_s2[0], gt_t[0], device),
    }
    _selection_score(rec)
    return rec


def scan_best_picks(model_s1, model_s2, test_loader, device, scan_tax: set[str], min_delta_cd: float):
    by_tax = defaultdict(list)
    n = 0
    for taxonomy_ids, model_ids, data in test_loader:
        partial = data[0].to(device)
        gt_t = data[1]
        for i in range(partial.shape[0]):
            tax = _to_tax(taxonomy_ids[i])
            if tax not in scan_tax:
                continue
            model_id = str(model_ids[i])
            if model_id in OLD_PICKS:
                continue
            rec = _infer_metrics(
                model_s1, model_s2, partial[i : i + 1], gt_t[i : i + 1], None, tax, model_id, device
            )
            by_tax[tax].append(rec)
            n += 1
            if n % 50 == 0:
                print(f"  scanned {n} ...", flush=True)

    selected = []
    for tax in sorted(scan_tax):
        pool = by_tax.get(tax, [])
        if not pool:
            print(f"WARN: no samples for {tax}", flush=True)
            continue
        winners = [
            r for r in pool
            if r["cd_s2"] < r["cd_s1"] and r["delta_miss"] >= 2.0 and r["delta_cd"] >= min_delta_cd
        ]
        if not winners:
            winners = [r for r in pool if r["cd_s2"] < r["cd_s1"] and r["delta_miss"] >= 1.5]
        if not winners:
            winners = [r for r in pool if r["cd_s2"] < r["cd_s1"]]
        if not winners:
            winners = pool
        best = max(winners, key=lambda r: r["score"])
        top5 = sorted(pool, key=lambda r: r["score"], reverse=True)[:5]
        selected.append({"best": best, "top5": top5})
        name = TAXONOMY_CN[tax]
        print(
            f"  {name}: {best['model_id'][:12]}...  "
            f"CD s1={best['cd_s1']:.2f} s2={best['cd_s2']:.2f} Δ={best['delta_cd']:.2f}  "
            f"Δmiss={best['delta_miss']:.2f} score={best['score']:.2f}",
            flush=True,
        )
    return selected


def _flatten_selected(selected):
    return [item["best"] if isinstance(item, dict) else item for item in selected]


def _save_picks(selected, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = ["# Stage-1 vs Stage-2 vis picks (tax/model_id)"]
    for rec in selected:
        lines.append(f"{rec['tax']}/{rec['model_id']}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _load_picks(path):
    picks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tax, mid = line.split("/", 1)
            name = TAXONOMY_CN.get(tax, tax)
            tag = TAG_EN.get(tax, tax)
            picks.append((name, tax, mid, tag))
    return picks


def _render_header(width: int) -> np.ndarray:
    from PIL import ImageDraw, ImageFont

    font_path = "/usr/share/fonts/truetype/arphic/uming.ttc"
    font = ImageFont.truetype(font_path, HEADER_FONT_SIZE) if os.path.isfile(font_path) else ImageFont.load_default()
    n = len(HEADER_COLUMNS)
    panel_w = (width - COL_GAP_PX * (n - 1)) // n
    centers = [panel_w // 2 + i * (panel_w + COL_GAP_PX) for i in range(n)]
    img = Image.new("RGB", (width, HEADER_H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    stroke = HEADER_STROKE
    for label, cx in zip(HEADER_COLUMNS, centers):
        bbox = draw.textbbox((0, 0), label, font=font, stroke_width=stroke)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            (cx - tw // 2, (HEADER_H - th) // 2 - bbox[1]), label,
            fill=(0, 0, 0), font=font, stroke_width=stroke, stroke_fill=(0, 0, 0),
        )
    return np.asarray(img)


def _pad_panel(img, target_h, target_w):
    h, w = img.shape[:2]
    out = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
    y0, x0 = max(0, (target_h - h) // 2), max(0, (target_w - w) // 2)
    out[y0 : y0 + h, x0 : x0 + w] = img[: min(h, target_h), : min(w, target_w)]
    return out


def _stitch_row(panels):
    th, tw = max(p.shape[0] for p in panels), max(p.shape[1] for p in panels)
    panels = [_pad_panel(p, th, tw) for p in panels]
    gap = np.ones((th, COL_GAP_PX, 3), dtype=np.uint8) * 255
    parts = []
    for i, p in enumerate(panels):
        parts.append(p)
        if i < len(panels) - 1:
            parts.append(gap)
    return np.concatenate(parts, axis=1)


def _stitch_body(body_rows):
    target_w = max(r.shape[1] for r in body_rows)
    padded = []
    for r in body_rows:
        if r.shape[1] < target_w:
            r = np.concatenate([r, np.ones((r.shape[0], target_w - r.shape[1], 3), dtype=np.uint8) * 255], axis=1)
        padded.append(r)
    body = padded[0]
    for r in padded[1:]:
        body = np.concatenate([body, np.ones((ROW_GAP, target_w, 3), dtype=np.uint8) * 255, r], axis=0)
    w = body.shape[1]
    return np.concatenate([
        np.ones((STITCH_TOP_PAD_PX, w, 3), dtype=np.uint8) * 255,
        _render_header(w),
        np.ones((ROW_GAP, w, 3), dtype=np.uint8) * 255,
        body,
    ], axis=0)


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
        (partial_s, COLORS["input"], SIZES["input"]),
        (pred_s1s, COLORS["stage1"], SIZES["msf"]),
        (pred_s2s, COLORS["stage2"], SIZES["msf"]),
        (gt_s, COLORS["gt"], SIZES["gt"]),
    ]
    return [
        _stitch_row([
            _render_square_panel(pts, c, sz, elev, azim, lim_min, lim_max, "")
            for pts, c, sz in data
        ])
        for elev, azim in VIEWS_2
    ]


def render_picks(picks, root, device, model_s1, model_s2, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    overview_rows = []
    for i, (name_cn, tax, model_id, tag) in enumerate(picks, 1):
        print(f"[{i}/{len(picks)}] render {name_cn} {model_id[:12]}...", flush=True)
        rows = _rows_for_sample(root, device, model_s1, model_s2, tax, model_id)
        out = os.path.join(out_dir, f"pick_{i:02d}_{name_cn}_{tag}.png")
        Image.fromarray(_stitch_body(rows)).save(out, dpi=(SAVE_DPI, SAVE_DPI), optimize=True)
        print(f"  -> {out}", flush=True)
        overview_rows.extend(rows)
    overview_path = os.path.join(out_dir, "00_overview_4objects.png")
    Image.fromarray(_stitch_body(overview_rows)).save(overview_path, dpi=(SAVE_DPI, SAVE_DPI), optimize=True)
    print(f"saved overview {overview_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", action="store_true", help="scan test set then render")
    parser.add_argument("--picks", type=str, default="", help="picks file (tax/model_id per line)")
    parser.add_argument("--min-delta-cd", type=float, default=1.5, help="min CDL1×1e3 drop for selection")
    parser.add_argument("--out-dir", type=str, default=OUT_DIR)
    args = parser.parse_args()

    _set_chinese_font()
    root = _COMPLETION_ROOT
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tmp_root = "/tmp" if os.path.isdir("/tmp") else root
    out_dir = args.out_dir

    model_s1, model_s2, cfg, vis_args = _load_models(root, device, tmp_root)
    _, test_loader = builder.dataset_builder(vis_args, cfg.dataset.test)

    if args.scan or not args.picks:
        scan_tax = set(DEFAULT_SCAN_TAX)
        print(f"Scanning PCN test for {len(scan_tax)} categories (ΔCD>={args.min_delta_cd})...", flush=True)
        selected_raw = scan_best_picks(model_s1, model_s2, test_loader, device, scan_tax, args.min_delta_cd)
        selected = _flatten_selected(selected_raw)
        _save_picks(selected, PICKS_FILE)
        summary_path = os.path.join(out_dir, "selected_samples_summary.txt")
        os.makedirs(out_dir, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("tax\tname\tmodel_id\tcd_s1\tcd_s2\tdelta_cd\tmiss_s1\tmiss_s2\tdelta_miss\tscore\n")
            for rec in selected:
                f.write(
                    f"{rec['tax']}\t{TAXONOMY_CN[rec['tax']]}\t{rec['model_id']}\t"
                    f"{rec['cd_s1']:.4f}\t{rec['cd_s2']:.4f}\t{rec['delta_cd']:.4f}\t"
                    f"{rec['miss_s1']:.4f}\t{rec['miss_s2']:.4f}\t{rec['delta_miss']:.4f}\t{rec['score']:.4f}\n"
                )
            f.write("\n# top5 per category\n")
            for item in selected_raw:
                best = item["best"]
                f.write(f"\n[{TAXONOMY_CN[best['tax']]}]\n")
                for rank, rec in enumerate(item["top5"], 1):
                    f.write(
                        f"  {rank}\t{rec['model_id']}\tΔcd={rec['delta_cd']:.2f}\t"
                        f"Δmiss={rec['delta_miss']:.2f}\tscore={rec['score']:.2f}\n"
                    )
        with open(os.path.join(out_dir, "selected_samples.json"), "w", encoding="utf-8") as f:
            json.dump(selected, f, indent=2, ensure_ascii=False)
        print(f"picks saved {PICKS_FILE}", flush=True)
        picks = [
            (TAXONOMY_CN[r["tax"]], r["tax"], r["model_id"], TAG_EN[r["tax"]])
            for r in selected
        ]
    else:
        picks = _load_picks(os.path.join(root, args.picks) if not os.path.isabs(args.picks) else args.picks)

    render_picks(picks, root, device, model_s1, model_s2, out_dir)
    print(f"done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
