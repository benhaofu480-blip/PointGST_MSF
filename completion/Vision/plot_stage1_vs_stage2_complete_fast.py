"""
Complete 权重：Stage-1 vs Stage-2 快速出图 / 扫描选样

用法:
  cd completion
  # 扫描 test 并出 10 张（排除 batch1，保留指定船只）
  CUDA_VISIBLE_DEVICES=1 python -u Vision/plot_stage1_vs_stage2_complete_fast.py --scan --out-subdir batch2
  # 仅渲染（使用脚本内 PICKS）
  CUDA_VISIBLE_DEVICES=1 python -u Vision/plot_stage1_vs_stage2_complete_fast.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
from PIL import Image

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

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

CFG_S1 = "cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid_complete.yaml"
CFG_S2 = "cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid_feedpointrs_ft_complete.yaml"
CKPT_S1 = (
    "experiments/AdaPoinTr_MSF_Pure_Group_sigmoid_complete/PCN_models/"
    "exp_MSF_Pure_Group_sigmoid_complete_seed42/ckpt-best.pth"
)
CKPT_S2 = (
    "experiments/AdaPoinTr_MSF_Pure_Group_sigmoid_feedpointrs_ft_complete/PCN_models/"
    "exp_MSF_Pure_Group_sigmoid_feedpointrs_ft_crop02_04_complete_s1ep150_seed42/ckpt-last.pth"
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
ALL_TAX = list(TAXONOMY_CN.keys())

# batch1 已用，扫描时排除
BANNED_IDS = {
    "ddb20a7778038d87f51f77a6d7299806",
    "f9b41c5ee5ce8b6fcb8d8c6d4df8143",
    "d2dc852235fe39ca1112a51947cf2b61",
    "24cd35785c38c6ccbdf89940ba47dea",
    "a224010a537bc683104e417f71823787",
    "4aee1567027d9dd14357a62465045ec4",
    "9c87aebafdb4830ba5dc3fef8d22887b",
    "9aecd48a3af10deeee83c0324834f3fa",
    "2aa624d7f91a5c16193d9e76bb15876",
    "2036aaa68d164c373fe047712e43e185",
}
FORCE_INCLUDE_ID = "390de3a1bd0191c881d9d9b1473043a2"  # 用户认可的船只

PICKS = [
    ("桌子", "04379243", "ddb20a7778038d87f51f77a6d7299806", "table", 4.43),
    ("台灯", "03636649", "f9b41c5ee5ce8b6fcb8d8c6d4df8143", "lamp", 4.65),
    ("橱柜", "02933112", "d2dc852235fe39ca1112a51947cf2b61", "cabinet", 2.55),
    ("椅子", "03001627", "24cd35785c38c6ccbdf89940ba47dea", "chair", 2.40),
    ("桌子", "04379243", "a224010a537bc683104e417f71823787", "table", 2.47),
    ("台灯", "03636649", "4aee1567027d9dd14357a62465045ec4", "lamp", 0.97),
    ("橱柜", "02933112", "9c87aebafdb4830ba5dc3fef8d22887b", "cabinet", 1.72),
    ("椅子", "03001627", "9aecd48a3af10deeee83c0324834f3fa", "chair", 1.22),
    ("桌子", "04379243", "2aa624d7f91a5c16193d9e76bb15876", "table", 1.71),
    ("橱柜", "02933112", "2036aaa68d164c373fe047712e43e185", "cabinet", 1.87),
]

VIEWS = [(18, 45), (10, 90)]
OUT_BASE = os.path.join(_ROOT, "VIS", "vis_stage1_vs_stage2_complete")
HEADERS = ("输入", "Stage-1", "Stage-2", "G.T.")
COLORS = {"input": "#7f7f7f", "s1": "#1f77b4", "s2": "#d62728", "gt": "#2ca02c"}
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
    log_name = "vis_stage1_vs_stage2_complete"
    seed = 0
    deterministic = False
    sync_bn = False
    mode = None
    config = ""
    val_freq = 10


def _cd_l1_x1e3(pred, gt, device):
    p = pred.unsqueeze(0).to(device)
    g = gt.unsqueeze(0).to(device)
    return float(_CD(p, g)[0].item() * 1000.0)


def _fscore(pred, gt, device):
    p = pred.unsqueeze(0).to(device)
    g = gt.unsqueeze(0).to(device)
    return float(Metrics.get().fscore(p, g).item())


def _missing_g2p_x1e3(partial, pred, gt, device, miss_ratio=0.25):
    p = partial.unsqueeze(0).to(device)
    pr = pred.unsqueeze(0).to(device)
    g = gt.unsqueeze(0).to(device)
    _, d_in = ChamferFunction.apply(g, p)
    d_in = torch.sqrt(d_in[0].clamp(min=1e-8))
    k = max(1, int(d_in.numel() * float(miss_ratio)))
    thr = torch.topk(d_in, k=k, largest=True).values.min()
    miss_mask = d_in >= thr
    _, d_gt_pred = ChamferFunction.apply(g, pr)
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
    cd_s1 = rec["cd_s1"]
    mid_bonus = 0.15 * min(d_cd, 4.0) if 5.5 <= cd_s1 <= 11.0 else 0.0
    rec["score"] = 0.10 * d_cd + 0.75 * d_miss + 0.15 * (d_f * 100.0) + mid_bonus
    return rec["score"]


@torch.no_grad()
def _infer_metrics(model_s1, model_s2, partial, gt_t, tax, model_id, device):
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


@torch.no_grad()
def scan_test(model_s1, model_s2, test_loader, device, banned: set[str]):
    by_tax = defaultdict(list)
    n = 0
    for taxonomy_ids, model_ids, data in test_loader:
        partial = data[0].to(device)
        gt_t = data[1]
        for i in range(partial.shape[0]):
            tax = _to_tax(taxonomy_ids[i])
            mid = str(model_ids[i])
            if mid in banned:
                continue
            rec = _infer_metrics(model_s1, model_s2, partial[i : i + 1], gt_t[i : i + 1], tax, mid, device)
            by_tax[tax].append(rec)
            n += 1
            if n % 100 == 0:
                print(f"  scanned {n} ...", flush=True)
    return by_tax


def _rec_to_pick(rec):
    tax = rec["tax"]
    return (TAXONOMY_CN[tax], tax, rec["model_id"], TAG_EN[tax], rec["delta_cd"])


def select_diverse_picks(by_tax, n_picks: int, force_mid: str | None):
    pools = {}
    for tax in ALL_TAX:
        pool = by_tax.get(tax, [])
        winners = [r for r in pool if r["cd_s2"] < r["cd_s1"]]
        if not winners:
            winners = pool
        pools[tax] = sorted(winners, key=lambda r: r["score"], reverse=True)

    picked_ids: set[str] = set()
    picks = []

    if force_mid:
        for pool in by_tax.values():
            for rec in pool:
                if rec["model_id"] == force_mid:
                    picks.append(_rec_to_pick(rec))
                    picked_ids.add(force_mid)
                    print(
                        f"  force {TAXONOMY_CN[rec['tax']]} {force_mid[:8]}... "
                        f"ΔCD={rec['delta_cd']:.2f} Δmiss={rec['delta_miss']:.2f}",
                        flush=True,
                    )
                    break

    for rank in range(20):
        if len(picks) >= n_picks:
            break
        for tax in ALL_TAX:
            if len(picks) >= n_picks:
                break
            pl = pools.get(tax, [])
            if rank >= len(pl):
                continue
            rec = pl[rank]
            if rec["model_id"] in picked_ids:
                continue
            picks.append(_rec_to_pick(rec))
            picked_ids.add(rec["model_id"])
            print(
                f"  pick {len(picks):02d} {TAXONOMY_CN[tax]} {rec['model_id'][:8]}... "
                f"ΔCD={rec['delta_cd']:.2f} score={rec['score']:.2f}",
                flush=True,
            )
    return picks[:n_picks]


def _header(w):
    from PIL import ImageDraw, ImageFont

    fp = "/usr/share/fonts/truetype/arphic/uming.ttc"
    font = ImageFont.truetype(fp, 52) if os.path.isfile(fp) else ImageFont.load_default()
    n = 4
    pw = (w - COL_GAP_PX * 3) // n
    centers = [pw // 2 + i * (pw + COL_GAP_PX) for i in range(n)]
    img = Image.new("RGB", (w, 108), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for lab, cx in zip(HEADERS, centers):
        bb = draw.textbbox((0, 0), lab, font=font, stroke_width=1)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        draw.text((cx - tw // 2, (108 - th) // 2 - bb[1]), lab, fill=(0, 0, 0), font=font, stroke_width=1, stroke_fill=(0, 0, 0))
    return np.asarray(img)


def _row(panels):
    th, tw = max(p.shape[0] for p in panels), max(p.shape[1] for p in panels)
    out = []
    for p in panels:
        pad = np.ones((th, tw, 3), dtype=np.uint8) * 255
        pad[: p.shape[0], : p.shape[1]] = p[:th, :tw]
        out.append(pad)
    gap = np.ones((th, COL_GAP_PX, 3), dtype=np.uint8) * 255
    parts = []
    for i, p in enumerate(out):
        parts.append(p)
        if i < 3:
            parts.append(gap)
    return np.concatenate(parts, axis=1)


def _body(rows):
    w = max(r.shape[1] for r in rows)
    rs = [r if r.shape[1] == w else np.concatenate([r, np.ones((r.shape[0], w - r.shape[1], 3), np.uint8) * 255], 1) for r in rows]
    body = rs[0]
    for r in rs[1:]:
        body = np.concatenate([body, np.ones((12, w, 3), np.uint8) * 255, r], 0)
    return np.concatenate([np.ones((10, w, 3), np.uint8) * 255, _header(w), np.ones((12, w, 3), np.uint8) * 255, body], 0)


@torch.no_grad()
def _render_one(root, dev, m1, m2, tax, mid):
    partial_np = _load_partial_for_vis(root, tax, mid, 0)
    partial = torch.from_numpy(partial_np).unsqueeze(0).float().to(dev)
    gt = _load_gt_for_vis(root, tax, mid)
    p1 = m1(partial)[-1][0].detach().cpu().numpy()
    p2 = m2(partial)[-1][0].detach().cpu().numpy()
    s = hash(mid) % 10000
    pts = [
        _subsample(partial_np, 2048, s),
        _subsample(p1, 6000, s + 1),
        _subsample(p2, 6000, s + 2),
        _subsample(gt, 6000, s + 3),
    ]
    lim = _limits(pts, pad_ratio=0.42)
    cols = [COLORS["input"], COLORS["s1"], COLORS["s2"], COLORS["gt"]]
    sz = [SIZES["input"], SIZES["msf"], SIZES["msf"], SIZES["gt"]]
    return [_row([_render_square_panel(pts[j], cols[j], sz[j], e, a, *lim, "") for j in range(4)]) for e, a in VIEWS]


def _load_models(dev, tmp):
    print("complete Stage-1 ...", flush=True)
    m1, cfg, args = _build_model(
        os.path.join(_ROOT, CFG_S1), os.path.join(_ROOT, CKPT_S1), os.path.join(tmp, "tmp_s1c_s1"), "msf_pure_group_sigmoid"
    )
    time.sleep(0.2)
    print("complete Stage-2 ...", flush=True)
    m2, _, _ = _build_model(
        os.path.join(_ROOT, CFG_S2), os.path.join(_ROOT, CKPT_S2), os.path.join(tmp, "tmp_s1c_s2"), "msf_pure_group_sigmoid"
    )
    return m1.to(dev).eval(), m2.to(dev).eval(), cfg, args


def render_picks(picks, out_dir, dev, m1, m2):
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    all_rows = []
    for i, (cn, tax, mid, tag, dcd) in enumerate(picks, 1):
        short_id = mid[:8]
        print(f"[{i}/{len(picks)}] {cn} {short_id}... ΔCD≈{dcd:.2f}", flush=True)
        rows = _render_one(_ROOT, dev, m1, m2, tax, mid)
        out = os.path.join(out_dir, f"pick_{i:02d}_{cn}_{tag}_{short_id}.png")
        Image.fromarray(_body(rows)).save(out, dpi=(220, 220), optimize=True)
        print(f"  -> {out}", flush=True)
        manifest.append(f"pick_{i:02d}\t{cn}\t{mid}\tΔCD={dcd:.2f}\t{out}")
        all_rows.extend(rows)

    with open(os.path.join(out_dir, "picks_manifest.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(manifest) + "\n")

    ov = os.path.join(out_dir, f"00_overview_{len(picks)}objects.png")
    Image.fromarray(_body(all_rows)).save(ov, dpi=(220, 220), optimize=True)
    print(f"overview -> {ov}", flush=True)
    print(f"done {out_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", action="store_true", help="scan PCN test (complete weights) then render")
    parser.add_argument("--n", type=int, default=10, help="number of picks when scanning")
    parser.add_argument("--out-subdir", type=str, default="", help="subdir under OUT_BASE, e.g. batch2")
    args = parser.parse_args()

    _set_chinese_font()
    out_dir = os.path.join(OUT_BASE, args.out_subdir) if args.out_subdir else OUT_BASE
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tmp = "/tmp" if os.path.isdir("/tmp") else _ROOT

    m1, m2, cfg, vis_args = _load_models(dev, tmp)

    if args.scan:
        print(f"Scanning PCN test (complete), exclude batch1 n={len(BANNED_IDS)} ...", flush=True)
        _, test_loader = builder.dataset_builder(vis_args, cfg.dataset.test)
        by_tax = scan_test(m1, m2, test_loader, dev, BANNED_IDS)
        print(f"Selecting {args.n} diverse picks (force boat {FORCE_INCLUDE_ID[:8]})...", flush=True)
        picks = select_diverse_picks(by_tax, args.n, FORCE_INCLUDE_ID)
        summary = os.path.join(out_dir, "scan_summary.txt")
        os.makedirs(out_dir, exist_ok=True)
        with open(summary, "w", encoding="utf-8") as f:
            f.write("# complete Stage-1 vs Stage-2 scan (excluded batch1)\n")
            for tax in ALL_TAX:
                f.write(f"\n[{TAXONOMY_CN[tax]}]\n")
                pool = sorted(by_tax.get(tax, []), key=lambda r: r["score"], reverse=True)
                for rank, rec in enumerate(pool[:8], 1):
                    f.write(
                        f"  {rank}\t{rec['model_id']}\tΔcd={rec['delta_cd']:.2f}\t"
                        f"Δmiss={rec['delta_miss']:.2f}\tscore={rec['score']:.2f}\n"
                    )
    else:
        picks = PICKS

    render_picks(picks, out_dir, dev, m1, m2)


if __name__ == "__main__":
    main()
