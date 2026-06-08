"""
PCN_core：Stage-1 (MSF Sigmoid ckpt-best) vs Stage-2 (FeedPoinTrS-FT ep50) 论文式 2×3 对比图。

Stage-2 权重与 test_test_exp5_epoch050 日志一致：
  exp_MSF_Pure_Group_sigmoid_feedpointrs_ft_crop02_04_seed42/ckpt-epoch-050.pth
  全测试集 F=0.8226, CDL1=7.0607

用法（在 completion 目录）:
  CUDA_VISIBLE_DEVICES=0 python -u Vision/plot_stage1_vs_stage2_pcn_core.py
  # 默认：沙发+台灯，不扫库、不加载 DataLoader
  python -u Vision/plot_stage1_vs_stage2_pcn_core.py --scan --n-extra 2   # 可选：再扫 2 例
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
import matplotlib.pyplot as plt
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
    _build_model,
    _load_gt_for_vis,
    _load_partial_for_vis,
    _set_chinese_font,
    _subsample,
    _to_tax,
)
from vis_compare_stage1_vs_stage2 import (
    VIEWS,
    _limits,
    _plot_scatter_3d,
    chamfer_l1_x1e3,
)

# PCN_core 权重（与 plot_stage1_sofa_table_4x4 / vis_compare_stage1_vs_stage2 一致）
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

# PCN_core 固定两例（与一阶段 4×4 沙发/台灯一致）
REFERENCE_PICKS = [
    ("沙发", "04256520", "2fc5cf498b0fa6de1525e8c8552c3a9c"),
    ("台灯", "03636649", "e1fe4f81f074abc3e6597d391ab6fcc1"),
]

SCAN_TAX_EXTRA = [
    "03636649",  # 台灯
    "04256520",  # 沙发
    "02933112",  # 橱柜
    "04379243",  # 桌子
    "03001627",  # 椅子
    "02958343",  # 汽车
]

# --no-scan 时的默认额外两例（可用 --scan 覆盖；台灯/桌子 Stage-2 改善较明显）
DEFAULT_EXTRA_FALLBACK = [
    ("台灯", "03636649", "e1fe4f81f074abc3e6597d391ab6fcc1"),
    ("桌子", "04379243", "ddb20a7778038d87f51f77a6d7299806"),
]

OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "output", "stage1_vs_stage2_pcn_core"
)
PICKS_JSON = os.path.join(OUT_DIR, "picks.json")

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
    log_name = "vis_stage1_vs_stage2_pcn_core"
    seed = 0
    deterministic = False
    sync_bn = False
    mode = None
    config = ""
    val_freq = 10


def _missing_g2p_x1e3(partial, pred, gt, device, miss_ratio=0.35):
    partial = torch.as_tensor(partial, dtype=torch.float32, device=device).unsqueeze(0)
    pred = torch.as_tensor(pred, dtype=torch.float32, device=device).unsqueeze(0)
    gt = torch.as_tensor(gt, dtype=torch.float32, device=device).unsqueeze(0)
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


def _load_models(root, device, tmp_root):
    print("Loading Stage-1 PCN_core MSF Sigmoid (ckpt-best)...", flush=True)
    model_s1, cfg, args = _build_model(
        os.path.join(root, CFG_STAGE1),
        os.path.join(root, CKPT_STAGE1),
        os.path.join(tmp_root, "tmp_vis_s1s2_core_s1"),
        "msf_pure_group_sigmoid",
    )
    time.sleep(0.3)
    print("Loading Stage-2 PCN_core FeedPoinTrS-FT (ckpt-epoch-050)...", flush=True)
    model_s2, _, _ = _build_model(
        os.path.join(root, CFG_STAGE2),
        os.path.join(root, CKPT_STAGE2),
        os.path.join(tmp_root, "tmp_vis_s1s2_core_s2"),
        "msf_pure_group_sigmoid",
    )
    return model_s1.to(device).eval(), model_s2.to(device).eval(), cfg, args


@torch.no_grad()
def _infer_sample(model_s1, model_s2, partial, gt_t, device):
    pred_s1 = model_s1(partial)[-1][0]
    pred_s2 = model_s2(partial)[-1][0]
    gt = gt_t[0]
    rec = {
        "cd_s1": chamfer_l1_x1e3(pred_s1, gt, device),
        "cd_s2": chamfer_l1_x1e3(pred_s2, gt, device),
        "f_s1": float(Metrics._get_f_score(pred_s1.unsqueeze(0), gt.unsqueeze(0)).item()),
        "f_s2": float(Metrics._get_f_score(pred_s2.unsqueeze(0), gt.unsqueeze(0)).item()),
        "miss_s1": _missing_g2p_x1e3(partial[0], pred_s1, gt, device),
        "miss_s2": _missing_g2p_x1e3(partial[0], pred_s2, gt, device),
        "pred_s1": pred_s1.detach().cpu().numpy(),
        "pred_s2": pred_s2.detach().cpu().numpy(),
        "gt": gt.detach().cpu().numpy(),
    }
    rec["delta_cd"] = rec["cd_s1"] - rec["cd_s2"]
    rec["delta_miss"] = rec["miss_s1"] - rec["miss_s2"]
    rec["score"] = 0.35 * rec["delta_cd"] + 0.55 * rec["delta_miss"] + 0.10 * (rec["f_s2"] - rec["f_s1"]) * 100.0
    return rec


def scan_extra_picks(model_s1, model_s2, test_loader, device, n_extra: int, min_delta_cd: float):
    ref_ids = {mid for _, _, mid in REFERENCE_PICKS}
    pool = []
    n = 0
    for taxonomy_ids, model_ids, data in test_loader:
        partial = data[0].to(device)
        gt_t = data[1]
        for i in range(partial.shape[0]):
            tax = _to_tax(taxonomy_ids[i])
            if tax not in SCAN_TAX_EXTRA:
                continue
            model_id = str(model_ids[i])
            if model_id in ref_ids:
                continue
            rec = _infer_sample(model_s1, model_s2, partial[i : i + 1], gt_t[i : i + 1], device)
            rec.update({"tax": tax, "model_id": model_id})
            pool.append(rec)
            n += 1
            if n % 80 == 0:
                print(f"  scanned {n} ...", flush=True)

    winners = [r for r in pool if r["cd_s2"] < r["cd_s1"] and r["delta_cd"] >= min_delta_cd]
    if len(winners) < n_extra:
        winners = [r for r in pool if r["cd_s2"] < r["cd_s1"]]
    if len(winners) < n_extra:
        winners = pool
    winners = sorted(winners, key=lambda r: r["score"], reverse=True)

    picked = []
    used_tax = set()
    for rec in winners:
        if rec["tax"] in used_tax:
            continue
        used_tax.add(rec["tax"])
        picked.append(rec)
        name = TAXONOMY_CN[rec["tax"]]
        print(
            f"  extra {name}: {rec['model_id'][:12]}...  "
            f"CD s1={rec['cd_s1']:.3f} s2={rec['cd_s2']:.3f} Δ={rec['delta_cd']:.3f}",
            flush=True,
        )
        if len(picked) >= n_extra:
            break
    return picked


@torch.no_grad()
def _sample_from_pick(root, device, model_s1, model_s2, tax, model_id):
    partial_np = _load_partial_for_vis(root, tax, model_id, 0)
    partial = torch.from_numpy(partial_np).unsqueeze(0).float().to(device)
    gt = _load_gt_for_vis(root, tax, model_id)
    gt_t = torch.from_numpy(gt).unsqueeze(0).float()
    rec = _infer_sample(model_s1, model_s2, partial, gt_t, device)
    return {
        "tax": tax,
        "model_id": model_id,
        "gt": gt,
        "pred_s1": rec["pred_s1"],
        "pred_s2": rec["pred_s2"],
        "cd_s1": rec["cd_s1"],
        "cd_s2": rec["cd_s2"],
    }


def draw_2x3(sample, out_path):
    gt = _subsample(sample["gt"], 6000, 23)
    pred_s1 = _subsample(sample["pred_s1"], 6000, 17)
    pred_s2 = _subsample(sample["pred_s2"], 6000, 19)
    lim_min, lim_max = _limits([gt, pred_s1, pred_s2])

    name = TAXONOMY_CN.get(sample["tax"], sample["tax"])
    delta = sample["cd_s1"] - sample["cd_s2"]
    delta_sign = "↓" if delta > 0 else "↑"

    fig = plt.figure(figsize=(14.0, 9.5), facecolor="white")
    fig.suptitle(
        f"{name}  |  {sample['model_id']}  "
        f"(Stage-1={sample['cd_s1']:.3f}, Stage-2={sample['cd_s2']:.3f}, Δ={delta_sign}{abs(delta):.3f})",
        fontsize=14,
        y=0.98,
        fontweight="bold",
    )
    pc_cols = [
        ("真值 (GT)", gt, "#2ca02c", 2.0, None),
        ("Stage-1 (纯组级 Sigmoid)", pred_s1, "#1f77b4", 3.5, sample["cd_s1"]),
        ("Stage-2 (主设置)", pred_s2, "#d62728", 3.5, sample["cd_s2"]),
    ]
    for r, (elev, azim) in enumerate(VIEWS):
        for c, (label, pts, color, size, cd) in enumerate(pc_cols):
            ax = fig.add_subplot(2, 3, r * 3 + c + 1, projection="3d")
            _plot_scatter_3d(ax, pts, color, size, elev, azim, lim_min, lim_max, label, cd)
    fig.text(
        0.5, 0.02,
        "Stage-2 相对 Stage-1 的 CDL1 改善用红色标注；数值越小越好",
        ha="center", fontsize=10, style="italic", color="#555555",
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _stitch_vertical(paths, out_path, gap_px=24):
    imgs = [np.array(Image.open(p).convert("RGB")) for p in paths]
    w = max(im.shape[1] for im in imgs)
    padded = []
    for im in imgs:
        if im.shape[1] < w:
            pad = np.ones((im.shape[0], w - im.shape[1], 3), dtype=np.uint8) * 255
            im = np.concatenate([im, pad], axis=1)
        padded.append(im)
    gap = np.ones((gap_px, w, 3), dtype=np.uint8) * 255
    out = padded[0]
    for im in padded[1:]:
        out = np.concatenate([out, gap, im], axis=0)
    Image.fromarray(out).save(out_path, dpi=(200, 200))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scan", action="store_true",
        help="scan test set for extra picks (slow; default off)",
    )
    parser.add_argument("--n-extra", type=int, default=0, help="extra samples beyond reference pair")
    parser.add_argument("--min-delta-cd", type=float, default=1.0)
    args = parser.parse_args()

    _set_chinese_font()
    root = _COMPLETION_ROOT
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tmp_root = "/tmp" if os.path.isdir("/tmp") else root
    os.makedirs(OUT_DIR, exist_ok=True)

    model_s1, model_s2, cfg, vis_args = _load_models(root, device, tmp_root)
    test_loader = None
    if args.scan and args.n_extra > 0:
        _, test_loader = builder.dataset_builder(vis_args, cfg.dataset.test)

    extra_recs = []
    if args.n_extra > 0 and args.scan and test_loader is not None:
        print(f"Scanning {len(SCAN_TAX_EXTRA)} categories for {args.n_extra} extra picks...", flush=True)
        extra_recs = scan_extra_picks(
            model_s1, model_s2, test_loader, device, args.n_extra, args.min_delta_cd
        )
    elif args.n_extra > 0:
        if os.path.isfile(PICKS_JSON):
            with open(PICKS_JSON, encoding="utf-8") as f:
                data = json.load(f)
            extra_recs = data.get("extra", [])
        if not extra_recs:
            extra_recs = [
                {"tax": tax, "model_id": mid}
                for _, tax, mid in DEFAULT_EXTRA_FALLBACK[: args.n_extra]
            ]
            print("Using DEFAULT_EXTRA_FALLBACK.", flush=True)

    all_picks = [
        {"tax": tax, "model_id": mid, "name_cn": name}
        for name, tax, mid in REFERENCE_PICKS
    ]
    for rec in extra_recs:
        all_picks.append({
            "tax": rec["tax"],
            "model_id": rec["model_id"],
            "name_cn": TAXONOMY_CN[rec["tax"]],
            "metrics": {k: rec[k] for k in ("cd_s1", "cd_s2", "delta_cd", "score") if k in rec},
        })

    with open(PICKS_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "stage1_ckpt": CKPT_STAGE1,
                "stage2_ckpt": CKPT_STAGE2,
                "reference": REFERENCE_PICKS,
                "extra": extra_recs,
                "all_picks": all_picks,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    rendered_paths = []
    for i, pick in enumerate(all_picks, 1):
        tax, mid = pick["tax"], pick["model_id"]
        name = pick["name_cn"]
        print(f"[{i}/{len(all_picks)}] {name} {mid[:12]}...", flush=True)
        sample = _sample_from_pick(root, device, model_s1, model_s2, tax, mid)
        out = os.path.join(OUT_DIR, f"{i:02d}_{tax}_{name}_{mid}.png")
        draw_2x3(sample, out)
        rendered_paths.append(out)
        print(f"  -> {out}", flush=True)

    if len(rendered_paths) >= 2:
        overview = os.path.join(OUT_DIR, "00_overview_stacked.png")
        _stitch_vertical(rendered_paths, overview)
        print(f"saved overview {overview}", flush=True)

    print(f"done -> {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
