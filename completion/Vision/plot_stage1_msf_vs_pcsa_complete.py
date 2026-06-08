"""
一阶段 MSF vs PCSA（complete ep150 best）对比图。

每类 1 例，4 列：输入 → PCSA → MSF → G.T.

用法:
  python Vision/plot_stage1_msf_vs_pcsa_complete.py
  python Vision/plot_stage1_msf_vs_pcsa_complete.py --rescan   # 扫 test 1200，每类选 MSF CD 更优且 Δ 最大
  python Vision/plot_stage1_msf_vs_pcsa_complete.py --elev 12 --azim -120
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
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

_COMPLETION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _COMPLETION_ROOT not in sys.path:
    sys.path.insert(0, _COMPLETION_ROOT)

from extensions.chamfer_dist import ChamferDistanceL1
from tools import builder
from utils.config import get_config

_CD = ChamferDistanceL1()
VISION_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

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
DEFAULT_PICKS = "data/stage1_complete_vis_8cat.txt"
RESCAN_PICKS = "data/stage1_complete_vis_8cat_rescan.txt"

# 侧视：避免正俯视“积木块”感（原 18,45 偏正面）
DEFAULT_ELEV = 10.0
DEFAULT_AZIM = 90.0

COLORS = {"input": "#7f7f7f", "pcsa": "#1f77b4", "msf": "#d62728", "gt": "#E8A020"}
SIZES = {"input": 2.5, "pcsa": 4.5, "msf": 4.5, "gt": 2.0}


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
    log_name = "vis_stage1_complete"
    seed = 0
    deterministic = False
    sync_bn = False
    mode = None
    config = ""
    val_freq = 10


def _set_chinese_font():
    for path, family in (
        ("/usr/share/fonts/truetype/arphic/uming.ttc", "AR PL UMing CN"),
        ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "Noto Sans CJK SC"),
    ):
        if os.path.exists(path):
            plt.rcParams["font.sans-serif"] = [family, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return family
    plt.rcParams["axes.unicode_minus"] = False
    return None


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


def _to_tax(v):
    if torch.is_tensor(v):
        return str(int(v.item())) if v.dtype == torch.int64 else str(v.item())
    return str(v)


def _subsample(pts, n, seed):
    if pts.shape[0] <= n:
        return pts
    rng = np.random.RandomState(seed)
    return pts[rng.choice(pts.shape[0], n, replace=False)]


def _limits(pts_list):
    all_pts = np.concatenate(pts_list, axis=0)
    c = all_pts.mean(0)
    s = float((all_pts.max(0) - all_pts.min(0)).max())
    pad = max(s * 0.55, 1e-3)
    return c - pad, c + pad


def _plot_pc(ax, pts, color, size, elev, azim, lim_min, lim_max, title="", note=None):
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
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=3)
    if note:
        ax.text2D(
            0.03, 0.94, note, transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.88, edgecolor="none"),
        )


def _cd_l1_x1e3(pred, gt, device):
    pred = torch.as_tensor(pred, dtype=torch.float32, device=device).unsqueeze(0)
    gt = torch.as_tensor(gt, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        return float(_CD(pred, gt).item() * 1000.0)


def _load_picks(root: str, rel_path: str) -> list[tuple[str, str]]:
    path = os.path.join(root, rel_path)
    picks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tax, mid = line.split("/", 1)
            picks.append((tax, mid))
    return picks


def _save_picks(root: str, rel_path: str, records: list[dict]):
    path = os.path.join(root, rel_path)
    lines = ["# taxonomy_id/model_id  (auto-selected for MSF vs PCSA vis)"]
    for tax in ALL_TAXONOMIES:
        rec = next(r for r in records if r["tax"] == tax)
        lines.append(f"{tax}/{rec['model_id']}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote picks -> {path}")


def _pick_best_per_tax(pool: list[dict]) -> dict:
    """每类优先 MSF CD 更低，且 ΔCD 尽量大。"""
    winners = [r for r in pool if r["cd_msf"] < r["cd_pcsa"]]
    if winners:
        return max(winners, key=lambda r: r["cd_pcsa"] - r["cd_msf"])
    return min(pool, key=lambda r: r["cd_msf"] - r["cd_pcsa"])


@torch.no_grad()
def _infer_one(model_msf, model_pcsa, partial, gt_t, gt, tax, model_id, device):
    out_m = model_msf(partial)
    out_p = model_pcsa(partial)
    pred_m = out_m[-1][0]
    pred_p = out_p[-1][0]
    rec = {
        "tax": tax,
        "model_id": str(model_id),
        "partial": _to_np(partial[0]),
        "gt": gt,
        "pred_msf": _to_np(pred_m),
        "pred_pcsa": _to_np(pred_p),
        "cd_msf": _cd_l1_x1e3(pred_m, gt_t, device),
        "cd_pcsa": _cd_l1_x1e3(pred_p, gt_t, device),
    }
    rec["delta_cd"] = rec["cd_pcsa"] - rec["cd_msf"]
    return rec


def draw_grid(selected: list[dict], out_path: str, elev: float, azim: float, dpi: int = 200):
    n = len(selected)
    fig = plt.figure(figsize=(18.0, 3.6 * n), facecolor="white")
    gs = GridSpec(n, 4, figure=fig, hspace=0.05, wspace=0.05)

    col_defs = [
        ("输入", "partial", "input", SIZES["input"], None),
        ("PCSA", "pred_pcsa", "pcsa", SIZES["pcsa"], "cd_pcsa"),
        ("MSF", "pred_msf", "msf", SIZES["msf"], "cd_msf"),
        ("G.T.", "gt", "gt", SIZES["gt"], None),
    ]

    for row, sample in enumerate(selected):
        partial = _subsample(sample["partial"], 2048, 11 + row)
        gt = _subsample(sample["gt"], 6000, 23 + row)
        pred_p = _subsample(sample["pred_pcsa"], 6000, 17 + row)
        pred_m = _subsample(sample["pred_msf"], 6000, 19 + row)
        lim_min, lim_max = _limits([partial, gt, pred_p, pred_m])

        tax = sample["tax"]
        name = TAXONOMY_CN.get(tax, tax)

        for c, (label, key, color_key, size, metric_key) in enumerate(col_defs):
            ax = fig.add_subplot(gs[row, c], projection="3d")
            if key == "partial":
                pts = partial
            elif key == "gt":
                pts = gt
            elif key == "pred_pcsa":
                pts = pred_p
            else:
                pts = pred_m

            note = f"CD×1e3={sample[metric_key]:.2f}" if metric_key else None
            title = label if row == 0 else ""
            _plot_pc(
                ax, pts, COLORS[color_key], size, elev, azim, lim_min, lim_max,
                title=title, note=note,
            )
            if c == 0:
                ax.text2D(
                    0.02, 0.04, f"{name}",
                    transform=ax.transAxes, fontsize=11, fontweight="bold",
                )

    fig.suptitle(
        f"一阶段：PCSA vs MSF  |  视角 elev={elev:.0f}° azim={azim:.0f}°  |  "
        "Test F: PCSA 0.836 / MSF 0.839",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=os.path.join(VISION_OUT, "stage1_msf_vs_pcsa_complete_8cat.png"),
    )
    parser.add_argument("--picks", type=str, default=DEFAULT_PICKS, help="相对 completion 的样本列表")
    parser.add_argument(
        "--rescan",
        action="store_true",
        help="扫描 PCN test 全集，每类选 MSF 更优且 ΔCD 最大的样本",
    )
    parser.add_argument("--elev", type=float, default=DEFAULT_ELEV)
    parser.add_argument("--azim", type=float, default=DEFAULT_AZIM)
    parser.add_argument("--dpi", type=int, default=200)
    cli = parser.parse_args()

    _set_chinese_font()
    root = _COMPLETION_ROOT
    picks_rel = RESCAN_PICKS if cli.rescan else cli.picks

    ckpt_msf = os.path.join(root, CKPT_MSF)
    ckpt_pcsa = os.path.join(root, CKPT_PCSA)
    cfg_msf = os.path.join(root, CFG_MSF)
    cfg_pcsa = os.path.join(root, CFG_PCSA)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  view=({cli.elev}, {cli.azim})  rescan={cli.rescan}")

    print("Loading MSF complete...")
    model_msf, cfg, args = _build_model(
        cfg_msf, ckpt_msf,
        os.path.join(root, "tmp_vis_complete_msf"),
        adapter_mode="msf_pure_group_sigmoid",
    )
    time.sleep(0.5)
    print("Loading PCSA complete...")
    model_pcsa, _, _ = _build_model(
        cfg_pcsa, ckpt_pcsa,
        os.path.join(root, "tmp_vis_complete_pcsa"),
        adapter_mode="pcsa",
    )
    model_msf = model_msf.to(device).eval()
    model_pcsa = model_pcsa.to(device).eval()

    if cli.rescan:
        cfg.dataset.test.others.pop("sample_list_file", None)
        _, test_loader = builder.dataset_builder(args, cfg.dataset.test)
        print(f"RESCAN: infer PCN test ({len(test_loader)} samples)...")
        by_tax: dict[str, list[dict]] = defaultdict(list)
        for i, (taxonomy_ids, model_ids, data) in enumerate(test_loader):
            partial = data[0].to(device)
            gt_t = data[1][0]
            gt = _to_np(gt_t)
            tax = _to_tax(taxonomy_ids[0])
            model_id = str(model_ids[0])
            rec = _infer_one(model_msf, model_pcsa, partial, gt_t, gt, tax, model_id, device)
            by_tax[tax].append(rec)
            if (i + 1) % 100 == 0 or i + 1 == len(test_loader):
                print(f"  progress {i + 1}/{len(test_loader)}")
        selected = []
        for tax in ALL_TAXONOMIES:
            pool = by_tax.get(tax, [])
            if not pool:
                raise RuntimeError(f"no test samples for {tax}")
            best = _pick_best_per_tax(pool)
            selected.append(best)
            name = TAXONOMY_CN.get(tax, tax)
            print(
                f"  PICK {name}: {best['model_id']}  "
                f"CD PCSA={best['cd_pcsa']:.2f} MSF={best['cd_msf']:.2f} Δ={best['delta_cd']:.2f}"
            )
        _save_picks(root, RESCAN_PICKS, selected)
    else:
        picks_file = os.path.join(root, picks_rel)
        if not os.path.isfile(picks_file):
            raise FileNotFoundError(picks_file)
        cfg.dataset.test.others.sample_list_file = picks_file
        _, test_loader = builder.dataset_builder(args, cfg.dataset.test)
        selected = []
        for i, (taxonomy_ids, model_ids, data) in enumerate(test_loader):
            partial = data[0].to(device)
            gt_t = data[1][0]
            gt = _to_np(gt_t)
            tax = _to_tax(taxonomy_ids[0])
            model_id = str(model_ids[0])
            rec = _infer_one(model_msf, model_pcsa, partial, gt_t, gt, tax, model_id, device)
            selected.append(rec)
            print(
                f"  [{i + 1}] {TAXONOMY_CN.get(tax, tax)}  "
                f"CD PCSA={rec['cd_pcsa']:.2f} MSF={rec['cd_msf']:.2f} Δ={rec['delta_cd']:.2f}"
            )
        selected.sort(key=lambda r: ALL_TAXONOMIES.index(r["tax"]))

    out_path = cli.output
    if cli.rescan and cli.output == os.path.join(VISION_OUT, "stage1_msf_vs_pcsa_complete_8cat.png"):
        out_path = os.path.join(VISION_OUT, "stage1_msf_vs_pcsa_complete_8cat_rescan.png")

    draw_grid(selected, out_path, elev=cli.elev, azim=cli.azim, dpi=cli.dpi)
    print(f"saved {os.path.abspath(out_path)}")

    meta_path = os.path.splitext(out_path)[0] + ".json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "tax": s["tax"],
                    "name": TAXONOMY_CN.get(s["tax"], s["tax"]),
                    "model_id": s["model_id"],
                    "cd_pcsa": s["cd_pcsa"],
                    "cd_msf": s["cd_msf"],
                    "delta_cd": s["delta_cd"],
                    "view": [cli.elev, cli.azim],
                }
                for s in selected
            ],
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"meta {meta_path}")


if __name__ == "__main__":
    main()
