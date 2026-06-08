#!/usr/bin/env python3
"""
Mine hard PCN samples by per-taxonomy relative CD (not global CD rank).

Outputs:
  data/PCN_hard/{split}_per_sample.csv
  data/PCN_hard/hard_{split}_per_class.txt
  data/PCN_hard/hard_train_ft_mix.txt  (hard×repeat + random fraction of full train)

Usage:
  cd completion
  python scripts/mine_hard_pc_samples.py --ckpts experiments/.../ckpt-best.pth
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from extensions.chamfer_dist import ChamferDistanceL1
from tools import builder
from utils.config import cfg_from_yaml_file
from utils.chamfer_loss_utils import chamfer_l1_cd_and_g2p


def _make_args(config_path: str, ckpt: str, exp_name: str):
    class A:
        pass

    a = A()
    a.config = config_path
    a.launcher = "none"
    a.local_rank = 0
    a.num_workers = 0
    a.seed = 0
    a.deterministic = False
    a.sync_bn = False
    a.exp_name = exp_name
    a.start_ckpts = ckpt
    a.ckpts = ckpt
    a.val_freq = 1
    a.model = "pgst"
    a.resume = False
    a.test = False
    a.mode = None
    a.experiment_path = os.path.join(ROOT, "tmp_mine_hard")
    a.tfboard_path = os.path.join(a.experiment_path, "TFBoard")
    a.log_name = "mine_hard"
    a.use_gpu = True
    a.distributed = False
    return a


@torch.no_grad()
def eval_split(model, dataloader, device):
    cd_fn = ChamferDistanceL1()
    rows = []
    for idx, (taxonomy_ids, model_ids, data) in enumerate(dataloader):
        tid = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
        mid = model_ids[0] if isinstance(model_ids[0], str) else model_ids[0].item()
        partial = data[0].to(device)
        gt = data[1].to(device)
        ret = model(partial)
        pred = ret[-1]
        cd_l1, g2p = chamfer_l1_cd_and_g2p(pred, gt)
        cd_legacy = cd_fn(pred, gt)
        rows.append({
            "taxonomy_id": tid,
            "model_id": mid,
            "cd_l1": float(cd_l1.item() * 1000.0),
            "g2p_l1": float(g2p.item() * 1000.0),
            "cd_l1_legacy": float(cd_legacy.item() * 1000.0),
            "idx": idx,
        })
        if (idx + 1) % 200 == 0:
            print(f"  [{idx + 1}/{len(dataloader)}] last cd_l1={rows[-1]['cd_l1']:.3f}", flush=True)
    return rows


def select_hard_per_class(rows, top_ratio: float, sigma_margin: float):
    """Per taxonomy: cd > mean + sigma_margin * std, capped by top_ratio fraction."""
    by_tax = defaultdict(list)
    for r in rows:
        by_tax[r["taxonomy_id"]].append(r)

    selected = []
    summary = {}
    for tid, items in by_tax.items():
        cds = np.array([x["cd_l1"] for x in items], dtype=np.float64)
        mu, std = cds.mean(), cds.std()
        thresh = mu + sigma_margin * std
        hard = [x for x in items if x["cd_l1"] >= thresh]
        n_top = max(1, int(len(items) * top_ratio))
        hard_sorted = sorted(hard, key=lambda x: x["cd_l1"], reverse=True)
        if len(hard_sorted) > n_top:
            hard_sorted = hard_sorted[:n_top]
        if len(hard_sorted) < max(1, n_top // 2):
            hard_sorted = sorted(items, key=lambda x: x["cd_l1"], reverse=True)[:n_top]
        for x in hard_sorted:
            x = dict(x)
            x["class_mean_cd"] = float(mu)
            x["class_std_cd"] = float(std)
            selected.append(x)
        summary[tid] = {"n": len(items), "n_hard": len(hard_sorted), "mean_cd": float(mu), "std_cd": float(std)}
    return selected, summary


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def write_id_list(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(f"{r['taxonomy_id']}/{r['model_id']}\n")


def load_id_set(path):
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def build_train_mix(hard_train_rows, full_train_path, repeat: int, random_ratio: float, seed: int, out_path):
    hard_keys = [f"{r['taxonomy_id']}/{r['model_id']}" for r in hard_train_rows]
    full_keys = list(load_id_set(full_train_path))
    rng = random.Random(seed)
    n_rand = max(1, int(len(full_keys) * random_ratio))
    rand_keys = rng.sample(full_keys, n_rand)

    mix = []
    for _ in range(repeat):
        mix.extend(hard_keys)
    mix.extend(rand_keys)
    rng.shuffle(mix)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for k in mix:
            f.write(k + "\n")
    return len(hard_keys), repeat, n_rand, len(mix)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid.yaml",
    )
    parser.add_argument(
        "--ckpts",
        default="experiments/AdaPoinTr_MSF_Pure_Group_sigmoid/PCN_models/exp_MSF_Pure_Group_sigmoid/ckpt-best.pth",
    )
    parser.add_argument("--out-dir", default="data/PCN_hard")
    parser.add_argument("--top-ratio", type=float, default=0.30, help="max fraction per class")
    parser.add_argument("--sigma-margin", type=float, default=0.5, help="cd > mean + margin*std")
    parser.add_argument("--hard-repeat", type=int, default=3)
    parser.add_argument("--random-ratio", type=float, default=0.20)
    parser.add_argument("--full-train-list", default="data/PCN_Core/pcn_core_train.txt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = cfg_from_yaml_file(args.config)
    cfg.model.NAME = "AdaPoinTr_PGST"

    model = builder.model_builder(cfg.model)
    builder.load_model(model, args.ckpts, logger=None)
    model.to(device).eval()
    model = torch.nn.DataParallel(model).cuda() if device.type == "cuda" else model

    os.makedirs(args.out_dir, exist_ok=True)

    for split in ("val", "train"):
        print(f"\n=== Mining split: {split} ===", flush=True)
        a = _make_args(args.config, args.ckpts, f"mine_{split}")
        cfg_ds = cfg_from_yaml_file(args.config)
        if split == "train":
            ds_cfg = cfg_ds.dataset.train
            ds_cfg.others.subset = "train"
            ds_cfg.others.bs = 1
            ds_cfg.others.sample_list_file = args.full_train_list
        else:
            ds_cfg = cfg_ds.dataset.val
            ds_cfg.others.subset = "val"
            ds_cfg.others.bs = 1
            if hasattr(ds_cfg.others, "sample_list_file"):
                del ds_cfg.others["sample_list_file"]

        _, loader = builder.dataset_builder(a, ds_cfg)
        rows = eval_split(model, loader, device)
        write_csv(
            os.path.join(args.out_dir, f"{split}_per_sample.csv"),
            rows,
            ["taxonomy_id", "model_id", "cd_l1", "g2p_l1", "cd_l1_legacy", "idx"],
        )
        hard_rows, summary = select_hard_per_class(rows, args.top_ratio, args.sigma_margin)
        write_id_list(os.path.join(args.out_dir, f"hard_{split}_per_class.txt"), hard_rows)
        print(f"  {split}: n={len(rows)} hard={len(hard_rows)}", flush=True)
        for tid, s in sorted(summary.items()):
            print(f"    {tid}: mean_cd={s['mean_cd']:.2f} hard={s['n_hard']}/{s['n']}", flush=True)

    hard_keys = sorted(load_id_set(os.path.join(args.out_dir, "hard_train_per_class.txt")))
    train_rows = [{"taxonomy_id": k.split("/")[0], "model_id": k.split("/")[1]} for k in hard_keys]

    n_hard, rep, n_rand, n_mix = build_train_mix(
        train_rows,
        args.full_train_list,
        args.hard_repeat,
        args.random_ratio,
        args.seed,
        os.path.join(args.out_dir, "hard_train_ft_mix.txt"),
    )
    print(
        f"\nWrote hard_train_ft_mix.txt: unique_hard={n_hard} x{rep} + random20%={n_rand} => total_lines={n_mix}",
        flush=True,
    )
    print(f"Done. Files under {args.out_dir}/", flush=True)


if __name__ == "__main__":
    main()
