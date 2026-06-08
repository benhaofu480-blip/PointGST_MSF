"""Dynamic hard-sample mining during stage-2 training (per-class CD on core train)."""

from __future__ import annotations

import copy
import csv
import gc
import os
import random
from collections import defaultdict

import numpy as np
import torch

from extensions.chamfer_dist import ChamferDistanceL1
from tools import builder
from utils.chamfer_loss_utils import chamfer_l1_cd_and_g2p
from utils.logger import print_log


def load_id_set(path):
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def select_hard_per_class(rows, top_ratio: float, sigma_margin: float):
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
        selected.extend(hard_sorted)
        summary[tid] = {"n": len(items), "n_hard": len(hard_sorted), "mean_cd": float(mu)}
    return selected, summary


def build_train_mix(hard_rows, full_train_path, repeat, random_ratio, seed, out_path):
    hard_keys = [f"{r['taxonomy_id']}/{r['model_id']}" for r in hard_rows]
    full_keys = list(load_id_set(full_train_path))
    rng = random.Random(seed)
    n_rand = max(1, int(len(full_keys) * random_ratio))
    rand_keys = rng.sample(full_keys, n_rand)

    mix = []
    for _ in range(repeat):
        mix.extend(hard_keys)
    mix.extend(rand_keys)
    rng.shuffle(mix)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for k in mix:
            f.write(k + "\n")
    return len(hard_keys), n_rand, len(mix)


@torch.no_grad()
def _eval_train_split(model, args, train_ds_cfg, config, device, logger=None):
    mine_args = copy.copy(args)
    mine_args.num_workers = 0
    ds_cfg = copy.deepcopy(train_ds_cfg)
    ds_cfg.others.subset = "train"
    ds_cfg.others.bs = 1
    ds_cfg.others.sample_list_file = config.dynamic_hard_mining.full_train_list

    _, loader = builder.dataset_builder(mine_args, ds_cfg)
    cd_fn = ChamferDistanceL1()
    rows = []
    n = len(loader)
    print_log(f"[DynamicHard] Scoring {n} core-train samples ...", logger=logger)
    for idx, (taxonomy_ids, model_ids, data) in enumerate(loader):
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
        })
        if (idx + 1) % 500 == 0:
            print_log(f"[DynamicHard]   {idx + 1}/{n}", logger=logger)
    return rows


def remine_and_rebuild_train_mix(base_model, args, config, epoch, logger=None):
    """Mine hard train ids with current model; write new mix list for following epochs."""
    dh = config.dynamic_hard_mining
    device = torch.device(f"cuda:{args.local_rank}" if args.use_gpu else "cpu")

    was_training = base_model.training
    base_model.eval()

    rows = _eval_train_split(
        base_model, args, config.dataset.train, config, device, logger=logger,
    )
    hard_rows, summary = select_hard_per_class(
        rows,
        float(dh.top_ratio),
        float(dh.sigma_margin),
    )

    out_dir = os.path.join(
        args.experiment_path,
        str(getattr(dh, "subdir", "dynamic_hard")),
    )
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"train_per_sample_ep{epoch:03d}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["taxonomy_id", "model_id", "cd_l1", "g2p_l1", "cd_l1_legacy"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    hard_txt = os.path.join(out_dir, f"hard_train_ep{epoch:03d}.txt")
    with open(hard_txt, "w") as f:
        for r in hard_rows:
            f.write(f"{r['taxonomy_id']}/{r['model_id']}\n")

    mix_path = os.path.join(out_dir, f"hard_train_ft_mix_ep{epoch:03d}.txt")
    n_hard, n_rand, n_mix = build_train_mix(
        hard_rows,
        dh.full_train_list,
        int(dh.hard_repeat),
        float(dh.random_ratio),
        int(getattr(dh, "mix_seed", 42)),
        mix_path,
    )

    print_log(
        f"[DynamicHard] epoch {epoch}: core_train={len(rows)} hard={n_hard} "
        f"mix_lines={n_mix} (x{dh.hard_repeat} + rand{int(float(dh.random_ratio)*100)}%) -> {mix_path}",
        logger=logger,
    )
    for tid, s in sorted(summary.items()):
        print_log(
            f"[DynamicHard]   {tid}: mean_cd={s['mean_cd']:.2f} hard={s['n_hard']}/{s['n']}",
            logger=logger,
        )

    if was_training:
        base_model.train()
    return mix_path


def shutdown_dataloader(dataloader):
    if dataloader is None:
        return
    try:
        it = getattr(dataloader, "_iterator", None)
        if it is not None:
            it._shutdown_workers()
    except Exception:
        pass
    del dataloader


def reload_train_dataloader(args, config, old_loader=None, logger=None):
    shutdown_dataloader(old_loader)
    gc.collect()
    train_sampler, train_dataloader = builder.dataset_builder(args, config.dataset.train)
    n = len(train_dataloader.dataset)
    print_log(
        f"[DynamicHard] Reloaded train loader: {n} files, list={config.dataset.train.others.sample_list_file}",
        logger=logger,
    )
    return train_sampler, train_dataloader
