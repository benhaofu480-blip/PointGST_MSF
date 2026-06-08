#!/usr/bin/env python3
"""
FeedPoinTrS-style inference on PCN test: open-loop (C0) vs feedback second pass (C1).

Uses MSF Pure Group sigmoid (C) architecture; load any ckpt trained with that adapter.

  cd completion
  python scripts/test_feedpointrs_feedback.py \\
    --config cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid.yaml \\
    --ckpts experiments/AdaPoinTr_MSF_Pure_Group_sigmoid/PCN_models/exp_MSF_Pure_Group_sigmoid/ckpt-best.pth \\
    --tag sigmoid_s1
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from tools import builder
from utils.AverageMeter import AverageMeter
from utils.config import get_config
from utils.feedback_crop import complete_to_partial_input
from utils.logger import get_logger, print_log
from utils.metrics import Metrics
from utils.open3d_postprocess import is_postprocess_enabled, postprocess_dense_points


class TestArgs:
    launcher = 'none'
    local_rank = 0
    distributed = False
    use_gpu = True
    num_workers = 4
    test = True
    model = 'pgst'
    ckpts = ''
    start_ckpts = ''
    resume = False
    exp_name = ''
    experiment_path = ''
    tfboard_path = ''
    log_name = 'feedpointrs_feedback'
    seed = 0
    deterministic = False
    sync_bn = False
    mode = None
    config = ''
    val_freq = 10


def _build_args(config_path: str, ckpt: str, exp_name: str) -> TestArgs:
    a = TestArgs()
    a.config = config_path
    a.ckpts = ckpt
    a.exp_name = exp_name
    a.experiment_path = os.path.join(ROOT, 'experiments', 'FeedPoinTrS_feedback', 'PCN_models', exp_name)
    a.tfboard_path = os.path.join(a.experiment_path, 'TFBoard')
    os.makedirs(a.experiment_path, exist_ok=True)
    return a


def _crop_ratio_for_sample(idx: int, base_ratio: float, crop_random: bool, seed: int) -> float:
    if not crop_random:
        return base_ratio
    lo, hi = 0.25, 0.75
    rng = np.random.RandomState(seed + idx * 9973)
    return float(lo + (hi - lo) * rng.rand())


@torch.no_grad()
def run_test(
    base_model,
    test_dataloader,
    config,
    logger,
    crop_ratio: float,
    crop_random: bool,
    seed: int,
):
    base_model.eval()
    npoints = config.dataset.test._base_.N_POINTS
    n_input = 2048
    device = next(base_model.parameters()).device
    meters = {
        'openloop': AverageMeter(Metrics.names()),
        'feedback': AverageMeter(Metrics.names()),
    }
    cat_meters = {
        'openloop': {},
        'feedback': {},
    }
    n_samples = len(test_dataloader)

    for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
        taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
        model_id = model_ids[0]

        partial = data[0].to(device)
        gt = data[1].to(device)

        ret0 = base_model(partial)
        dense0 = ret0[-1]
        if is_postprocess_enabled(config):
            dense0 = postprocess_dense_points(dense0, config, logger=logger)

        sample_seed = seed + idx * 9973
        ratio = _crop_ratio_for_sample(idx, crop_ratio, crop_random, seed)
        partial1 = complete_to_partial_input(dense0, n_input, ratio, seed=sample_seed)
        ret1 = base_model(partial1)
        dense1 = ret1[-1]
        if is_postprocess_enabled(config):
            dense1 = postprocess_dense_points(dense1, config, logger=logger)

        m0 = Metrics.get(dense0, gt, require_emd=True)
        m1 = Metrics.get(dense1, gt, require_emd=True)

        meters['openloop'].update(m0)
        meters['feedback'].update(m1)

        for mode, m in (('openloop', m0), ('feedback', m1)):
            if taxonomy_id not in cat_meters[mode]:
                cat_meters[mode][taxonomy_id] = AverageMeter(Metrics.names())
            cat_meters[mode][taxonomy_id].update(m)

        if (idx + 1) % 200 == 0:
            print_log(
                f'[{idx + 1}/{n_samples}] {taxonomy_id}/{model_id} '
                f'crop_ratio={ratio:.3f} '
                f'CDL1 open={m0[1]:.4f} feed={m1[1]:.4f}',
                logger=logger,
            )

    try:
        shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    except FileNotFoundError:
        shapenet_dict = {}

    for mode in ('openloop', 'feedback'):
        print_log(f'========== {mode.upper()} ({"C0" if mode == "openloop" else "C1"}) ==========', logger=logger)
        print_log('[TEST] Metrics = %s' % (['%.4f' % m for m in meters[mode].avg()]), logger=logger)
        msg = 'Taxonomy\t#Sample\t' + '\t'.join(Metrics.names()) + '\t#ModelName'
        print_log(msg, logger=logger)
        for taxonomy_id in sorted(cat_meters[mode].keys()):
            vals = cat_meters[mode][taxonomy_id].avg()
            print_log(
                f'{taxonomy_id}\t{cat_meters[mode][taxonomy_id].count(0)}\t'
                + '\t'.join('%.3f' % v for v in vals)
                + f'\t{shapenet_dict.get(taxonomy_id, taxonomy_id)}',
                logger=logger,
            )
        print_log('Overall\t\t' + '\t'.join('%.3f' % v for v in meters[mode].avg()), logger=logger)

    return meters['openloop'].avg(), meters['feedback'].avg()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid.yaml')
    ap.add_argument('--ckpts', required=True)
    ap.add_argument('--tag', default='run', help='log subfolder name')
    ap.add_argument('--crop_ratio', type=float, default=0.30, help='fixed crop ratio if not --crop_random')
    ap.add_argument('--crop_random', action='store_true', help='sample ratio in [0.25, 0.75] per sample')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--num_workers', type=int, default=4)
    args = ap.parse_args()

    ckpt = args.ckpts if os.path.isabs(args.ckpts) else os.path.join(ROOT, args.ckpts)
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(ckpt)

    targs = _build_args(args.config, ckpt, args.tag)
    targs.num_workers = args.num_workers
    logger = get_logger(targs.log_name, log_file=os.path.join(targs.experiment_path, 'test.log'))
    print_log(f'FeedPoinTrS feedback test tag={args.tag}', logger=logger)
    print_log(f'config={args.config} ckpts={ckpt}', logger=logger)
    print_log(f'crop_ratio={args.crop_ratio} crop_random={args.crop_random} seed={args.seed}', logger=logger)

    cfg = get_config(targs, logger=logger)
    cfg.model.NAME = 'AdaPoinTr_PGST'
    cfg.model.encoder_config.pop('use_msf', None)
    cfg.model.msf_route_mode = 'none'
    cfg.model.use_msf_route_to_decoder = False

    _, test_loader = builder.dataset_builder(targs, cfg.dataset.test)
    model = builder.model_builder(cfg.model)
    builder.load_model(model, ckpt, logger=logger)
    if targs.use_gpu:
        model.cuda()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_test(
        model,
        test_loader,
        cfg,
        logger,
        crop_ratio=args.crop_ratio,
        crop_random=args.crop_random,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
