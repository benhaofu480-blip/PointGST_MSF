#!/usr/bin/env python3
"""Sanity check for random vs error_aware feedback crop."""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from utils.feedback_crop import (
    CROP_MODE_ERROR,
    CROP_MODE_RANDOM,
    complete_to_partial_input,
    error_aware_centroid,
    pred_point_errors,
    random_unit_centroid,
    resolve_crop_centroid,
)


class _FbCfg:
    def __init__(self, crop_mode, error_top_ratio=0.25, error_mix_random=0.0):
        self.crop_mode = crop_mode
        self.error_top_ratio = error_top_ratio
        self.error_mix_random = error_mix_random


class _Cfg:
    def __init__(self, crop_mode):
        self.feedback_training = _FbCfg(crop_mode)


def _synthetic_bad_completion(gt: torch.Tensor) -> torch.Tensor:
    """Shift half of GT along +x to create a high-error region."""
    b, m, _ = gt.shape
    n = min(4096, m)
    pred = gt[:, :n, :].clone()
    pred[:, n // 2 :, 0] += 0.35
    return pred


def main() -> int:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gt = torch.randn(2, 2048, 3, device=device)
    gt = gt / gt.norm(dim=-1, keepdim=True).clamp(min=1e-6) * 0.5
    pred = _synthetic_bad_completion(gt)

    errors = pred_point_errors(pred, gt)
    assert errors.shape == (2, pred.size(1))

    c_rand = random_unit_centroid(2, device, seed=123)
    c_err = error_aware_centroid(pred, gt, top_ratio=0.25, mix_random=0.0, seed=123)
    assert c_rand.shape == (2, 3)
    assert c_err.shape == (2, 3)

    # High-error points are on +x side; centroid x should be larger than random mean |x|.
    assert c_err[:, 0].mean().item() > abs(c_rand[:, 0].mean().item()) - 0.2

    cfg_rand = _Cfg(CROP_MODE_RANDOM)
    cfg_err = _Cfg(CROP_MODE_ERROR)
    c0 = resolve_crop_centroid(pred, gt, cfg_rand, seed=7)
    c1 = resolve_crop_centroid(pred, gt, cfg_err, seed=7)
    assert torch.allclose(c0, random_unit_centroid(2, device, seed=7), atol=1e-5)

    partial_rand = complete_to_partial_input(pred, 512, 0.3, seed=11, gt=gt, config=cfg_rand)
    partial_err = complete_to_partial_input(pred, 512, 0.3, seed=11, gt=gt, config=cfg_err)
    assert partial_rand.shape == (2, 512, 3)
    assert partial_err.shape == (2, 512, 3)
    assert not torch.allclose(partial_rand, partial_err, atol=1e-4)

    print('OK: error_aware crop module switch works (random vs error_aware differ).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
