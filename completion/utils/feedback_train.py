"""FeedPoinTrS training: two forward passes with 2:1 combined loss."""

from __future__ import annotations

import torch

from utils.feedback_crop import complete_to_partial_input


def _core_model(model):
    """Unwrap DataParallel / DDP for get_loss; forward still uses wrapped model."""
    return model.module if hasattr(model, 'module') else model


def feedback_enabled(config) -> bool:
    fb = getattr(config, 'feedback_training', None)
    return fb is not None and bool(getattr(fb, 'enabled', False))


def _sample_crop_ratio(config) -> float:
    fb = config.feedback_training
    lo = float(getattr(fb, 'crop_ratio_min', 0.20))
    hi = float(getattr(fb, 'crop_ratio_max', 0.40))
    if hi < lo:
        lo, hi = hi, lo
    return lo + (hi - lo) * torch.rand(1).item()


def feedback_training_loss(model, partial: torch.Tensor, gt: torch.Tensor, epoch: int, config):
    """
    Pass0: partial -> C0, Pass1: gv(C0) -> C1.
    Loss = (w0 * L0 + w1 * L1) / (w0 + w1), default w0:w1 = 2:1 (FeedPoinTrS).
    Crop is non-differentiable; C0 is detached before gv.
    """
    fb = config.feedback_training
    w0 = float(getattr(fb, 'pass_weight_first', 2.0))
    w1 = float(getattr(fb, 'pass_weight_second', 1.0))
    norm = max(w0 + w1, 1e-8)

    core = _core_model(model)
    ret0 = model(partial)
    sparse0, dense0 = core.get_loss(ret0, gt, epoch)
    loss0 = sparse0 + dense0

    dense0_pred = ret0[-1].detach()
    crop_ratio = _sample_crop_ratio(config)
    partial1 = complete_to_partial_input(
        dense0_pred,
        partial.size(1),
        crop_ratio,
        seed=None,
        gt=gt,
        config=config,
    )

    # Backward pass0 before pass1 forward to avoid inplace conflicts on shared weights.
    scaled0 = (w0 * loss0) / norm
    scaled0.backward()

    ret1 = model(partial1)
    sparse1, dense1 = core.get_loss(ret1, gt, epoch)
    loss1 = sparse1 + dense1
    scaled1 = (w1 * loss1) / norm
    scaled1.backward()

    sparse_loss = (w0 * sparse0 + w1 * sparse1) / norm
    dense_loss = (w0 * dense0 + w1 * dense1) / norm
    total = sparse_loss + dense_loss
    return sparse_loss, dense_loss, total
