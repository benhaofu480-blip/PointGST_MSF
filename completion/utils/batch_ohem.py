"""Batch-level OHEM: reweight or select hard samples within a minibatch (no dataset relist)."""

import torch


def ohem_enabled(config) -> bool:
    ohem = getattr(config, 'ohem', None)
    return ohem is not None and bool(getattr(ohem, 'enabled', False))


def apply_batch_ohem(per_sample_loss: torch.Tensor, config) -> torch.Tensor:
    """
    Aggregate per-sample losses for backward.

    Modes (config.ohem.mode):
      - hard: mean of top keep_ratio fraction (classic OHEM-style within batch)
      - soft: mean with weights proportional to loss (capped by soft_max_weight)
    """
    ohem = getattr(config, 'ohem', None)
    keep_ratio = float(getattr(ohem, 'keep_ratio', 0.7))
    mode = str(getattr(ohem, 'mode', 'hard')).lower()

    if per_sample_loss.dim() != 1:
        raise ValueError(f'per_sample_loss must be 1D (B,), got shape {tuple(per_sample_loss.shape)}')

    if mode == 'soft':
        max_w = float(getattr(ohem, 'soft_max_weight', 2.0))
        denom = per_sample_loss.detach().mean().clamp(min=1e-8)
        w = (per_sample_loss / denom).clamp(max=max_w)
        return (per_sample_loss * w).mean()

    k = max(1, int(per_sample_loss.size(0) * keep_ratio))
    sorted_vals, _ = torch.sort(per_sample_loss, descending=True)
    return sorted_vals[:k].mean()
