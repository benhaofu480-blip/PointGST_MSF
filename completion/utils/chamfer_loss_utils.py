"""Chamfer L1 with optional GT->pred (G2P) coverage term, reusing one chamfer forward."""

import torch
from extensions.chamfer_dist import ChamferFunction


def g2p_cover_aggregate(
    d2: torch.Tensor,
    critical_ratio: float = 0.0,
    per_sample: bool = False,
):
    """
    G2P coverage term on sqrt(dist2), shape (B, M).

    critical_ratio in (0, 1): mean over top fraction of hardest GT points (MSPCN-style).
    critical_ratio <= 0 or >= 1: mean over all GT points (legacy L_cover).
    """
    if d2.dim() != 2:
        raise ValueError(f'd2 must be (B, M), got {tuple(d2.shape)}')

    if critical_ratio <= 0.0 or critical_ratio >= 1.0:
        return d2.mean(dim=1) if per_sample else d2.mean()

    num_points = d2.size(1)
    k = max(1, int(num_points * critical_ratio))
    sorted_d2, _ = torch.sort(d2, dim=1, descending=True)
    critical = sorted_d2[:, :k]
    return critical.mean(dim=1) if per_sample else critical.mean()


def chamfer_l1_cd_and_g2p(pred: torch.Tensor, gt: torch.Tensor, cover_critical_ratio: float = 0.0):
    """
    One chamfer forward. Returns symmetric CD-L1 and G2P mean (sqrt dist2).

    dist1: pred -> gt (P2G), dist2: gt -> pred (G2P), same as extensions.chamfer_dist.
    """
    dist1, dist2 = ChamferFunction.apply(pred, gt)
    d1 = torch.sqrt(dist1.clamp(min=1e-8))
    d2 = torch.sqrt(dist2.clamp(min=1e-8))
    cd_l1 = (d1.mean() + d2.mean()) * 0.5
    g2p = g2p_cover_aggregate(d2, critical_ratio=cover_critical_ratio, per_sample=False)
    return cd_l1, g2p


def chamfer_l1_with_cover(
    pred: torch.Tensor,
    gt: torch.Tensor,
    cover_weight: float = 0.0,
    cover_critical_ratio: float = 0.0,
):
    """loss = CD_L1 + cover_weight * G2P (optional critical-set top-ratio on GT points)."""
    cd_l1, g2p = chamfer_l1_cd_and_g2p(pred, gt, cover_critical_ratio=cover_critical_ratio)
    if cover_weight > 0:
        return cd_l1 + cover_weight * g2p
    return cd_l1


def chamfer_l1_per_sample(
    pred: torch.Tensor,
    gt: torch.Tensor,
    cover_weight: float = 0.0,
    cover_critical_ratio: float = 0.0,
):
    """
    Per-sample symmetric CD-L1 (+ optional G2P cover), shape (B,).

    dist1/dist2 from ChamferFunction are (B, N); mean over points per batch item.
    """
    dist1, dist2 = ChamferFunction.apply(pred, gt)
    d1 = torch.sqrt(dist1.clamp(min=1e-8))
    d2 = torch.sqrt(dist2.clamp(min=1e-8))
    cd_l1 = (d1.mean(dim=1) + d2.mean(dim=1)) * 0.5
    if cover_weight > 0:
        g2p = g2p_cover_aggregate(d2, critical_ratio=cover_critical_ratio, per_sample=True)
        return cd_l1 + cover_weight * g2p
    return cd_l1
