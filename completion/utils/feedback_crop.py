"""FeedPoinTrS-style point cloud cropper (gv): synthetic partial from first completion."""

from __future__ import annotations

import torch

from extensions.chamfer_dist import ChamferFunction
from utils import misc


CROP_MODE_RANDOM = 'random'
CROP_MODE_ERROR = 'error_aware'
SUPPORTED_CROP_MODES = (CROP_MODE_RANDOM, CROP_MODE_ERROR)


def normalize_centroid(centroid: torch.Tensor) -> torch.Tensor:
    return centroid / centroid.norm(dim=1, keepdim=True).clamp(min=1e-8)


def random_unit_centroid(batch: int, device: torch.device, seed: int | None = None) -> torch.Tensor:
    """Centroid on unit sphere (FeedPoinTrS Sec. 3.3)."""
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))
        c = torch.randn(batch, 3, device=device, generator=g)
    else:
        c = torch.randn(batch, 3, device=device)
    return normalize_centroid(c)


def pred_point_errors(complete: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Per predicted point P2G error sqrt(min dist to GT), shape (B, N)."""
    if complete.dim() != 3 or gt.dim() != 3:
        raise ValueError(f'complete/gt must be (B,N,3), got {tuple(complete.shape)}, {tuple(gt.shape)}')
    dist1, _ = ChamferFunction.apply(complete, gt)
    return torch.sqrt(dist1.clamp(min=1e-8))


def error_aware_centroid(
    complete: torch.Tensor,
    gt: torch.Tensor,
    top_ratio: float = 0.25,
    mix_random: float = 0.0,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Build crop centroid from high-error predicted points (P2G distance to GT).

    Uses mean of top-k error points, normalized to unit sphere (same geometry as random crop).
    mix_random in [0,1]: blend with random unit centroid for diversity / ablation.
    """
    b, n, _ = complete.shape
    if gt.shape[0] != b:
        raise ValueError(f'batch mismatch: complete {b}, gt {gt.shape[0]}')

    errors = pred_point_errors(complete, gt)
    k = max(1, int(n * float(top_ratio)))
    k = min(k, n)
    _, top_idx = torch.topk(errors, k=k, dim=1, largest=True, sorted=False)

    idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, 3)
    top_pts = torch.gather(complete, 1, idx_exp)
    centroid = top_pts.mean(dim=1)
    norm = centroid.norm(dim=1, keepdim=True)
    fallback = norm.squeeze(-1) < 1e-6
    if fallback.any():
        rand_c = random_unit_centroid(b, complete.device, seed=seed)
        centroid = torch.where(fallback.unsqueeze(-1), rand_c, centroid)
        norm = centroid.norm(dim=1, keepdim=True)
    centroid = centroid / norm.clamp(min=1e-8)

    mix = float(max(0.0, min(1.0, mix_random)))
    if mix > 0.0:
        rand_c = random_unit_centroid(b, complete.device, seed=seed)
        centroid = normalize_centroid((1.0 - mix) * centroid + mix * rand_c)
    return centroid


def resolve_crop_mode(config) -> str:
    fb = getattr(config, 'feedback_training', None)
    mode = str(getattr(fb, 'crop_mode', CROP_MODE_RANDOM) or CROP_MODE_RANDOM).lower()
    if mode not in SUPPORTED_CROP_MODES:
        raise ValueError(f'Unsupported feedback crop_mode={mode!r}, expected one of {SUPPORTED_CROP_MODES}')
    return mode


def resolve_crop_centroid(
    complete: torch.Tensor,
    gt: torch.Tensor | None,
    config,
    seed: int | None = None,
) -> torch.Tensor:
    """Pick crop centroid according to feedback_training.crop_mode."""
    mode = resolve_crop_mode(config)
    if mode == CROP_MODE_RANDOM:
        return random_unit_centroid(complete.size(0), complete.device, seed=seed)

    if gt is None:
        raise ValueError('error_aware crop requires gt for pass-1 centroid selection')
    fb = config.feedback_training
    top_ratio = float(getattr(fb, 'error_top_ratio', 0.25))
    mix_random = float(getattr(fb, 'error_mix_random', 0.0))
    return error_aware_centroid(
        complete,
        gt,
        top_ratio=top_ratio,
        mix_random=mix_random,
        seed=seed,
    )


def crop_partial_from_complete(
    complete: torch.Tensor,
    crop_ratio: float,
    centroid: torch.Tensor | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    """
    Remove points closest to centroid; return remaining points.

    Args:
        complete: (B, N, 3) dense completion C0
        crop_ratio: fraction in [0.2, 0.4] recommended for finetune; fraction of N to remove
        centroid: optional (B, 3)
    """
    if complete.dim() != 3:
        raise ValueError(f'complete must be (B,N,3), got {tuple(complete.shape)}')
    b, n, _ = complete.shape
    ratio = float(max(0.0, min(0.99, crop_ratio)))
    num_crop = max(1, int(n * ratio))
    keep_k = max(1, n - num_crop)

    if centroid is None:
        centroid = random_unit_centroid(b, complete.device, seed=seed)
    else:
        centroid = normalize_centroid(centroid.to(complete.device))

    dist2 = ((complete - centroid.unsqueeze(1)) ** 2).sum(dim=-1)
    _, far_idx = torch.topk(dist2, k=keep_k, dim=1, largest=True, sorted=False)

    out = []
    for bi in range(b):
        out.append(complete[bi].index_select(0, far_idx[bi]))
    return torch.stack(out, dim=0)


def complete_to_partial_input(
    complete: torch.Tensor,
    n_input: int,
    crop_ratio: float,
    seed: int | None = None,
    gt: torch.Tensor | None = None,
    config=None,
) -> torch.Tensor:
    """C0 -> crop -> FPS to n_input (2048 for PCN).

    When config.feedback_training.crop_mode is error_aware, gt must be provided.
    """
    centroid = None
    if config is not None and resolve_crop_mode(config) != CROP_MODE_RANDOM:
        centroid = resolve_crop_centroid(complete, gt, config, seed=seed)
    cropped = crop_partial_from_complete(complete, crop_ratio, centroid=centroid, seed=seed)
    return misc.fps(cropped, n_input)
