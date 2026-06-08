"""
Test-time dense point cloud post-processing.

Preferred backend: Open3D remove_statistical_outlier (library call).
Fallback backend: torch knn_point + index_points (same SOR rule) when Open3D
import hangs or is unavailable on headless nodes — does not modify PGST.py.
"""

from __future__ import annotations

import multiprocessing as mp
import numpy as np
import torch

from utils.misc import fps
from utils.logger import print_log

_BACKEND = None  # None | 'open3d' | 'torch_knn'


def get_test_postprocess_cfg(config):
    if config is None:
        return {}
    block = getattr(config, 'test_postprocess', None)
    if block is None:
        return {}
    if hasattr(block, 'to_dict'):
        return block.to_dict()
    return dict(block)


def is_postprocess_enabled(config) -> bool:
    return bool(get_test_postprocess_cfg(config).get('enabled', False))


def _open3d_import_worker(q):
    try:
        import open3d as o3d  # noqa: F401
        q.put('ok')
    except Exception as e:
        q.put(f'err:{e}')


def _resolve_backend(cfg, logger=None) -> str:
    """Pick open3d or torch_knn once per process."""
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND

    want = str(cfg.get('backend', 'auto')).lower()
    if want == 'torch_knn':
        _BACKEND = 'torch_knn'
    elif want == 'open3d':
        _BACKEND = 'open3d'
    else:
        # auto: try Open3D import in subprocess with timeout
        timeout = float(cfg.get('open3d_import_timeout_sec', 25))
        ctx = mp.get_context('spawn')
        q = ctx.Queue()
        p = ctx.Process(target=_open3d_import_worker, args=(q,))
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            p.join(5)
            _BACKEND = 'torch_knn'
            if logger:
                print_log(
                    f'[postprocess] Open3D import timeout ({timeout}s) -> torch_knn fallback',
                    logger=logger,
                )
        else:
            msg = q.get() if not q.empty() else 'err:empty'
            if msg == 'ok':
                _BACKEND = 'open3d'
                if logger:
                    print_log('[postprocess] backend=open3d', logger=logger)
            else:
                _BACKEND = 'torch_knn'
                if logger:
                    print_log(f'[postprocess] Open3D unavailable ({msg}) -> torch_knn', logger=logger)

    return _BACKEND


def statistical_outlier_filter_open3d(points: np.ndarray, nb_neighbors: int, std_ratio: float):
    import open3d as o3d

    if points.shape[0] < max(nb_neighbors + 1, 3):
        return points

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(points, dtype=np.float64))
    _, inlier_idx = pcd.remove_statistical_outlier(
        nb_neighbors=int(nb_neighbors),
        std_ratio=float(std_ratio),
    )
    if len(inlier_idx) == 0:
        return points
    return np.asarray(pcd.select_by_index(inlier_idx).points, dtype=np.float32)


def statistical_outlier_filter_torch_knn(points: np.ndarray, nb_neighbors: int, std_ratio: float):
    """Same SOR rule as Open3D, using project knn_point (CUDA if tensor on GPU)."""
    from models.Transformer_utils import knn_point, index_points

    n = points.shape[0]
    if n < max(nb_neighbors + 1, 3):
        return points

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pts = torch.from_numpy(np.ascontiguousarray(points, dtype=np.float32)).unsqueeze(0).to(device)
    k = int(nb_neighbors) + 1
    idx = knn_point(k, pts, pts)
    neighbors = index_points(pts, idx)
    dist = (neighbors - pts.unsqueeze(2)).norm(dim=-1)
    mean_dist = dist[:, :, 1:].mean(dim=-1).squeeze(0)
    mu = mean_dist.mean()
    sigma = mean_dist.std().clamp(min=1e-8)
    thresh = mu + float(std_ratio) * sigma
    mask = mean_dist < thresh
    if mask.sum() < 3:
        return points
    return pts.squeeze(0)[mask].detach().cpu().numpy()


def statistical_outlier_filter_numpy(
    points: np.ndarray,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    backend: str = 'torch_knn',
):
    if backend == 'open3d':
        return statistical_outlier_filter_open3d(points, nb_neighbors, std_ratio)
    return statistical_outlier_filter_torch_knn(points, nb_neighbors, std_ratio)


def postprocess_dense_points(
    dense_points: torch.Tensor,
    config,
    logger=None,
) -> torch.Tensor:
    cfg = get_test_postprocess_cfg(config)
    if not cfg.get('enabled', False):
        return dense_points

    method = str(cfg.get('method', 'statistical_outlier'))
    if method != 'statistical_outlier':
        raise ValueError(f'Unknown test_postprocess.method: {method}')

    backend = _resolve_backend(cfg, logger=logger)
    nb_neighbors = int(cfg.get('nb_neighbors', 20))
    std_ratio = float(cfg.get('std_ratio', 2.0))
    min_keep_ratio = float(cfg.get('min_keep_ratio', 0.85))
    resample = bool(cfg.get('resample_to_input', True))

    device = dense_points.device
    dtype = dense_points.dtype
    b, n, _ = dense_points.shape
    out_list = []
    n_skipped = 0

    for bi in range(b):
        pts = dense_points[bi].detach().cpu().numpy()
        n_in = pts.shape[0]
        filtered = statistical_outlier_filter_numpy(
            pts,
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio,
            backend=backend,
        )
        if filtered.shape[0] < max(3, int(n_in * min_keep_ratio)):
            out_list.append(dense_points[bi:bi + 1])
            n_skipped += 1
            continue
        t = torch.from_numpy(filtered).to(device=device, dtype=dtype).unsqueeze(0)
        if resample and t.shape[1] != n:
            t = fps(t, n)
        out_list.append(t)

    return torch.cat(out_list, dim=0)
