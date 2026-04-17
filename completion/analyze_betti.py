#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Betti数定量分析脚本 v4 - 持续同调分析 (高速版)

核心优化:
  - 用gudhi.hera(C++后端)替代scipy匈牙利算法, Wasserstein计算从秒级降到毫秒级
  - FPS下采样降到256点, Alpha Complex构建更快
  - 去掉不必要的持久图可视化(持久图图像), 只保留定量指标

用法:
    conda activate pgst
    cd /home/fubenhao/data/fubenhao_data/PointGST-main_pure/completion
    python analyze_betti.py --max_samples 100
    python analyze_betti.py              # 全量1200样本
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from tools import builder
from utils.config import cfg_from_yaml_file
from datasets.build import build_dataset_from_cfg
from easydict import EasyDict
from torch.utils.data import DataLoader

import gudhi
from gudhi.hera import wasserstein_distance as hera_wasserstein

# GPU加速FPS
from pointnet2_ops import pointnet2_utils


# ============================================================================
# 快速下采样
# ============================================================================
def fps_gpu(points_np, n_points):
    """GPU加速FPS下采样"""
    pts = torch.from_numpy(points_np).float().unsqueeze(0).cuda()
    fps_idx = pointnet2_utils.furthest_point_sample(pts, n_points)
    fps_pts = pointnet2_utils.gather_operation(
        pts.transpose(1, 2).contiguous(), fps_idx
    ).transpose(1, 2).contiguous()
    return fps_pts[0].cpu().numpy()


# ============================================================================
# 持续同调计算 (gudhi C++后端)
# ============================================================================
def compute_persistence_and_pd(points):
    """
    计算Alpha Complex持续同调, 返回各维度的持久图(numpy数组)
    
    返回: dict with keys 'pd0', 'pd1' (each np.ndarray shape (n,2))
    """
    alpha = gudhi.AlphaComplex(points=points)
    tree = alpha.create_simplex_tree()
    persistence = tree.persistence()

    pd0, pd1 = [], []
    for dim, (b, d) in persistence:
        b, d = float(b), float(d)
        if b == d:
            continue
        if dim == 0:
            pd0.append([b, np.inf if not np.isfinite(d) else d])
        elif dim == 1:
            pd1.append([b, np.inf if not np.isfinite(d) else d])

    return {
        'pd0': np.array(pd0) if pd0 else np.empty((0, 2)),
        'pd1': np.array(pd1) if pd1 else np.empty((0, 2)),
        'betti_inf': tree.persistent_betti_numbers(0, float('inf')),
    }


def compute_wasserstein(pd1, pd2, p=2):
    """
    用gudhi.hera计算Wasserstein距离 (C++实现, 毫秒级)
    自动处理对角线投影和inf值
    """
    if len(pd1) == 0 and len(pd2) == 0:
        return 0.0
    # hera_wasserstein接受numpy array, 自动处理inf和对角线
    # 返回的是 (distance)^p 的开p次方
    try:
        return hera_wasserstein(pd1, pd2, order=p)
    except Exception:
        return float('inf')


def compute_bottleneck(pd1, pd2):
    """用gudhi计算Bottleneck距离"""
    if len(pd1) == 0 and len(pd2) == 0:
        return 0.0
    return gudhi.bottleneck_distance(pd1, pd2)


def betti_curve_distance(persistence, dim, n_bins=100):
    """
    计算Betti曲线, 返回一个标量特征(曲线下面积和峰值)
    用于快速比较
    """
    dim_pers = [(float(b), float(d)) for dd, (b, d) in persistence if dd == dim]
    if not dim_pers:
        return 0.0, 0, 0.0

    # 收集所有有限death
    deaths_finite = [d for _, d in dim_pers if np.isfinite(d)]
    max_filt = max(deaths_finite) * 1.1 if deaths_finite else 1.0

    # 构建Betti曲线
    events = []
    for b, d in dim_pers:
        events.append((b, +1))
        if np.isfinite(d):
            events.append((d, -1))
    events.sort()

    filts = np.linspace(0, max_filt, n_bins)
    betti = np.zeros(n_bins, dtype=float)
    current = 0
    event_idx = 0
    for i, f in enumerate(filts):
        while event_idx < len(events) and events[event_idx][0] <= f:
            current += events[event_idx][1]
            event_idx += 1
        betti[i] = max(current, 0)

    area = np.trapz(betti, filts)
    peak = int(np.max(betti))
    return area, peak, float(np.max(betti))


# ============================================================================
# 主分析流程
# ============================================================================
TOPO_SAMPLE_SIZE = 256  # 降到256, Alpha Complex更快


def load_model(config, checkpoint_path):
    model = builder.model_builder(config.model)
    builder.load_model(model, checkpoint_path)
    model.cuda()
    model.eval()
    if not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model).cuda()
    return model


def build_test_loader(config):
    test_cfg = EasyDict()
    if '_base_' in config.dataset.test:
        for key, val in config.dataset.test['_base_'].items():
            test_cfg[key] = val
    if 'others' in config.dataset.test:
        for key, val in config.dataset.test['others'].items():
            test_cfg[key] = val
    test_cfg['NAME'] = test_cfg.get('NAME', 'PCN')
    test_cfg['subset'] = test_cfg.get('subset', 'test')
    dataset = build_dataset_from_cfg(test_cfg)
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


def analyze(model, test_loader, max_samples=None):
    """持续同调分析"""
    synset_to_name = {
        '04256520': 'sofa', '03001627': 'chair', '02958343': 'car',
        '04530566': 'watercraft', '04379243': 'table', '02691156': 'airplane',
        '02933112': 'cabinet', '03636649': 'lamp'
    }

    results = []
    t0 = time.time()

    from extensions.chamfer_dist import ChamferDistanceL1
    chamfer_fn = ChamferDistanceL1()

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_loader):
            if max_samples and idx >= max_samples:
                break

            tid = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            mid = model_ids[0]
            cat = synset_to_name.get(str(tid), str(tid))

            partial = data[0].cuda()
            gt = data[1].cuda()

            ret = model(partial)
            fine_pred = ret[-1]
            cd_l1 = chamfer_fn(fine_pred, gt).item() * 1000

            gt_np = gt[0].cpu().numpy()
            pred_np = fine_pred[0].cpu().numpy()

            # FPS下采样
            gt_sub = fps_gpu(gt_np, TOPO_SAMPLE_SIZE)
            pred_sub = fps_gpu(pred_np, TOPO_SAMPLE_SIZE)

            # 持续同调 (gudhi C++)
            gt_topo = compute_persistence_and_pd(gt_sub)
            pred_topo = compute_persistence_and_pd(pred_sub)

            # Wasserstein距离 (gudhi.hera C++)
            w2_h0 = compute_wasserstein(pred_topo['pd0'], gt_topo['pd0'], p=2)
            w2_h1 = compute_wasserstein(pred_topo['pd1'], gt_topo['pd1'], p=2)
            w1_h0 = compute_wasserstein(pred_topo['pd0'], gt_topo['pd0'], p=1)

            # Bottleneck距离
            bd_h0 = compute_bottleneck(pred_topo['pd0'], gt_topo['pd0'])
            bd_h1 = compute_bottleneck(pred_topo['pd1'], gt_topo['pd1'])

            # Betti数
            gt_betti = gt_topo['betti_inf']
            pred_betti = pred_topo['betti_inf']
            gt_b0 = gt_betti[0] if len(gt_betti) > 0 else 0
            gt_b1 = gt_betti[1] if len(gt_betti) > 1 else 0
            pred_b0 = pred_betti[0] if len(pred_betti) > 0 else 0
            pred_b1 = pred_betti[1] if len(pred_betti) > 1 else 0

            # 持久图特征数
            n_gt_h0 = len(gt_topo['pd0'])
            n_pred_h0 = len(pred_topo['pd0'])
            n_gt_h1 = len(gt_topo['pd1'])
            n_pred_h1 = len(pred_topo['pd1'])

            result = {
                'idx': idx, 'category': cat, 'model_id': mid,
                'cd_l1': cd_l1,
                'gt_b0': gt_b0, 'gt_b1': gt_b1,
                'pred_b0': pred_b0, 'pred_b1': pred_b1,
                'w2_h0': w2_h0, 'w2_h1': w2_h1,
                'w1_h0': w1_h0,
                'bd_h0': bd_h0, 'bd_h1': bd_h1,
                'n_gt_h0': n_gt_h0, 'n_pred_h0': n_pred_h0,
                'n_gt_h1': n_gt_h1, 'n_pred_h1': n_pred_h1,
                'topo_error': w2_h0 + w2_h1,
            }
            results.append(result)

            elapsed = time.time() - t0
            speed = (idx + 1) / elapsed
            if (idx + 1) % 20 == 0:
                total = min(len(test_loader), max_samples or len(test_loader))
                eta = (total - idx - 1) / speed if speed > 0 else 0
                print(f"  [{idx+1}/{total}] {cat}: W2(H0)={w2_h0:.4f} W2(H1)={w2_h1:.4f} "
                      f"CD={cd_l1:.2f} | {speed:.1f}samp/s ETA={eta:.0f}s")

    elapsed = time.time() - t0
    print(f"  完成 {len(results)} 样本, 耗时 {elapsed:.1f}s ({len(results)/max(elapsed,1):.1f} samp/s)")
    return results


def print_statistics(results):
    """打印统计"""
    print("\n" + "=" * 90)
    print("持续同调 (Persistent Homology) 定量分析报告")
    print("=" * 90)

    n = len(results)
    w2_h0s = [r['w2_h0'] for r in results]
    w2_h1s = [r['w2_h1'] for r in results]
    w1_h0s = [r['w1_h0'] for r in results]
    bd_h0s = [r['bd_h0'] for r in results]
    bd_h1s = [r['bd_h1'] for r in results]
    topo_errors = [r['topo_error'] for r in results]
    cd_l1s = [r['cd_l1'] for r in results]

    # 过滤inf值用于统计
    def safe_stats(vals, name):
        finite = [v for v in vals if np.isfinite(v)]
        if not finite:
            return
        arr = np.array(finite)
        print(f"  {name}: mean={np.mean(arr):.6f}  std={np.std(arr):.6f}  "
              f"median={np.median(arr):.6f}  range=[{np.min(arr):.6f}, {np.max(arr):.6f}]")

    print(f"\n总样本数: {n}  |  下采样: {TOPO_SAMPLE_SIZE}点 (GPU FPS)")
    print(f"拓扑计算: gudhi Alpha Complex + hera Wasserstein (C++后端)")

    print(f"\n{'─'*70}")
    print(f"  Wasserstein 距离 (越小=拓扑越接近GT)")
    print(f"{'─'*70}")
    safe_stats(w2_h0s, "W2(H0)")
    safe_stats(w2_h1s, "W2(H1)")
    safe_stats(topo_errors, "W2(H0+H1)")
    safe_stats(w1_h0s, "W1(H0)")

    print(f"\n{'─'*70}")
    print(f"  Bottleneck 距离")
    print(f"{'─'*70}")
    safe_stats(bd_h0s, "BD(H0)")
    safe_stats(bd_h1s, "BD(H1)")

    print(f"\n{'─'*70}")
    print(f"  持久图特征统计")
    print(f"{'─'*70}")
    print(f"  H0特征数: GT mean={np.mean([r['n_gt_h0'] for r in results]):.1f}  "
          f"Pred mean={np.mean([r['n_pred_h0'] for r in results]):.1f}")
    print(f"  H1特征数: GT mean={np.mean([r['n_gt_h1'] for r in results]):.1f}  "
          f"Pred mean={np.mean([r['n_pred_h1'] for r in results]):.1f}")

    # 按类别
    print(f"\n{'─'*70}")
    print(f"  按类别统计")
    print(f"{'─'*70}")
    print(f"{'Category':<12} {'#':>4} {'W2(H0)':>9} {'W2(H1)':>9} {'W2(sum)':>9} "
          f"{'BD(H0)':>9} {'CDL1':>7}")
    print("-" * 78)

    categories = sorted(set(r['category'] for r in results))
    for cat in categories:
        cr = [r for r in results if r['category'] == cat]
        cn = len(cr)
        def safe_mean(key):
            vals = [r[key] for r in cr if np.isfinite(r[key])]
            return np.mean(vals) if vals else float('nan')
        print(f"{cat:<12} {cn:>4} "
              f"{safe_mean('w2_h0'):>9.5f} "
              f"{safe_mean('w2_h1'):>9.5f} "
              f"{safe_mean('topo_error'):>9.5f} "
              f"{safe_mean('bd_h0'):>9.5f} "
              f"{np.mean([r['cd_l1'] for r in cr]):>7.2f}")

    # 相关性
    if n > 10:
        from scipy.stats import pearsonr, spearmanr
        print(f"\n{'─'*70}")
        print(f"  拓扑误差 vs 几何误差 (CD-L1) 相关性")
        print(f"{'─'*70}")
        for metric_name, metric_vals in [
            ('W2(H0)', w2_h0s), ('W2(H1)', w2_h1s), ('W2(sum)', topo_errors),
            ('BD(H0)', bd_h0s), ('BD(H1)', bd_h1s),
        ]:
            # 过滤inf
            pairs = [(m, c) for m, c in zip(metric_vals, cd_l1s) if np.isfinite(m)]
            if len(pairs) < 5:
                print(f"  {metric_name}: 样本不足")
                continue
            m_arr, c_arr = zip(*pairs)
            if np.std(m_arr) < 1e-12 or np.std(c_arr) < 1e-12:
                print(f"  {metric_name}: 常数, 无法计算相关")
                continue
            r_p, p_p = pearsonr(m_arr, c_arr)
            r_s, p_s = spearmanr(m_arr, c_arr)
            sig = "***" if p_p < 0.001 else "**" if p_p < 0.01 else "*" if p_p < 0.05 else "ns"
            print(f"  {metric_name:>8} vs CD-L1: Pearson r={r_p:+.4f} p={p_p:.4f} {sig}  "
                  f"| Spearman r={r_s:+.4f} p={p_s:.4f}")

        # 关键结论
        pairs = [(m, c) for m, c in zip(topo_errors, cd_l1s) if np.isfinite(m)]
        if len(pairs) >= 10:
            m_arr, c_arr = zip(*pairs)
            if np.std(m_arr) > 1e-12 and np.std(c_arr) > 1e-12:
                r_w2, p_w2 = pearsonr(m_arr, c_arr)
                if p_w2 > 0.05:
                    print(f"\n  >>> 关键发现: W2拓扑误差与CD-L1无显著相关 (p={p_w2:.4f}>0.05)")
                    print(f"  >>> 即使CD指标很好, 拓扑结构也可能有显著差异!")
                    print(f"  >>> 这表明现有损失函数无法充分约束拓扑结构, 引入拓扑损失有必要性。")
                else:
                    print(f"\n  >>> W2拓扑误差与CD-L1显著相关 (p={p_w2:.4f}), r={r_w2:.4f}")
                    print(f"  >>> 说明几何重建好的样本拓扑保持也较好, 但拓扑仍可能提供额外信息。")

    print("\n" + "=" * 90)
    return {'n': n, 'mean_w2': np.mean([v for v in topo_errors if np.isfinite(v)])}


def save_results(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'betti_analysis_results.csv')
    headers = ['idx', 'category', 'cd_l1', 'gt_b0', 'pred_b0', 'gt_b1', 'pred_b1',
               'w2_h0', 'w2_h1', 'w1_h0', 'bd_h0', 'bd_h1',
               'n_gt_h0', 'n_pred_h0', 'n_gt_h1', 'n_pred_h1', 'topo_error']
    with open(csv_path, 'w') as f:
        f.write(','.join(headers) + '\n')
        for r in results:
            f.write(','.join(str(r[h]) for h in headers) + '\n')
    print(f"CSV: {csv_path}")


def generate_plots(results, output_dir):
    """生成可视化"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    w2_h0s = np.array([r['w2_h0'] for r in results])
    w2_h1s = np.array([r['w2_h1'] for r in results])
    topo_errors = np.array([r['topo_error'] for r in results])
    cd_l1s = np.array([r['cd_l1'] for r in results])
    bd_h0s = np.array([r['bd_h0'] for r in results])
    bd_h1s = np.array([r['bd_h1'] for r in results])
    categories = sorted(set(r['category'] for r in results))

    # 过滤有限值
    finite_mask = np.isfinite(topo_errors) & np.isfinite(cd_l1s)

    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    fig.suptitle('Persistent Homology Analysis: Pred vs GT Topology\n'
                 f'(n={len(results)}, gudhi Alpha Complex + hera Wasserstein, '
                 f'FPS {TOPO_SAMPLE_SIZE}pts)',
                 fontsize=14, fontweight='bold')

    # 1. W2(H0)分布
    ax = axes[0, 0]
    finite_h0 = w2_h0s[np.isfinite(w2_h0s)]
    if len(finite_h0) > 0:
        ax.hist(finite_h0, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
        ax.axvline(x=np.mean(finite_h0), color='red', linestyle='--',
                   label=f'mean={np.mean(finite_h0):.5f}')
    ax.set_xlabel(r'$W_2$ distance (H0)')
    ax.set_ylabel('Count')
    ax.set_title(r'$W_2$ Distance: $PD_0$(Pred) vs $PD_0$(GT)')
    ax.legend()

    # 2. W2(H1)分布
    ax = axes[0, 1]
    finite_h1 = w2_h1s[np.isfinite(w2_h1s)]
    if len(finite_h1) > 0:
        ax.hist(finite_h1, bins=30, color='coral', edgecolor='white', alpha=0.8)
        ax.axvline(x=np.mean(finite_h1), color='red', linestyle='--',
                   label=f'mean={np.mean(finite_h1):.5f}')
    ax.set_xlabel(r'$W_2$ distance (H1)')
    ax.set_ylabel('Count')
    ax.set_title(r'$W_2$ Distance: $PD_1$(Pred) vs $PD_1$(GT)')
    ax.legend()

    # 3. 拓扑误差 vs CD
    ax = axes[0, 2]
    mask = finite_mask & np.isfinite(topo_errors)
    if np.sum(mask) > 2:
        sc = ax.scatter(cd_l1s[mask], topo_errors[mask], c=topo_errors[mask], cmap='hot_r',
                        alpha=0.6, s=25, edgecolors='gray', linewidths=0.3)
        if np.std(cd_l1s[mask]) > 0 and np.std(topo_errors[mask]) > 0:
            z = np.polyfit(cd_l1s[mask], topo_errors[mask], 1)
            p_line = np.poly1d(z)
            x_line = np.linspace(np.min(cd_l1s[mask]), np.max(cd_l1s[mask]), 100)
            ax.plot(x_line, p_line(x_line), 'r--', alpha=0.5, label='linear fit')
            from scipy.stats import pearsonr
            r, p = pearsonr(cd_l1s[mask], topo_errors[mask])
            ax.text(0.05, 0.95, f'r={r:.3f}\np={p:.4f}', transform=ax.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.colorbar(sc, ax=ax, label='Topo Error')
    ax.set_xlabel('CD-L1 (geometric error)')
    ax.set_ylabel(r'$W_2(H0) + W_2(H1)$')
    ax.set_title('Topology Error vs Geometric Error')
    ax.legend()

    # 4. 按类别W2距离
    ax = axes[1, 0]
    cat_means_w2h0, cat_means_w2h1 = [], []
    for cat in categories:
        cr = [r for r in results if r['category'] == cat]
        cat_means_w2h0.append(np.nanmean([r['w2_h0'] for r in cr]))
        cat_means_w2h1.append(np.nanmean([r['w2_h1'] for r in cr]))
    x = np.arange(len(categories))
    w = 0.35
    ax.bar(x - w/2, cat_means_w2h0, w, label=r'$W_2(H0)$', color='steelblue', alpha=0.8)
    ax.bar(x + w/2, cat_means_w2h1, w, label=r'$W_2(H1)$', color='coral', alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_ylabel(r'$W_2$ distance')
    ax.set_title('Per-Category Topology Error')
    ax.legend()

    # 5. Bottleneck距离分布
    ax = axes[1, 1]
    finite_bd0 = bd_h0s[np.isfinite(bd_h0s)]
    finite_bd1 = bd_h1s[np.isfinite(bd_h1s)]
    if len(finite_bd0) > 0:
        ax.hist(finite_bd0, bins=30, color='steelblue', alpha=0.6, label='H0', edgecolor='white')
    if len(finite_bd1) > 0:
        ax.hist(finite_bd1, bins=30, color='coral', alpha=0.6, label='H1', edgecolor='white')
    ax.set_xlabel('Bottleneck Distance')
    ax.set_ylabel('Count')
    ax.set_title('Bottleneck Distance Distribution')
    ax.legend()

    # 6. H0特征数对比
    ax = axes[1, 2]
    gt_h0_counts = [r['n_gt_h0'] for r in results]
    pred_h0_counts = [r['n_pred_h0'] for r in results]
    max_count = max(max(gt_h0_counts), max(pred_h0_counts))
    bins = np.arange(-0.5, max_count + 1.5, 1)
    ax.hist(gt_h0_counts, bins=bins, alpha=0.6, color='dodgerblue', label='GT', edgecolor='white')
    ax.hist(pred_h0_counts, bins=bins, alpha=0.6, color='tomato', label='Pred', edgecolor='white')
    ax.set_xlabel('Number of H0 Features')
    ax.set_ylabel('Count')
    ax.set_title('H0 Persistence Feature Count: GT vs Pred')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'betti_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"主图: {output_dir}/betti_analysis.png")

    # 第二张图: 按类别详细
    fig2, axes2 = plt.subplots(1, 3, figsize=(20, 6))
    fig2.suptitle('Detailed Category Analysis', fontsize=14, fontweight='bold')

    colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))

    ax = axes2[0]
    cat_data = [[r['w2_h0'] for r in results if r['category'] == cat and np.isfinite(r['w2_h0'])]
                for cat in categories]
    bp = ax.boxplot(cat_data, labels=categories, patch_artist=True)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
    ax.set_ylabel(r'$W_2(H0)$'); ax.set_title('W2(H0) by Category')
    ax.tick_params(axis='x', rotation=45)

    ax = axes2[1]
    cat_data = [[r['w2_h1'] for r in results if r['category'] == cat and np.isfinite(r['w2_h1'])]
                for cat in categories]
    bp = ax.boxplot(cat_data, labels=categories, patch_artist=True)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
    ax.set_ylabel(r'$W_2(H1)$'); ax.set_title('W2(H1) by Category')
    ax.tick_params(axis='x', rotation=45)

    ax = axes2[2]
    for ci, cat in enumerate(categories):
        cr = [r for r in results if r['category'] == cat and np.isfinite(r['topo_error'])]
        if cr:
            ax.scatter([r['cd_l1'] for r in cr], [r['topo_error'] for r in cr],
                       c=[plt.cm.Set3(ci)], label=cat, alpha=0.6, s=20)
    ax.set_xlabel('CD-L1'); ax.set_ylabel(r'$W_2(H0+H1)$')
    ax.set_title('CD-L1 vs Topology Error by Category')
    ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'betti_analysis_detail.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"详图: {output_dir}/betti_analysis_detail.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='cfgs/PCN_models/AdaPoinTr_pgst.yaml')
    parser.add_argument('--checkpoint', type=str,
                        default='experiments/AdaPoinTr_pgst_baseline_gpu0/PCN_models/exp13_repulsion/ckpt-best.pth')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--sample_size', type=int, default=256)
    parser.add_argument('--output_dir', type=str, default='./betti_analysis_output')
    parser.add_argument('--no_plots', action='store_true')
    args = parser.parse_args()

    global TOPO_SAMPLE_SIZE
    TOPO_SAMPLE_SIZE = args.sample_size

    print("=" * 60)
    print("持续同调分析 - 点云补全拓扑保持评估 v4 (hera加速)")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"FPS采样: {TOPO_SAMPLE_SIZE}点 (GPU)")
    print(f"Max samples: {args.max_samples or 'All'}")

    print("[1/4] 加载模型...")
    config = cfg_from_yaml_file(args.config)
    config.model.NAME = 'AdaPoinTr'
    model = load_model(config, args.checkpoint)

    print("[2/4] 加载数据集...")
    test_loader = build_test_loader(config)
    print(f"  测试集: {len(test_loader)} 样本")

    print("[3/4] 计算持续同调 (gudhi.hera C++后端)...")
    results = analyze(model, test_loader, max_samples=args.max_samples)

    print("[4/4] 生成报告...")
    print_statistics(results)
    save_results(results, args.output_dir)
    if not args.no_plots:
        generate_plots(results, args.output_dir)

    print(f"\n完成! 结果在: {args.output_dir}/")


if __name__ == '__main__':
    main()
