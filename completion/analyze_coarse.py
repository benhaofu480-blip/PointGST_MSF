"""
定量分析 coarse 预测质量 — 比较 exp4 vs exp5 的 coarse 点云覆盖度
"""
import os
import sys
import torch
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from tools import builder
from utils.config import cfg_from_yaml_file
from datasets.build import build_dataset_from_cfg
from easydict import EasyDict
from torch.utils.data import DataLoader
from extensions.chamfer_dist import ChamferDistanceL1
import misc


def load_model(config, checkpoint):
    config_copy = cfg_from_yaml_file(config)
    config_copy.model.NAME = 'AdaPoinTr_PGST'
    model = builder.model_builder(config_copy.model)
    builder.load_model(model, checkpoint)
    model.cuda()
    model.eval()
    return model


def main():
    config_path = 'cfgs/PCN_models/pertoken_core_gpu1.yaml'
    ckpt1 = 'experiments/pertoken_core_gpu1/PCN_models/exp4_pertoken_core_gpu1/ckpt-best.pth'
    ckpt2 = 'experiments/pertoken_core_gpu1/PCN_models/exp5_pertoken_diff_lr/ckpt-best.pth'

    model1 = load_model(config_path, ckpt1)
    model2 = load_model(config_path, ckpt2)

    config = cfg_from_yaml_file(config_path)
    test_cfg = EasyDict()
    if '_base_' in config.dataset.test:
        for key, val in config.dataset.test['_base_'].items():
            test_cfg[key] = val
    if 'others' in config.dataset.test:
        for key, val in config.dataset.test['others'].items():
            test_cfg[key] = val
    test_cfg['NAME'] = test_cfg.get('NAME', 'PCN')
    test_cfg['subset'] = test_cfg.get('subset', 'test')

    val_dataset = build_dataset_from_cfg(test_cfg)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    chamfer_fn = ChamferDistanceL1()
    
    cd_coarse_exp4_list = []
    cd_coarse_exp5_list = []
    cd_fine_exp4_list = []
    cd_fine_exp5_list = []

    # coarse spread: 计算coarse点的bbox volume
    spread_exp4_list = []
    spread_exp5_list = []

    with torch.no_grad():
        for idx, (taxonomy_id, model_id, data) in enumerate(val_loader):
            partial, gt = data
            partial = partial.cuda()
            gt = gt.cuda()

            ret1 = model1(partial)
            coarse1, fine1 = ret1[0], ret1[1]
            cd_coarse1 = chamfer_fn(coarse1, gt).item() * 1000
            cd_fine1 = chamfer_fn(fine1, gt).item() * 1000

            ret2 = model2(partial)
            coarse2, fine2 = ret2[0], ret2[1]
            cd_coarse2 = chamfer_fn(coarse2, gt).item() * 1000
            cd_fine2 = chamfer_fn(fine2, gt).item() * 1000

            # bbox spread
            def bbox_volume(pts):
                pts_np = pts[0].cpu().numpy()
                return np.prod(pts_np.max(axis=0) - pts_np.min(axis=0))

            gt_vol = bbox_volume(gt)
            sp1 = bbox_volume(coarse1) / gt_vol if gt_vol > 0 else 0
            sp2 = bbox_volume(coarse2) / gt_vol if gt_vol > 0 else 0

            cd_coarse_exp4_list.append(cd_coarse1)
            cd_coarse_exp5_list.append(cd_coarse2)
            cd_fine_exp4_list.append(cd_fine1)
            cd_fine_exp5_list.append(cd_fine2)
            spread_exp4_list.append(sp1)
            spread_exp5_list.append(sp2)

    n = len(cd_coarse_exp4_list)
    print(f"===== 统计 {n} 个测试样本 =====\n")
    print(f"{'指标':<25} {'exp4 (pertoken)':<20} {'exp5 (5x lr)':<20} {'差异':<15}")
    print("-" * 80)

    def stat(lst):
        return np.mean(lst), np.median(lst), np.std(lst)

    m1, md1, s1 = stat(cd_coarse_exp4_list)
    m2, md2, s2 = stat(cd_coarse_exp5_list)
    print(f"{'Coarse CD-L1 mean':<25} {m1:.3f}              {m2:.3f}              {m2-m1:+.3f}")
    print(f"{'Coarse CD-L1 median':<25} {md1:.3f}              {md2:.3f}              {md2-md1:+.3f}")

    m1, md1, s1 = stat(cd_fine_exp4_list)
    m2, md2, s2 = stat(cd_fine_exp5_list)
    print(f"{'Fine CD-L1 mean':<25} {m1:.3f}              {m2:.3f}              {m2-m1:+.3f}")
    print(f"{'Fine CD-L1 median':<25} {md1:.3f}              {md2:.3f}              {md2-md1:+.3f}")

    m1, md1, s1 = stat(spread_exp4_list)
    m2, md2, s2 = stat(spread_exp5_list)
    print(f"{'Coarse/GT bbox ratio':<25} {m1:.3f}              {m2:.3f}              {m2-m1:+.3f}")
    print(f"  (ratio < 1 表示 coarse 没覆盖 GT 的完整范围)")

    # per-sample comparison
    print(f"\n===== 逐类分析 =====\n")
    synset_to_name = {
        '04256520': 'sofa', '03001627': 'chair', '02958343': 'car',
        '04530566': 'watercraft', '04379243': 'table', '02691156': 'airplane',
        '02933112': 'cabinet', '03636649': 'lamp'
    }

    # 重新遍历一次按类别统计
    cat_stats = {}
    val_loader2 = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    with torch.no_grad():
        for idx, (taxonomy_id, model_id, data) in enumerate(val_loader2):
            tax = synset_to_name.get(taxonomy_id[0], taxonomy_id[0])
            if tax not in cat_stats:
                cat_stats[tax] = {'cd_coarse1': [], 'cd_coarse2': [], 'cd_fine1': [], 'cd_fine2': [], 'spread1': [], 'spread2': []}

            partial, gt = data
            partial = partial.cuda()
            gt = gt.cuda()

            ret1 = model1(partial)
            ret2 = model2(partial)

            cat_stats[tax]['cd_coarse1'].append(chamfer_fn(ret1[0], gt).item() * 1000)
            cat_stats[tax]['cd_fine1'].append(chamfer_fn(ret1[1], gt).item() * 1000)
            cat_stats[tax]['cd_coarse2'].append(chamfer_fn(ret2[0], gt).item() * 1000)
            cat_stats[tax]['cd_fine2'].append(chamfer_fn(ret2[1], gt).item() * 1000)

            gt_vol = bbox_volume(gt)
            if gt_vol > 0:
                cat_stats[tax]['spread1'].append(bbox_volume(ret1[0]) / gt_vol)
                cat_stats[tax]['spread2'].append(bbox_volume(ret2[0]) / gt_vol)

    print(f"{'类别':<12} {'exp4 CoarseCD':<15} {'exp5 CoarseCD':<15} {'exp4 FineCD':<15} {'exp5 FineCD':<15} {'exp4 Spread':<12} {'exp5 Spread':<12}")
    print("-" * 98)
    for cat, v in sorted(cat_stats.items()):
        print(f"{cat:<12} {np.mean(v['cd_coarse1']):<15.3f} {np.mean(v['cd_coarse2']):<15.3f} {np.mean(v['cd_fine1']):<15.3f} {np.mean(v['cd_fine2']):<15.3f} {np.mean(v['spread1']):<12.3f} {np.mean(v['spread2']):<12.3f}")


if __name__ == '__main__':
    main()
