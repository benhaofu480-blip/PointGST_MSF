"""
可视化诊断脚本 - 保存 Cabinet/Sofa/Airplane 的预测结果用于分析
"""
import os
import torch
import numpy as np
import argparse
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from tools import builder
from utils.config import cfg_from_yaml_file
from datasets.build import build_dataset_from_cfg
from easydict import EasyDict
from torch.utils.data import DataLoader


def save_point_cloud(points, filename):
    """保存点云为 .xyz 文件，可用 MeshLab 打开"""
    np.savetxt(filename, points, fmt='%.6f', delimiter=' ')
    print(f"Saved: {filename}")


def main(args):
    # 创建输出目录
    os.makedirs('./visualize_output', exist_ok=True)
    
    # 加载配置
    config = cfg_from_yaml_file(args.config)
    config.model.NAME = 'AdaPoinTr'
    
    # 加载模型
    base_model = builder.model_builder(config.model)
    builder.load_model(base_model, args.checkpoint)
    base_model.cuda()
    base_model.eval()
    
    # 加载验证集 - 合并配置
    test_cfg = EasyDict()
    # 合并 _base_ 配置
    if '_base_' in config.dataset.test:
        base_cfg = config.dataset.test['_base_']
        for key, val in base_cfg.items():
            test_cfg[key] = val
    # 合并 others 配置
    if 'others' in config.dataset.test:
        for key, val in config.dataset.test['others'].items():
            test_cfg[key] = val
    # 确保有 NAME 和 subset
    test_cfg['NAME'] = test_cfg.get('NAME', 'PCN')
    test_cfg['subset'] = test_cfg.get('subset', 'test')
    
    val_dataset = build_dataset_from_cfg(test_cfg)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1,  # 一个一个看
        shuffle=False, 
        num_workers=0
    )
    
    # 类别映射
    synset_to_name = {
        '04256520': 'sofa',
        '03001627': 'chair',
        '02958343': 'car',
        '04530566': 'watercraft',
        '04379243': 'table',
        '02691156': 'airplane',
        '02933112': 'cabinet',
        '03636649': 'lamp'
    }
    
    # 收集各类别的样本
    collected = {'cabinet': 0, 'sofa': 0, 'airplane': 0}
    max_samples = args.num_samples  # 每类保存几个
    
    print(f"开始收集样本 (每类 {max_samples} 个)...")
    
    with torch.no_grad():
        for idx, (taxonomy_id, model_id, data) in enumerate(val_loader):
            if all(v >= max_samples for v in collected.values()):
                break
                
            taxonomy_name = synset_to_name.get(taxonomy_id[0], taxonomy_id[0])
            
            # 只收集我们关心的类别
            if taxonomy_name not in collected or collected[taxonomy_name] >= max_samples:
                continue
            
            # 解包数据
            partial, gt = data
            
            # 前向传播
            partial = partial.cuda()
            gt = gt.cuda()
            
            ret = base_model(partial)
            coarse_pred = ret[0]  # (1, 512, 3)
            fine_pred = ret[1]    # (1, 16384, 3)
            
            # 转 numpy
            partial_np = partial[0].cpu().numpy()
            gt_np = gt[0].cpu().numpy()
            coarse_np = coarse_pred[0].cpu().numpy()
            fine_np = fine_pred[0].cpu().numpy()
            
            # 保存
            sample_idx = collected[taxonomy_name]
            prefix = f"./visualize_output/{taxonomy_name}_{sample_idx}"
            
            save_point_cloud(partial_np, f"{prefix}_partial.xyz")
            save_point_cloud(gt_np, f"{prefix}_gt.xyz")
            save_point_cloud(coarse_np, f"{prefix}_coarse.xyz")
            save_point_cloud(fine_np, f"{prefix}_fine.xyz")
            
            # 计算 CD-L1 用于标注
            from extensions.chamfer_dist import ChamferDistanceL1
            chamfer_fn = ChamferDistanceL1()
            cd_l1 = chamfer_fn(fine_pred, gt).item() * 1000  # 转换为 mm
            
            print(f"  [{taxonomy_name}] CD-L1: {cd_l1:.2f}")
            
            collected[taxonomy_name] += 1
    
    print("\n完成！可视化文件保存在 ./visualize_output/")
    print("可用 MeshLab 或 CloudCompare 打开查看：")
    print("  - cabinet_X_partial.xyz: 输入的不完整点云")
    print("  - cabinet_X_gt.xyz: 真实完整点云")
    print("  - cabinet_X_coarse.xyz: 预测的 512 点骨架")
    print("  - cabinet_X_fine.xyz: 预测的 16384 点精细点云")
    print("\n重点关注 cabinet 的 coarse 和 fine 是否有：")
    print("  1. 平面塌陷（本该平整的面坑坑洼洼）")
    print("  2. 飞点（漂浮在物体外的零散点）")
    print("  3. 缺失大片区域")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='cfgs/PCN_models/AdaPoinTr_pgst_baseline_gpu0.yaml')
    parser.add_argument('--checkpoint', type=str,
                        default='./experiments/AdaPoinTr_pgst_baseline_gpu0/PCN_models/exp12_fixed/ckpt-last.pth')
    parser.add_argument('--num_samples', type=int, default=2, help='每类保存几个样本')
    args = parser.parse_args()
    main(args)
