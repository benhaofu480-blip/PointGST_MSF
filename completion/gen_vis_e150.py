#!/usr/bin/env python
"""生成第150轮可视化结果"""
import sys
sys.path.insert(0, '/home/fubenhao/data/fubenhao_data/PointGST-main_pure/completion')

import torch
import numpy as np
from utils.config import cfg_from_yaml_file
from tools import builder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    config = cfg_from_yaml_file('cfgs/PCN_models/AdaPoinTr_pgst_baseline_gpu0.yaml')
    model = builder.build_model_from_cfg(config.model)
    model.cuda()
    model.eval()
    
    # 加载最佳权重
    builder.load_model(model, 'experiments/AdaPoinTr_pgst_baseline_gpu0/PCN_models/exp13_repulsion/ckpt-best.pth')
    print("Loaded best checkpoint (epoch 150)")
    
    # 读取输入
    partial_cab = np.loadtxt('visualize_output/cabinet_0_partial.xyz')
    partial_sofa = np.loadtxt('visualize_output/sofa_0_partial.xyz')
    gt_cab = np.loadtxt('visualize_output/cabinet_0_gt.xyz')
    gt_sofa = np.loadtxt('visualize_output/sofa_0_gt.xyz')
    
    out_dir = 'visualize_output/exp13_e150'
    import os
    os.makedirs(out_dir, exist_ok=True)
    
    def predict(partial_np):
        p = torch.from_numpy(partial_np).float().unsqueeze(0).cuda()
        with torch.no_grad():
            ret = model(p)
            coarse = ret[0][0].cpu().numpy()
            fine = ret[1][0].cpu().numpy()
        return coarse, fine
    
    # 预测并保存
    coarse_cab, fine_cab = predict(partial_cab)
    coarse_sofa, fine_sofa = predict(partial_sofa)
    
    np.savetxt(f'{out_dir}/cabinet_coarse.xyz', coarse_cab)
    np.savetxt(f'{out_dir}/cabinet_fine.xyz', fine_cab)
    np.savetxt(f'{out_dir}/sofa_coarse.xyz', coarse_sofa)
    np.savetxt(f'{out_dir}/sofa_fine.xyz', fine_sofa)
    
    # 生成对比图
    def compare_vis(gt, pred, out_file, title):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].scatter(gt[:, 0], gt[:, 1], s=0.5, c='blue')
        axes[0, 0].set_title('GT Top')
        axes[0, 1].scatter(gt[:, 0], gt[:, 2], s=0.5, c='blue')
        axes[0, 1].set_title('GT Side XZ')
        axes[0, 2].scatter(gt[:, 1], gt[:, 2], s=0.5, c='blue')
        axes[0, 2].set_title('GT Side YZ')
        
        axes[1, 0].scatter(pred[:, 0], pred[:, 1], s=0.5, c='red')
        axes[1, 0].set_title('Pred Top')
        axes[1, 1].scatter(pred[:, 0], pred[:, 2], s=0.5, c='red')
        axes[1, 1].set_title('Pred Side XZ')
        axes[1, 2].scatter(pred[:, 1], pred[:, 2], s=0.5, c='red')
        axes[1, 2].set_title('Pred Side YZ')
        
        plt.suptitle(f'{title} - Blue=GT, Red=Pred')
        plt.tight_layout()
        plt.savefig(out_file, dpi=150)
        plt.close()
        print(f"Saved {out_file}")
    
    compare_vis(gt_cab, fine_cab, f'{out_dir}/compare_cabinet.png', 'Cabinet Epoch 150')
    compare_vis(gt_sofa, fine_sofa, f'{out_dir}/compare_sofa.png', 'Sofa Epoch 150')
    
    print(f"\nAll files saved to {out_dir}")

if __name__ == '__main__':
    main()
