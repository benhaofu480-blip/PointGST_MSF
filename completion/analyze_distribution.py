"""分析点云分布"""
import numpy as np

def analyze_xyz(filepath, name):
    points = np.loadtxt(filepath)
    print(f"\n=== {name} ===")
    print(f"点数: {len(points)}")
    print(f"X 范围: [{points[:, 0].min():.4f}, {points[:, 0].max():.4f}]")
    print(f"Y 范围: [{points[:, 1].min():.4f}, {points[:, 1].max():.4f}]")
    print(f"Z 范围: [{points[:, 2].min():.4f}, {points[:, 2].max():.4f}]")
    
    # 分析点的分布
    x_bins = np.histogram(points[:, 0], bins=5)[0]
    y_bins = np.histogram(points[:, 1], bins=5)[0]
    z_bins = np.histogram(points[:, 2], bins=5)[0]
    print(f"X 分布 (5 bins): {x_bins}")
    print(f"Y 分布 (5 bins): {y_bins}")
    print(f"Z 分布 (5 bins): {z_bins}")
    
    return points

# 分析 Cabinet
print("\n" + "="*60)
print("CABINET 分析")
print("="*60)
gt = analyze_xyz("visualize_output/cabinet_0_gt.xyz", "GT")
pred = analyze_xyz("visualize_output/cabinet_0_fine.xyz", "预测")
partial = analyze_xyz("visualize_output/cabinet_0_partial.xyz", "Partial输入")

# 检查预测是否集中在某个区域
print("\n=== 预测 vs GT 的覆盖差异 ===")
# 计算每个维度上预测覆盖了GT的多少比例
for i, axis in enumerate(['X', 'Y', 'Z']):
    gt_range = gt[:, i].max() - gt[:, i].min()
    pred_range = pred[:, i].max() - pred[:, i].min()
    coverage = pred_range / gt_range * 100
    print(f"{axis}轴: GT范围={gt_range:.4f}, 预测范围={pred_range:.4f}, 覆盖率={coverage:.1f}%")

# Sofa
print("\n" + "="*60)
print("SOFA 分析")
print("="*60)
gt = analyze_xyz("visualize_output/sofa_0_gt.xyz", "GT")
pred = analyze_xyz("visualize_output/sofa_0_fine.xyz", "预测")
partial = analyze_xyz("visualize_output/sofa_0_partial.xyz", "Partial输入")

# Airplane
print("\n" + "="*60)
print("AIRPLANE 分析")
print("="*60)
gt = analyze_xyz("visualize_output/airplane_0_gt.xyz", "GT")
pred = analyze_xyz("visualize_output/airplane_0_fine.xyz", "预测")
partial = analyze_xyz("visualize_output/airplane_0_partial.xyz", "Partial输入")
