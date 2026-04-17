#!/usr/bin/env python3
"""
分析TDA结果，构建PCN-Core子集
"""
import csv
import json
from collections import defaultdict

# 读取TDA结果
with open('betti_analysis_output/betti_analysis_results.csv', 'r') as f:
    reader = csv.DictReader(f)
    data = list(reader)

# 按类别统计
by_category = defaultdict(list)
for row in data:
    cat = row['category']
    w2_h1 = float(row['w2_h1'])
    idx = int(row['idx'])
    by_category[cat].append({'idx': idx, 'w2_h1': w2_h1, 'cd_l1': float(row['cd_l1'])})

print("="*70)
print("各类别W2_H1分布统计（测试集1200个样本）")
print("="*70)
print(f"{'Category':<12} {'Count':>6} {'Mean':>10} {'Std':>10} {'Max':>10} {'>0.005':>8} {'>0.008':>8} {'>0.01':>8}")
print("-"*70)

stats = {}
for cat in sorted(by_category.keys()):
    values = [x['w2_h1'] for x in by_category[cat]]
    mean = sum(values) / len(values)
    std = (sum((x-mean)**2 for x in values) / len(values)) ** 0.5
    max_val = max(values)
    cnt_005 = sum(1 for x in values if x > 0.005)
    cnt_008 = sum(1 for x in values if x > 0.008)
    cnt_01 = sum(1 for x in values if x > 0.01)
    stats[cat] = {'mean': mean, 'std': std, 'cnt_01': cnt_01}
    print(f"{cat:<12} {len(values):>6} {mean:>10.6f} {std:>10.6f} {max_val:>10.6f} {cnt_005:>8} {cnt_008:>8} {cnt_01:>8}")

print("="*70)

# 根据TDA结果，确定Hard Case阈值
# Cabinet: W2>0.01作为Hard Case
# Lamp: W2>0.008作为Hard Case
# Table: W2>0.008作为Hard Case
# 其他类别: W2>0.005作为Hard Case

print("\n" + "="*70)
print("Hard Case筛选策略（用于构建训练子集）")
print("="*70)

hard_cases = defaultdict(list)
for cat in by_category:
    if cat == 'cabinet':
        threshold = 0.01
    elif cat in ['lamp', 'table']:
        threshold = 0.008
    else:
        threshold = 0.005
    
    hard = [x for x in by_category[cat] if x['w2_h1'] > threshold]
    hard_cases[cat] = hard
    print(f"{cat:<12}: W2_H1 > {threshold:.3f} -> {len(hard)} Hard Cases")

print("\n" + "="*70)
print("建议的PCN-Core子集构成（5000样本）")
print("="*70)

# 构建训练子集分配方案
target_total = 5000
allocation = {
    'cabinet': 1500,   # 主攻方向
    'lamp': 800,       # H1冗余严重
    'table': 800,      # W2高但H1反降（分布问题）
    'sofa': 600,       # 密度失衡
    'chair': 500,      # 验证拓扑
    'airplane': 400,   # 基础保障
    'car': 200,
    'watercraft': 200
}

print(f"{'Category':<12} {'Target':>8} {'Train_Avail':>12} {'Hard_Cases':>12}")
print("-"*50)
total_target = 0
for cat, target in allocation.items():
    train_avail = {'cabinet': 1322, 'lamp': 2068, 'table': 5750, 'sofa': 2923, 
                   'chair': 5750, 'airplane': 3795, 'car': 5677, 'watercraft': 1689}.get(cat, 0)
    hard_cnt = len(hard_cases.get(cat, []))
    print(f"{cat:<12} {target:>8} {train_avail:>12} {hard_cnt:>12}")
    total_target += target

print(f"{'Total':<12} {total_target:>8}")
print("="*70)

# 输出详细建议
print("\n" + "="*70)
print("子集构建策略")
print("="*70)
print("""
1. 从训练集中按以下策略采样：
   - Cabinet: 从Hard Case对应的训练样本中优先选取（如果映射关系可用）
   - Lamp/Table: 同样优先选取拓扑误差大的样本
   - 其他类别: 随机采样补充

2. 由于没有直接的测试集到训练集的Hard Case映射，建议：
   - 在训练集中按类别重新计算TDA（成本高）
   - 或采用简单策略：Cabinet/Lamp/Table类别全量或高比例采样

3. 实际操作建议：
   - 先使用类别级抽样（不区分Hard/Easy）
   - 观察Cabinet/Lamp/Table在子集上的表现
   - 如果仍有差距，再考虑在子集内筛选Hard Case
""")
EOF
