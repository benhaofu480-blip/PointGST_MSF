#!/usr/bin/env python3
"""
创建PCN-Core子集（5000个训练样本）
策略：分层抽样，重点类别高比例采样
"""
import json
import random
import os

# 设置随机种子保证可复现
random.seed(42)

# 读取PCN数据集元数据
with open('data/PCN/PCN.json', 'r') as f:
    pcn_data = json.load(f)

# 类别映射
category_map = {
    '04256520': 'sofa',
    '03001627': 'chair',
    '02958343': 'car',
    '04530566': 'watercraft',
    '04379243': 'table',
    '02691156': 'airplane',
    '02933112': 'cabinet',
    '03636649': 'lamp'
}

# PCN-Core子集分配方案（5000样本）
# 策略：问题类别(Cabinet/Lamp/Table)高比例，简单类别低比例
allocation = {
    'cabinet': 1500,    # 主攻方向（虽然训练集只有1322，全取）
    'lamp': 800,        # H1冗余问题
    'table': 800,       # W2高但H1反降
    'sofa': 600,        # 密度失衡
    'chair': 500,       # 验证拓扑
    'airplane': 400,    # 基础保障
    'car': 200,         # 简单样本
    'watercraft': 200   # 简单样本
}

print("="*70)
print("构建PCN-Core子集")
print("="*70)

# 构建子集
pcn_core = []
for cat_data in pcn_data:
    tax_id = cat_data['taxonomy_id']
    cat_name = category_map.get(tax_id, tax_id)
    
    if cat_name not in allocation:
        continue
    
    target = allocation[cat_name]
    train_ids = cat_data.get('train', [])
    
    # 如果目标数大于可用数，全取；否则随机采样
    if target >= len(train_ids):
        selected = train_ids
        print(f"{cat_name}: 全取 {len(selected)}/{len(train_ids)} 个样本")
    else:
        selected = random.sample(train_ids, target)
        print(f"{cat_name}: 随机采样 {target}/{len(train_ids)} 个样本")
    
    for sample_id in selected:
        pcn_core.append({
            'taxonomy_id': tax_id,
            'taxonomy_name': cat_name,
            'sample_id': sample_id
        })

print("="*70)
print(f"PCN-Core子集总计: {len(pcn_core)} 个样本")
print("="*70)

# 保存子集列表
os.makedirs('data/PCN_Core', exist_ok=True)

# 保存JSON格式
with open('data/PCN_Core/pcn_core_train.json', 'w') as f:
    json.dump(pcn_core, f, indent=2)

# 保存txt格式（每行一个sample_id，方便DataLoader读取）
with open('data/PCN_Core/pcn_core_train.txt', 'w') as f:
    for item in pcn_core:
        f.write(f"{item['taxonomy_id']}/{item['sample_id']}\n")

# 保存类别统计
stats = {}
for item in pcn_core:
    cat = item['taxonomy_name']
    stats[cat] = stats.get(cat, 0) + 1

print("\n子集类别分布:")
for cat, count in sorted(stats.items()):
    pct = count / len(pcn_core) * 100
    print(f"  {cat}: {count} ({pct:.1f}%)")

print("\n文件保存位置:")
print("  - data/PCN_Core/pcn_core_train.json (详细元数据)")
print("  - data/PCN_Core/pcn_core_train.txt (sample_id列表)")
