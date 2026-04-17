# PCN-Core 子集使用说明

## 1. 子集概述

PCN-Core是一个精选的PCN训练子集，用于**快速迭代实验**。相比全量训练集（28,974个样本），PCN-Core只包含**4,822个样本**（约16%），但能覆盖主要的问题类别。

## 2. 子集构成

| 类别 | 样本数 | 占比 | 选择策略 |
|------|--------|------|----------|
| cabinet | 1,322 | 27.4% | **全取**（训练集只有这些） |
| lamp | 800 | 16.6% | 随机采样 |
| table | 800 | 16.6% | 随机采样 |
| sofa | 600 | 12.4% | 随机采样 |
| chair | 500 | 10.4% | 随机采样 |
| airplane | 400 | 8.3% | 随机采样 |
| car | 200 | 4.1% | 随机采样 |
| watercraft | 200 | 4.1% | 随机采样 |
| **总计** | **4,822** | **100%** | |

## 3. 为什么这样设计？

基于TDA分析结果：
- **Cabinet**: W2_H1最高（0.00443），是主要问题类别，全取
- **Lamp**: H1冗余严重（+109个假环），高比例采样
- **Table**: W2高但H1反降（分布问题），高比例采样
- **Sofa/Chair**: 中等复杂度，中等比例
- **Airplane/Car/Watercraft**: 简单类别，低比例（基础保障）

## 4. 使用方法

### 4.1 使用PCN-Core配置

```bash
python main.py \
    --config cfgs/PCN_models/AdaPoinTr_pgst_mssa_core.yaml \
    --exp_name exp15_pcn_core_test
```

### 4.2 在原配置上使用子集

在任意配置文件中，修改train数据集配置：

```yaml
dataset : {
  train : { _base_: cfgs/dataset_configs/PCN.yaml, 
            others: {subset: 'train', 
                     sample_list_file: 'data/PCN_Core/pcn_core_train.txt'}},
  ...
}
```

### 4.3 创建新的子集

修改`create_pcn_core.py`中的`allocation`字典，然后运行：

```bash
python create_pcn_core.py
```

## 5. 训练时间对比

| 数据集 | 样本数 | 每轮时间 | 80轮总时间 |
|--------|--------|----------|------------|
| PCN全量 | 28,974 | ~6分钟 | ~8小时 |
| PCN-Core | 4,822 | ~1分钟 | ~1.3小时 |

**加速比：约6倍**

## 6. 实验策略建议

### 阶段1：快速筛选（Quick Prototyping）
- 使用PCN-Core
- 只跑60-80轮
- 观察Cabinet/Lamp/Table的CD和可视化
- **标准**：60轮CD<7，Cabinet可视化无严重问题

### 阶段2：中场复盘（Robustness Check）
- 通过筛选的方案，跑120轮
- 与Baseline（exp4）在Core上的分数对比
- **标准**：优于Baseline

### 阶段3：全量验证（Full Training）
- 通过中场复盘的方案，在全量PCN上跑200轮
- 验证最终指标和泛化能力

## 7. 注意事项

1. **验证集保持不变**：仍使用800个验证样本，不缩减
2. **测试集保持不变**：最终评估仍使用1200个测试样本
3. **缓存机制**：第一次使用PCN-Core会重建磁盘缓存，稍慢
4. **可复现性**：子集使用固定随机种子（seed=42），确保可复现

## 8. 文件位置

- 子集列表：`data/PCN_Core/pcn_core_train.txt`
- 子集元数据：`data/PCN_Core/pcn_core_train.json`
- 配置文件：`cfgs/PCN_models/AdaPoinTr_pgst_mssa_core.yaml`
- 构建脚本：`create_pcn_core.py`
- 分析脚本：`analyze_subset.py`

## 9. 下一步行动

现在可以开始**"局部特征诱导Query"**方案的实验：

```bash
# 1. 修改模型代码（AdaPoinTr.py）
# 2. 使用PCN-Core快速验证
# 3. 观察60轮指标，决定是否继续
```
