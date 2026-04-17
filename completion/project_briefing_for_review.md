# Point Cloud Completion 项目技术简报（供 AI 审阅）

## 一、项目背景

本项目基于 [PointGST](https://github.com/yangengy/PointGST) 框架，核心任务是**单视角点云补全**（partial → complete），数据集为 PCN（ShapeNet 子集，8 类物体）。评价指标：**CDL1**（Chamfer Distance L1，越低越好）。

基线模型是 **PoinTr**（CVPR 2022），一个 encoder-decoder 架构的点云补全模型。我们在其基础上进行改进。

## 二、整体架构

完整流水线：

```
Input: partial point cloud (B, 2048, 3)
  │
  ▼
DGCNN_Grouper:  FPS 下采样 + 图卷积提取局部特征
  │              输出: coor (B, 256, 3), f (B, 256, 128)
  │              FPS 链: xyz(2048) → FPS→512 → FPS→256
  │              所有的 coor 点都来自输入表面的 FPS 采样
  ▼
PosEmbed + InputProj: 位置编码 + 特征投影到 embed_dim(384)
  │
  ▼
Point Transformer Encoder (6层): 自注意力 + 图注意力
  │              输出: token_features (B, 256, 384)
  ▼
increase_dim: Linear(384→1024)
  │              输出: token_features (B, 256, 1024)
  │              global_feature = max_pool(token_features) → (B, 1024)
  ▼
Coarse Generation (本项目的核心改动点，详见第三节)
  │              输出: coarse (B, 512, 3)
  ▼
query_ranking: 对 512 个 coarse 点打分排序（学习到的置信度）
  │
  ▼
mlp_query: [global_feature, coarse] → query features (B, 512, 384)
  │
  ▼
Point Transformer Decoder (8层): self-attn + cross-attn(KNN-indexed)
  │              训练时额外 concat 64 个 denoise jitter 点
  │              输出: q (B, 512, 384)
  ▼
Fold: query features → 16384 个精细点
  │
  ▼
Output: complete point cloud (B, 16384, 3)
```

**训练配置**：
- 优化器: AdamW, base_lr=0.0002, weight_decay=0.0005
- 调参策略: GFT (Grouper-Finetuning)，冻结 backbone grouper + pos_embed + input_proj，只训练 encoder adapter + decoder + pertoken 模块
- pertoken 模块（`coarse_pred`, `global_coarse_pred`, `mlp_query`, `query_ranking`）使用 3x 学习率（0.0006）
- batch_size=32, max_epoch=150, 数据集 PCN_Core（精简版）
- 预训练起点: AdaPoinTr_ps55.pth（361 epoch 预训练权重）
- Checkpoint 加载: shape-filtering（只加载 shape 匹配的权重，不匹配的跳过）

## 三、Coarse Generation 演进历史（核心改动）

### 3.1 原始 PoinTr（基线）

文件: `models/Transformer.py` 行 383-411

```python
global_feature = self.increase_dim(x)  # (B, 1024, N)
global_feature = torch.max(global_feature, dim=-1)[0]  # (B, 1024)
coarse_point_cloud = self.coarse_pred(global_feature).reshape(bs, -1, 3)  # (B, 512, 3)
```

- `coarse_pred`: `Linear(1024,1024)→ReLU→Linear(1024,1536)`，其中 1536=3×512
- **全局特征直接预测 512 个绝对坐标**，不依赖任何锚点
- Grouper 输出 128 个 token（`coor: (B,3,128), f: (B,128,128)`）

### 3.2 本项目的 Per-Token 改造 (exp1-exp4)

文件: `models/PGST.py`

关键改动：
1. **Grouper 输出从 128 增加到 256 个 token**（`center_num: [512, 256]`），FPS 链: `xyz(2048) → FPS→512 → FPS→256`
2. **每个 encoder token 独立预测一个 coarse 点**，而非全局预测所有点
3. 使用 offset 回归: `coarse = coor + offset`

```python
# 每个 encoder token 预测相对于其 grouper center 的偏移
coarse_offset = self.coarse_pred(
    torch.cat([token_features, coor], dim=-1))  # (B, 256, 3)
coarse = coor + coarse_offset  # (B, 256, 3)
# 不足 512 个时，用 FPS 从输入补充
if coarse.size(1) < 512:
    supp = fps(xyz, 256)  # 从输入表面采 256 个点
    coarse = cat([coarse, supp], dim=1)  # (B, 512, 3)
```

- `coarse_pred`: `Linear(1027,512)→GELU→Linear(512,3)`（输入 1024 维 token feature + 3 维坐标）
- **关键缺陷**: 所有的 coor 来自输入表面的 FPS 采样，offset 只能在已知表面附近微调，**无法覆盖缺失/遮挡区域**。即使有 FPS supplement 补充的 256 个点，也全部在已知表面上。

### 3.3 Global-Local Anchor 改造 (exp6，当前版本)

为了解决上述缺陷，引入了双头 coarse 生成：

```python
# Local: per-token offset（表面附近精细对齐）
coarse_offset = self.coarse_pred(
    torch.cat([token_features, coor], dim=-1))  # (B, 256, 3)
coarse_local = coor + coarse_offset  # (B, 256, 3)

# Global: 从 global_feature 直接预测绝对坐标（覆盖缺失区域）
coarse_global = self.global_coarse_pred(global_feature).reshape(B, G, 3)  # (B, 256, 3)

# 拼接: 256 + 256 = 512 = num_query
coarse = torch.cat([coarse_local, coarse_global], dim=1)  # (B, 512, 3)
```

新增模块:
- `global_coarse_pred`: `Linear(1024,1024)→GELU→Linear(1024,768)`（768=256×3，输出 256 个绝对坐标）
- 新增参数: 1.84M（全随机初始化，不在预训练 checkpoint 中）
- FPS supplement 分支已移除（256+256=512 刚好等于 num_query）

## 四、实验结果

### 4.1 各实验对比（指标: F-Score ↑, CDL1 ↓）

| 实验 | 策略 | CDL1 | F-Score | 训练轮数 |
|------|------|------|---------|---------|
| PoinTr 原版 | 全局预测 | - | - | 361 epoch 预训练 |
| exp4 (基线) | Per-token offset + FPS supplement | **8.072** | 0.775 | 150 epoch, best@140 |
| exp5 | exp4 + 差分学习率 5x | ~8.25 | ~0.775 | 80+ epoch |
| exp6 (当前) | Global-Local Anchor, 差分 3x | **8.160** (best@70) | 0.780 (best@90) | Early Stop@100 |

### 4.2 exp6 训练曲线 (validation)

```
Epoch   CDL1     F-Score
  0     8.883    0.750
 10     8.346    0.765
 20     8.366    0.767
 30     8.354    0.769
 40     8.241    0.773
 50     8.241    0.774
 60     8.294    0.775
 70     8.160    0.780  ← CDL1 best
 80     8.318    0.775
 90     8.239    0.782  ← F-Score best
100     8.273    0.779  ← Early Stop
```

### 4.3 问题分析

exp6 的 Global-Local Anchor 改造**效果不如预期**：
- CDL1 best=8.160 vs exp4 的 8.072，**反而变差了 0.088**
- F-Score 有提升（0.780 vs 0.775），说明点云覆盖率有改善
- 但精度下降，可能原因：global head 随机初始化，150 epoch 内尚未充分收敛；local 和 global 两路点质量不均衡导致 decoder 困扰

## 五、关键代码位置索引

| 内容 | 文件 | 行号 |
|------|------|------|
| DGCNN_Grouper (PGST版) | `models/PGST.py` | 654-767 |
| PGST.__init__ (模型定义) | `models/PGST.py` | 940-1020 |
| global_coarse_pred 定义 | `models/PGST.py` | 988-992 |
| coarse_pred 定义 | `models/PGST.py` | 993-997 |
| query_ranking 定义 | `models/PGST.py` | 1013-1020 |
| PGST.forward (coarse生成) | `models/PGST.py` | 1055-1063 |
| query_ranking + decoder | `models/PGST.py` | 1067-1089 |
| Decoder cross-attn (KNN索引) | `models/PGST.py` | 497-506 |
| CrossAttnBlock dual-attn | `models/PGST.py` | 330-450 |
| 原始 PoinTr coarse 生成 | `models/Transformer.py` | 383-411 |
| 原始 PoinTr grouper | `models/dgcnn_group.py` | 43-144 |
| Checkpoint shape-filtering | `tools/builder.py` | 223-234 |
| pertoken 差分学习率 | `tools/builder.py` | 88-104 |
| 训练配置 | `cfgs/PCN_models/pertoken_core_gpu1.yaml` | 全文件 |

## 六、已知约束

1. **DGCNN_Grouper 的 FPS 链**: `xyz(2048) → FPS→512 → FPS→256`，所有 `coor` 点都在输入表面上。这是 `coarse_local` 无法脱离表面的根本原因。
2. **Decoder cross-attention 双机制**: global attention（feature-only，无位置约束）+ local attention（`knn_point(K, v_pos, q_pos)`，按位置 KNN 索引）。理论上 decoder 能处理全局/局部混合的 coarse 点。
3. **query_ranking 当 num_query=512 时等于只重排序**（全部保留），但它仍在学习区分点质量。
4. **训练策略**: 冻结 grouper + pos_embed + input_proj，只训练 encoder adapter + decoder + pertoken 模块。`global_coarse_pred` 虽然是随机初始化，但通过 3x 差分学习率加速收敛。
5. **数据集**: PCN_Core 是 PCN 的精简版，训练样本较少。

## 七、待解决问题

Global-Local Anchor 方案 (exp6) 未能超越简单的 per-token offset 方案 (exp4)。需要诊断原因并改进。可能的改进方向：

- global_coarse_pred 的容量/初始化/学习率是否足够？
- 256 global + 256 local 的比例是否合理？
- local 和 global 点直接拼接后 decoder 是否能有效利用？
- 是否需要更长的训练？
- 整体架构方向是否正确？
