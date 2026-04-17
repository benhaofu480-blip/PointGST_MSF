# ShapeNetPart 复现说明

按照本指南，使用Point-BERT + PointGST在ShapeNetPart数据集上复现论文结果（mIoU = 85.8%）。

## 📋 复现步骤

### 1. 数据集准备

```bash
cd data/

# 下载ShapeNetPart数据集
wget https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip
unzip shapenetcore_partanno_segmentation_benchmark_v0_normal.zip
mv shapenetcore_partanno_segmentation_benchmark_v0_normal ShapeNetPart

# 返回项目根目录
cd ..
```

### 2. 数据预处理

```bash
python tools/preprocess_shapenetpart.py
```

这将生成以下文件：
- `data/ShapeNetPart/train_points.npy` (训练点云)
- `data/ShapeNetPart/train_labels.npy` (物体类别)
- `data/ShapeNetPart/train_seg.npy` (部件分割标签)
- `data/ShapeNetPart/test_*.npy` (测试集)

### 3. 下载预训练权重

下载Point-BERT预训练权重：
```bash
mkdir -p pretrained
cd pretrained
wget https://github.com/Pang-Yatian/Point-MAE/releases/download/main/pretrain.pth
mv pretrain.pth pointbert_pretrain.pth
cd ..
```

### 4. 训练模型

```bash
bash train_shapenetpart.sh
```

训练参数：
- 优化器：AdamW (lr=5e-4, weight_decay=0.05)
- 调度器：CosineLR (300 epochs, warmup=10)
- Batch size：32
- Epochs：300

### 5. 测试模型

```bash
bash test_shapenetpart.sh
```

**期望结果（输出三个指标）**：
```
>> Trainable Parameters: 5.58M / 357.60M (1.56%)
============================================================
Final Test Results:
  Params.(M): 5.58
  Cls.mIoU(%): 85.12
  Inst.mIoU(%): 85.80
============================================================
```

**指标说明**：
- **Params.(M)**: 可训练参数量（百万），应为5.58M
- **Cls.mIoU(%)**: 类别平均mIoU（16个物体类别各自的mIoU再平均）
- **Inst.mIoU(%)**: 实例平均mIoU（每个样本的mIoU再平均），论文指标

**复现成功标准**：Inst.mIoU达到**85.8%**

### 6. 监控训练

**TensorBoard**：
```bash
tensorboard --logdir experiments/finetune_shapenetpart_pgst/mae/shapenetpart_reproduce/TFBoard
```

**日志文件**：
```bash
tail -f experiments/finetune_shapenetpart_pgst/mae/shapenetpart_reproduce/*.log
```

## 🎯 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `trans_dim` | 384 | Point-BERT特征维度 |
| `depth` | 12 | Transformer层数 |
| `num_heads` | 6 | 注意力头数 |
| `rank` | 36 | PCSA适配器秩 |
| `lr` | 5e-4 | 学习率 |
| `epochs` | 300 | 训练轮数 |
| `npoints` | 2048 | 输入点数 |

## 📊 可训练参数

**总计：5.58M**
- PCSA适配器：0.35M (12层 × 29K)
- 上采样网络：0.4M
- 分割头：0.15M
- 其他可训练层：~4.7M (解码器部分层)

## 🔧 代码结构

```
datasets/ShapeNetPartDataset.py          # 数据集加载
models/PointTransformerPartSeg_PGST.py   # 分割模型
cfgs/mae/finetune_shapenetpart_pgst.yaml # 训练配置
main.py                                  # 入口（已修改支持ShapeNetPart）
tools/preprocess_shapenetpart.py         # 数据预处理
```

## 🐛 常见问题

**Q1: 数据预处理报错？**
A: 确保ShapeNetPart目录结构正确，包含train_test_split文件夹和16个子目录。

**Q2: 显存不足？**
A: 减小batch size（total_bs）或npoints。

**Q3: mIoU上不去？**
A: 检查是否加载了预训练权重，确保optimizer.part='adapt'。

## 📈 改进方向（可选）

当前代码已高度模块化，便于后续研究：
- **Multi-Scale PCSA**: 修改PCSA支持多尺度特征
- **Part-Aware**: 添加类别感知的适配器
- **Edge Loss**: 增强边界分割精度

## ✅ 复现成功标准

当测试mIoU达到**85.8%**时，复现成功！

---

**注意**：训练大约需要8-12小时（单张RTX 3090）。
