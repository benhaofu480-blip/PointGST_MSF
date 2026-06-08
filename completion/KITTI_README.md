# KITTI数据集测试指南

本文档说明如何在PGST项目中添加和测试KITTI数据集。

## 文件结构

```
completion/
├── data/KITTI/                       # KITTI数据集目录
│   ├── bboxes/                         # 边界框文件 (*.txt)
│   ├── cars/                           # 点云文件 (*.pcd)
│   ├── tracklets/                      # 追踪信息
│   └── KITTI.json                      # 数据集索引文件 (已创建)
├── cfgs/dataset_configs/KITTI.yaml     # KITTI数据集配置 (已创建)
├── cfgs/KITTI_models/PGST.yaml         # KITTI测试模型配置 (已创建)
├── KITTI_metric.py                     # KITTI评估脚本 (已创建)
└── scripts/download_kitti.sh           # 下载辅助脚本 (已创建)
```

## 步骤1: 下载KITTI数据集

### 方法1: 使用下载脚本
```bash
cd /home/fubenhao/data/fubenhao_data/PointGST-main_pure/completion
bash scripts/download_kitti.sh
```

### 方法2: 手动下载
1. 从以下任一来源下载:
   - HuggingFace: https://huggingface.co/datasets/XuminYu/KITTI_point_cloud_completion/tree/main
   - Google Drive: https://drive.google.com/drive/folders/1fSu0_huWhticAlzLh3Ejpg8zxzqO1z-F
   - Tsinghua Cloud: https://cloud.tsinghua.edu.cn/f/ac82414f884d445ebd54/?dl=1

2. 解压到 `data/KITTI/` 目录:
```bash
unzip KITTI.zip -d data/KITTI/
# 确保结构为:
# data/KITTI/
# ├── bboxes/
# ├── cars/
# ├── tracklets/
# └── KITTI.json (如覆盖请恢复)
```

## 步骤2: 下载预训练模型

### PoinTr官方KITTI模型
```bash
# 下载到ckpt目录
cd /home/fubenhao/data/fubenhao_data/PointGST-main_pure/completion
wget --no-check-certificate \
  "https://cloud.tsinghua.edu.cn/f/734011f0b3574ab58cff/?dl=1" \
  -O ckpt/PoinTr_KITTI.pth
```

或使用脚本:
```bash
bash scripts/download_kitti.sh
```

### 使用自己的模型
也可以使用在PCN上训练好的模型进行KITTI测试:
```bash
--ckpts ckpt/AdaPoinTr_ps55.pth
```

## 步骤3: 运行KITTI测试

### 标准测试命令
```bash
cd /home/fubenhao/data/fubenhao_data/PointGST-main_pure/completion

screen -dmS kitti_test bash -c '
source /home/fubenhao/data/fubenhao_data/miniforge3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate pgst
CUDA_VISIBLE_DEVICES=1 python main.py \
  --config cfgs/KITTI_models/PGST.yaml \
  --model pgst \
  --test \
  --ckpts ckpt/PoinTr_KITTI.pth \
  --exp_name kitti_test
'
```

### 使用PoinTr原始模型测试对比
```bash
screen -dmS kitti_pointr bash -c '
source /home/fubenhao/data/fubenhao_data/miniforge3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate pgst
CUDA_VISIBLE_DEVICES=1 python main.py \
  --config cfgs/KITTI_models/PGST.yaml \
  --model linear \
  --test \
  --ckpts ckpt/PoinTr_KITTI.pth \
  --exp_name kitti_pointr_test
'
```

## 步骤4: 计算KITTI指标 (MMD)

KITTI数据集使用 **MMD (Minimum Matching Distance)** 作为评估指标。

### 运行评估脚本
```bash
cd /home/fubenhao/data/fubenhao_data/PointGST-main_pure/completion

python KITTI_metric.py \
  --vis experiments/config/kitti_test/ \
  --output kitti_results.txt
```

### 指标说明
- **MMD**: Mean of Chamfer Distance across all test samples
- PoinTr官方KITTI性能: MMD ≈ 5.04e-4 (0.000504)

## 配置文件说明

### cfgs/dataset_configs/KITTI.yaml
```yaml
NAME: KITTI
CATEGORY_FILE_PATH: data/KITTI/KITTI.json
N_POINTS: 2048                    # KITTI输入点云数
CLOUD_PATH: data/KITTI/cars/%s.pcd
BBOX_PATH: data/KITTI/bboxes/%s.txt
```

### cfgs/KITTI_models/PGST.yaml
关键配置:
- `model.NAME`: AdaPoinTr_PGST (使用MSF改进版)
- `dataset.test.subset`: test (仅测试模式)
- `loss_config`: 测试时禁用Laplacian等辅助损失

## 故障排除

### 问题1: 找不到KITTI.json
```bash
# 确认文件存在
ls -la data/KITTI/KITTI.json

# 如缺失,配置文件已自动创建
```

### 问题2: 点云文件找不到
确保解压后的结构正确:
```bash
# 应显示约2409个.pcd文件 (KITTI.json中定义的样本数)
ls data/KITTI/cars/*.pcd | wc -l

# 应显示约2409个.txt文件
ls data/KITTI/bboxes/*.txt | wc -l
```

### 问题3: 模型加载失败
检查:
1. `--model pgst` 参数是否正确
2. 模型文件路径是否正确
3. 模型配置与权重是否匹配

### 问题4: CUDA内存不足
KITTI测试batch size固定为1（单个汽车点云），一般不会出现OOM。
如出现，检查:
```bash
# 清空GPU缓存
python -c "import torch; torch.cuda.empty_cache()"
```

## 参考

- PoinTr GitHub: https://github.com/yuxumin/PoinTr
- PoinTr DATASET.md: https://github.com/yuxumin/PoinTr/blob/master/DATASET.md
- GRNet (KITTI数据源): https://github.com/hzxie/GRNet
