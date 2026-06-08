#!/bin/bash
# KITTI数据集和预训练模型下载脚本
# 使用方法: bash scripts/download_kitti.sh

set -e

echo "========================================"
echo "KITTI数据集和预训练模型下载脚本"
echo "========================================"
echo ""

# 设置目录
KITTI_DIR="data/KITTI"
CKPT_DIR="ckpt"

# 创建目录
mkdir -p ${KITTI_DIR}/{bboxes,cars,tracklets}
mkdir -p ${CKPT_DIR}

echo "Step 1: 下载KITTI预训练模型"
echo "----------------------------------------"
# KITTI预训练模型 (来自PoinTr官方)
KITTI_MODEL_URL="https://cloud.tsinghua.edu.cn/f/734011f0b3574ab58cff/?dl=1"
KITTI_MODEL_PATH="${CKPT_DIR}/PoinTr_KITTI.pth"

if [ -f "${KITTI_MODEL_PATH}" ]; then
    echo "预训练模型已存在: ${KITTI_MODEL_PATH}"
else
    echo "正在下载KITTI预训练模型..."
    echo "URL: ${KITTI_MODEL_URL}"
    wget --no-check-certificate -O "${KITTI_MODEL_PATH}" "${KITTI_MODEL_URL}" || {
        echo "下载失败，请手动下载:"
        echo "  1. 访问: https://github.com/yuxumin/PoinTr"
        echo "  2. 下载KITTI预训练模型"
        echo "  3. 放置到: ${KITTI_MODEL_PATH}"
    }
fi

echo ""
echo "Step 2: 下载KITTI数据集"
echo "----------------------------------------"
# KITTI数据集下载链接
KITTI_DATA_URL="https://huggingface.co/datasets/XuminYu/KITTI_point_cloud_completion/tree/main"
KITTI_ZIP="${KITTI_DIR}/KITTI.zip"

if [ -d "${KITTI_DIR}/cars" ] && [ "$(ls -A ${KITTI_DIR}/cars)" ]; then
    echo "KITTI数据集已存在"
else
    echo "请手动下载KITTI数据集:"
    echo "  方法1 - HuggingFace:"
    echo "    访问: ${KITTI_DATA_URL}"
    echo "    下载KITTI.zip并解压到: ${KITTI_DIR}/"
    echo ""
    echo "  方法2 - Google Drive:"
    echo "    访问: https://drive.google.com/drive/folders/1fSu0_huWhticAlzLh3Ejpg8zxzqO1z-F"
    echo "    下载后解压到: ${KITTI_DIR}/"
    echo ""
    echo "  方法3 - Tsinghua Cloud:"
    echo "    访问: https://cloud.tsinghua.edu.cn/f/ac82414f884d445ebd54/?dl=1"
    echo "    下载后解压到: ${KITTI_DIR}/"
    echo ""
    echo "数据集结构应为:"
    echo "  data/KITTI/"
    echo "  ├── bboxes/       # 边界框文件 *.txt"
    echo "  ├── cars/         # 点云文件 *.pcd"
    echo "  ├── tracklets/    # 追踪信息"
    echo "  └── KITTI.json    # 数据集索引(已创建)"
fi

echo ""
echo "Step 3: 验证安装"
echo "----------------------------------------"
# 检查文件
if [ -f "${KITTI_MODEL_PATH}" ]; then
    echo "✓ 预训练模型: ${KITTI_MODEL_PATH}"
    ls -lh "${KITTI_MODEL_PATH}"
else
    echo "✗ 预训练模型缺失: ${KITTI_MODEL_PATH}"
fi

if [ -d "${KITTI_DIR}/cars" ] && [ "$(ls -A ${KITTI_DIR}/cars 2>/dev/null)" ]; then
    echo "✓ KITTI数据集: ${KITTI_DIR}/"
    echo "  点云文件数: $(ls ${KITTI_DIR}/cars/*.pcd 2>/dev/null | wc -l)"
else
    echo "✗ KITTI数据集缺失: ${KITTI_DIR}/"
fi

echo ""
echo "========================================"
echo "下载和验证完成!"
echo "========================================"
echo ""
echo "测试命令示例:"
echo "  screen -dmS kitti_test bash -c '"
echo "    source /home/fubenhao/data/fubenhao_data/miniforge3/etc/profile.d/conda.sh"
echo "    eval \"\$(conda shell.bash hook)\""
echo "    conda activate pgst"
echo "    cd /home/fubenhao/data/fubenhao_data/PointGST-main_pure/completion"
echo "    CUDA_VISIBLE_DEVICES=1 python main.py \\"
echo "      --config cfgs/KITTI_models/PGST.yaml \\"
echo "      --model pgst \\"
echo "      --test \\"
echo "      --ckpts ckpt/PoinTr_KITTI.pth \\"
echo "      --exp_name kitti_test"
echo "  '"
