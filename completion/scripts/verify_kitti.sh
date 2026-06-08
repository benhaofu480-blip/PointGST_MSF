#!/bin/bash
# KITTI测试环境验证脚本
# 用法: bash scripts/verify_kitti.sh

echo "========================================"
echo "KITTI测试环境验证"
echo "========================================"
echo ""

CKPT_DIR="ckpt"
KITTI_DIR="data/KITTI"
CONFIG_DIR="cfgs/KITTI_models"
ERRORS=0

# 1. 检查KITTI预训练模型
echo "[1/5] 检查KITTI预训练模型..."
if [ -f "${CKPT_DIR}/PoinTr_KITTI.pth" ]; then
    echo "  ✓ KITTI预训练模型存在"
    ls -lh ${CKPT_DIR}/PoinTr_KITTI.pth
else
    echo "  ✗ KITTI预训练模型缺失: ${CKPT_DIR}/PoinTr_KITTI.pth"
    echo "    下载命令: wget --no-check-certificate 'https://cloud.tsinghua.edu.cn/f/734011f0b3574ab58cff/?dl=1' -O ${CKPT_DIR}/PoinTr_KITTI.pth"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# 2. 检查KITTI数据集
echo "[2/5] 检查KITTI数据集..."
if [ -d "${KITTI_DIR}/cars" ]; then
    PCD_COUNT=$(ls ${KITTI_DIR}/cars/*.pcd 2>/dev/null | wc -l)
    if [ "$PCD_COUNT" -gt 0 ]; then
        echo "  ✓ KITTI点云文件存在: ${PCD_COUNT}个.pcd文件"
    else
        echo "  ✗ KITTI点云文件缺失: ${KITTI_DIR}/cars/ 为空"
        echo "    请下载并解压KITTI数据集到 ${KITTI_DIR}/"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo "  ✗ KITTI cars目录不存在: ${KITTI_DIR}/cars/"
    ERRORS=$((ERRORS + 1))
fi

if [ -d "${KITTI_DIR}/bboxes" ]; then
    BBOX_COUNT=$(ls ${KITTI_DIR}/bboxes/*.txt 2>/dev/null | wc -l)
    if [ "$BBOX_COUNT" -gt 0 ]; then
        echo "  ✓ KITTI边界框文件存在: ${BBOX_COUNT}个.txt文件"
    else
        echo "  ✗ KITTI边界框文件缺失: ${KITTI_DIR}/bboxes/ 为空"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo "  ✗ KITTI bboxes目录不存在: ${KITTI_DIR}/bboxes/"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# 3. 检查KITTI配置文件
echo "[3/5] 检查KITTI配置文件..."
if [ -f "cfgs/dataset_configs/KITTI.yaml" ]; then
    echo "  ✓ KITTI数据集配置存在"
else
    echo "  ✗ KITTI数据集配置缺失: cfgs/dataset_configs/KITTI.yaml"
    ERRORS=$((ERRORS + 1))
fi

if [ -f "${CONFIG_DIR}/PGST.yaml" ]; then
    echo "  ✓ KITTI模型配置存在"
else
    echo "  ✗ KITTI模型配置缺失: ${CONFIG_DIR}/PGST.yaml"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# 4. 检查KITTI.json
echo "[4/5] 检查KITTI数据集索引..."
if [ -f "${KITTI_DIR}/KITTI.json" ]; then
    echo "  ✓ KITTI.json索引文件存在"
    # 检查JSON格式
    python3 -c "import json; json.load(open('${KITTI_DIR}/KITTI.json'))" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "  ✓ KITTI.json格式正确"
    else
        echo "  ✗ KITTI.json格式错误"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo "  ✗ KITTI.json索引文件缺失"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# 5. 检查评估脚本
echo "[5/5] 检查评估脚本..."
if [ -f "KITTI_metric.py" ]; then
    echo "  ✓ KITTI_metric.py评估脚本存在"
else
    echo "  ✗ KITTI_metric.py评估脚本缺失"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# 总结
echo "========================================"
if [ $ERRORS -eq 0 ]; then
    echo "✓ 验证通过! KITTI测试环境已就绪"
    echo ""
    echo "测试命令:"
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
else
    echo "✗ 验证失败: 发现 ${ERRORS} 个问题"
    echo "请根据上方提示修复问题"
    echo ""
    echo "参考文档: KITTI_README.md"
fi
echo "========================================"

exit $ERRORS
