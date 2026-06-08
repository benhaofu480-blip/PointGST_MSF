#!/usr/bin/env python3
"""
Stage-1 论文图红圈标注脚本

功能：在 Sigmoid 比 PCSA 明显生成得更好的局部区域画红色半透明椭圆圈。

用法:
  cd completion
  python annotate_stage1_figures.py

输出:
  VIS/vis_stage1_sigmoid_vs_pcsa_paper/annotated/
    带红圈的 PNG（文件名加 _annotated）

注意：
- 坐标以图像像素为单位（左上角为 (0,0)）
- 每个图的圈坐标已硬编码为示例，用户可根据实际图片调整
- 原图不会被覆盖
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from PIL import Image


# ==================== 配置区（可修改）====================
SRC_DIR = Path("VIS/vis_stage1_sigmoid_vs_pcsa_paper")
OUT_DIR = SRC_DIR / "annotated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 每张图要标注的红圈列表
# 格式：(文件名, [(cx, cy, width, height, angle), ...])
# cx, cy: 椭圆中心（像素，左上角为原点）
# width, height: 椭圆长短轴像素长度
# angle: 旋转角度（度）
ANNOTATIONS = {
    "01_02691156_飞机_25a057e935aeb4b6842007370970c479.png": [
        # 飞机尾翼细杆（Sigmoid 明显更完整）
        (920, 380, 180, 45, -25),
        # 机翼尖端缺失区域
        (620, 520, 140, 55, 15),
    ],
    "04_03001627_椅子_68c7f82dd1e1634d9338458f802f5ad7.png": [
        # 椅子靠背竖杆（Sigmoid 连贯，PCSA 断裂）
        (780, 420, 220, 38, -8),
        # 座面边缘尖角
        (550, 680, 120, 65, 30),
    ],
    "07_04379243_桌子_1dc7f7d076afd0ccf11c3739edd52fa3.png": [
        # 用户橙色圈：桌腿下部结构（Sigmoid 更完整连贯）
        (920, 820, 120, 180, 0),
        # 桌面边缘遮挡恢复
        (480, 380, 160, 70, -5),
    ],
    "05_03636649_台灯_e1fe4f81f074abc3e6597d391ab6fcc1.png": [
        # 台灯灯罩边缘
        (720, 290, 150, 90, 0),
        # 灯杆细部
        (710, 620, 80, 35, -20),
    ],
    "06_04256520_沙发_2fc5cf498b0fa6de1525e8c8552c3a9c.png": [
        # 用户橙色圈：沙发右侧局部结构
        (1050, 420, 180, 90, 0),
    ],
}

# 红圈样式
ELLIPSE_COLOR = "#FF2D2D"
ELLIPSE_ALPHA = 0.85
ELLIPSE_LINEWIDTH = 3.5


def draw_annotated_figure(src_path: Path, circles: list[tuple], out_path: Path):
    """读取原图，在指定位置画红椭圆，保存新图。"""
    img = Image.open(src_path)
    fig, ax = plt.subplots(figsize=(img.width / 100, img.height / 100), dpi=100)
    ax.imshow(img)
    ax.axis("off")

    for cx, cy, w, h, angle in circles:
        ellipse = mpatches.Ellipse(
            (cx, cy),
            width=w,
            height=h,
            angle=angle,
            fill=False,
            edgecolor=ELLIPSE_COLOR,
            linewidth=ELLIPSE_LINEWIDTH,
            alpha=ELLIPSE_ALPHA,
        )
        ax.add_patch(ellipse)

    plt.tight_layout(pad=0)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  saved annotated: {out_path.name}")


def main():
    print(f"Source dir : {SRC_DIR.resolve()}")
    print(f"Output dir : {OUT_DIR.resolve()}")
    print("-" * 60)

    for fname, circles in ANNOTATIONS.items():
        src = SRC_DIR / fname
        if not src.exists():
            print(f"WARNING: {fname} not found, skip")
            continue

        out_name = src.stem + "_annotated.png"
        out_path = OUT_DIR / out_name
        draw_annotated_figure(src, circles, out_path)

    print("-" * 60)
    print(f"Done. Annotated figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
