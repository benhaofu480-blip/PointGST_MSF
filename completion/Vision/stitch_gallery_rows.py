"""
把多张画廊 PNG 竖拼：最上一行列标题，下面每行一张原图（去掉各行自带的小标题）。

用法:
  python Vision/stitch_gallery_rows.py --images a.png b.png c.png --out Vision/output/xxx.png
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

COLUMNS = ("输入", "PCSA", "MSF", "G.T.")
FONT_PATH = "/usr/share/fonts/truetype/arphic/uming.ttc"
# 默认加大加粗，便于插入 PDF
HEADER_FONT_SIZE = 52
HEADER_STROKE = 1
HEADER_H = 108
ROW_GAP = 12
SAVE_DPI = 300
# 画廊图每列自带标题区，裁掉后只保留点云对比
CROP_TOP_PX = 88


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if os.path.isfile(FONT_PATH):
        return ImageFont.truetype(FONT_PATH, size=size)
    return ImageFont.load_default()


def _render_header(
    width: int,
    col_gap: int,
    fontsize: int = HEADER_FONT_SIZE,
    stroke: int = HEADER_STROKE,
    header_h: int = HEADER_H,
) -> np.ndarray:
    """四列标题（PIL 绘制，stroke 模拟加粗）。"""
    n = len(COLUMNS)
    panel_w = (width - col_gap * (n - 1)) // n
    centers = [panel_w // 2 + i * (panel_w + col_gap) for i in range(n)]

    img = Image.new("RGB", (width, header_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = _load_font(fontsize)

    for label, cx in zip(COLUMNS, centers):
        bbox = draw.textbbox((0, 0), label, font=font, stroke_width=stroke)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = cx - tw // 2
        y = (header_h - th) // 2 - bbox[1]
        draw.text(
            (x, y), label, fill=(0, 0, 0), font=font,
            stroke_width=stroke, stroke_fill=(0, 0, 0),
        )

    return np.asarray(img)


def stitch(
    images: list[np.ndarray],
    col_gap: int = 36,
    fontsize: int = HEADER_FONT_SIZE,
    stroke: int = HEADER_STROKE,
    header_h: int = HEADER_H,
) -> np.ndarray:
    w = max(im.shape[1] for im in images)
    header = _render_header(w, col_gap, fontsize=fontsize, stroke=stroke, header_h=header_h)
    rows = []
    for i, im in enumerate(images):
        if im.shape[1] < w:
            pad = np.ones((im.shape[0], w - im.shape[1], 3), dtype=np.uint8) * 255
            im = np.concatenate([im, pad], axis=1)
        rows.append(im)
        if i < len(images) - 1:
            rows.append(np.ones((ROW_GAP, w, 3), dtype=np.uint8) * 255)
    body = np.concatenate(rows, axis=0)
    gap = np.ones((ROW_GAP, w, 3), dtype=np.uint8) * 255
    return np.concatenate([header, gap, body], axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--crop-top", type=int, default=CROP_TOP_PX)
    parser.add_argument("--col-gap", type=int, default=36)
    parser.add_argument("--no-crop", action="store_true", help="保留原图每行自带列名")
    parser.add_argument("--header-fontsize", type=int, default=HEADER_FONT_SIZE)
    parser.add_argument("--header-stroke", type=int, default=HEADER_STROKE, help="描边粗细，越大越粗")
    parser.add_argument("--header-h", type=int, default=HEADER_H)
    parser.add_argument("--save-dpi", type=int, default=SAVE_DPI, help="写入 PNG 的 DPI 元数据，PDF 插入更清晰")
    cli = parser.parse_args()

    imgs = []
    for p in cli.images:
        arr = np.asarray(Image.open(p).convert("RGB"))
        if not cli.no_crop and arr.shape[0] > cli.crop_top:
            arr = arr[cli.crop_top :, :, :]
        imgs.append(arr)

    out = stitch(
        imgs, col_gap=cli.col_gap,
        fontsize=cli.header_fontsize, stroke=cli.header_stroke, header_h=cli.header_h,
    )
    os.makedirs(os.path.dirname(os.path.abspath(cli.out)), exist_ok=True)
    dpi = (cli.save_dpi, cli.save_dpi)
    Image.fromarray(out).save(cli.out, dpi=dpi)
    print(f"saved {cli.out}  size={out.shape[1]}x{out.shape[0]}")


if __name__ == "__main__":
    main()
