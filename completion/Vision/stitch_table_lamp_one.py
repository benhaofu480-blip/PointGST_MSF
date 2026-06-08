"""从已有候选图拼接：上=01 图上半（沙发），下=pick_01 下半（台灯）（不重跑模型）。"""

import os

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "output")
CAND = os.path.join(OUT_DIR, "sofa_bottom_candidates")

IMG_TABLE = os.path.join(CAND, "01_桌子_table_dCD1.7.png")
IMG_LAMP = os.path.join(CAND, "pick_01_台灯_lamp.png")
OUT = os.path.join(OUT_DIR, "stage1_sofa_table_4x4.png")

# 与 stitch_gallery_rows / plot_stage1_sofa_table_4x4 一致
TOP_PAD = 10
HEADER_H = 108
ROW_GAP = 36


def _body_split(img: np.ndarray, take: str):
    """顶栏以下 body：take='top' 第1个物体两行；take='bottom' 第2个物体（去掉底部标注行）。"""
    top = TOP_PAD + HEADER_H + ROW_GAP
    body = img[top:]
    # 去掉 pick 图底部「下行: …」标注
    if body.shape[0] > 60:
        body = body[:-52]
    mid = body.shape[0] // 2
    if take == "top":
        return body[:mid]
    return body[mid:]


def main():
    t = np.array(Image.open(IMG_TABLE).convert("RGB"))
    l = np.array(Image.open(IMG_LAMP).convert("RGB"))
    header = t[: TOP_PAD + HEADER_H + ROW_GAP]
    # 01 图上半 = 沙发；pick_01 下半 = 台灯
    sofa_rows = _body_split(t, "top")
    lamp_rows = _body_split(l, "bottom")
    gap = np.ones((ROW_GAP, header.shape[1], 3), dtype=np.uint8) * 255
    out = np.concatenate([header, sofa_rows, gap, lamp_rows], axis=0)
    os.makedirs(OUT_DIR, exist_ok=True)
    Image.fromarray(out).save(OUT, dpi=(300, 300))
    print(f"saved {OUT}  {out.shape[1]}x{out.shape[0]}")


if __name__ == "__main__":
    main()
