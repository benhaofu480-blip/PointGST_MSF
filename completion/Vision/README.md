# Vision — 论文配图

本目录存放点云补全相关的高清示意图脚本与输出。

## 图1：PCN 数据集（完整 + 8 残缺视角）

```bash
cd completion
conda activate pgst
python Vision/plot_pcn_dataset_fig1.py
```

默认样本：训练集 **椅子**（`03001627` / `1015e71a0d21b127de03ab2a27ba7531`）。

输出：`Vision/output/pcn_fig1_complete_and_8partials.png`

### 字体（对齐图3 PDF 配置）

| 元素 | 字号 | 粗细 |
|------|------|------|
| 总标题「PCN 数据集示例」 | 32 | bold |
| 各子图标题（完整点云 / 残缺视角） | 24 | bold |

重新生成：`python Vision/plot_pcn_dataset_fig1.py`

### 风格说明（对齐 PoinTr Fig.8）

- **完整点云**：橙黄 `#E8A020`（G.T. 行）
- **8 个残缺视角**：浅灰 `#8C8C8C`（Input 行）
- **3D scatter**（同 `vis_stage1` / `vis_hard_ft`）；`dpi=220`；无底部说明文字

### 常用参数

```bash
# 换类别 / 样本
python Vision/plot_pcn_dataset_fig1.py --taxonomy_id 02958343 --model_id 1005ca47e516495512da0dbf3c68e847

# 微调视角
python Vision/plot_pcn_dataset_fig1.py --elev 20 --azim -60
```

## 图2：一阶段 MSF vs PCSA（complete ep150 best）

8 类各 1 例，**4 列**：输入 → PCSA（蓝）→ MSF（红）→ G.T.（橙）

```bash
python Vision/plot_stage1_msf_vs_pcsa_complete.py
```

| 模型 | 权重 | Official test |
|------|------|----------------|
| MSF Sigmoid | `exp_MSF_Pure_Group_sigmoid_complete_seed42/ckpt-best.pth` | F=0.839, CDL1=6.654 |
| PCSA | `exp_PCSA_complete_seed42/ckpt-best.pth` | F=0.836, CDL1=6.715 |

- 样本：`data/stage1_complete_vis_8cat.txt`
- 输出：`Vision/output/stage1_msf_vs_pcsa_complete_8cat.png`

## 画廊：多样本分开出图（供人工挑选）

不显示 CD；**排除**此前 `8cat` / `8cat_alt` 用过的样本；每类默认 3 个，**每张图一个物体**。

```bash
python Vision/plot_stage1_msf_vs_pcsa_gallery.py
python Vision/plot_stage1_msf_vs_pcsa_gallery.py --per-category 4 --elev 12 --azim -120
```

- 推荐输出目录：`Vision/output/stage1_gallery_18_45/`（视角 `(18°, 45°)`，`INDEX.txt` 索引）
- 列顺序：输入 → PCSA → MSF → G.T.
- 权重与指标：同 **图2**（complete ep150 `ckpt-best`）

### 画廊单图绘制要点

- 从原始 PCD 读 partial / GT（避免 `sample_list` 与 `cache_pcn` 下标错位）
- 无 `model_id` 标注；列标题默认 26pt；`--pad-ratio 0.42`、`--col-gap 36`

```bash
python Vision/plot_stage1_msf_vs_pcsa_gallery.py \
  --picks-file data/stage1_complete_vis_gallery.txt \
  --out-dir Vision/output/stage1_gallery_18_45 \
  --pad-ratio 0.42 --col-gap 36
```

## 图3：多张画廊竖拼（论文用，推荐配置）

将多张 `stage1_gallery_18_45` 单图拼成 **一行列名 + 多行对比**，顶栏文字加大，便于插入 PDF。

```bash
python Vision/stitch_gallery_rows.py \
  --images \
  Vision/output/stage1_gallery_18_45/03001627_椅子_3db18530ff6bbb50f130e9cdbbb1cf40.png \
  Vision/output/stage1_gallery_18_45/03001627_椅子_91b8fe4616208bd4cf752e9bed38184f.png \
  Vision/output/stage1_gallery_18_45/03636649_台灯_c9a0c193805df62accbc9e602dbf6a4a.png \
  --out Vision/output/stage1_gallery_18_45/three_samples_stitched.png
```

**成品示例**：`Vision/output/stage1_gallery_18_45/three_samples_stitched.png`

### 顶栏文字配置（已验证，适合 PDF）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--header-fontsize` | `52` | 列名：输入 / PCSA / MSF / G.T. |
| `--header-stroke` | `1` | PIL 描边模拟加粗；`0` 为不加粗，`3` 过粗 |
| `--header-h` | `108` | 顶栏高度（像素） |
| `--save-dpi` | `300` | 写入 PNG 的 DPI 元数据 |
| `--col-gap` | `36` | 须与画廊单图 `--col-gap` 一致，列名才对齐 |
| `--crop-top` | `88` | 裁掉单图自带列名，避免与顶栏重复 |

- 字体：`/usr/share/fonts/truetype/arphic/uming.ttc`（PIL 绘制，避免 matplotlib 中文不显示）
- 略加粗：仅 `--header-stroke 1`；字号 52 已足够时勿再加大 stroke

### 可选参数

```bash
# 保留每张单图自带列名（会出现两套标题，一般不用）
python Vision/stitch_gallery_rows.py --images ... --out ... --no-crop

# 顶栏再大一点
python Vision/stitch_gallery_rows.py ... --header-fontsize 60 --header-stroke 1
```

## 图4：沙发 + 桌子 4×4（双视角，无红圈）

沙发 `1f75847…`、桌子 `1dc7f7d0…` 各 2 个视角，4 列：**输入 → PCSA → MSF → G.T.**  
顶栏字号 52 / stroke 1（同图3）；**PCN_core 一阶段权重**（见下）；**无** CD 注释、**无**红圈。

| 列 | 权重 |
|----|------|
| MSF | `exp_MSF_Pure_Group_sigmoid/ckpt-best.pth` |
| PCSA | `exp_pksa_bug_author_impl/ckpt-epoch-150.pth` |

```bash
python Vision/plot_stage1_sofa_table_4x4.py
```

输出：`Vision/output/stage1_sofa_table_4x4.png`（4 行 × 4 列）

## 其他脚本

| 脚本 | 用途 |
|------|------|
| `plot_stage1_gallery_auto_zoom.py` | 自动局部放大第二行（实验性，非论文主图） |
| `annotate_stage1_figures.py`（项目根目录） | 在已有 PNG 上画红圈标注（本仓库论文主图不用） |
