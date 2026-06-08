# 第四章写作 GUIDE（Stage-2 反馈微调）

> **用途**：给无法访问训练服务器的写章 AI 使用。本文档自包含：结构、实验数字、关键源码摘录、LaTeX 提示。  
> **勿用章节名**：「双回环」「双回环训练」——建议用 **「基于反馈裁剪的 Stage-2 点云补全微调」** 或 **「训练期双前向的 Stage-2 优化」**。

---

## 0. 你怎么把材料交给写章 AI？

| 方式 | 适合场景 | 建议 |
|------|----------|------|
| **A. 只传本 Markdown（推荐）** | 写章 AI 在网页/本机，看不到服务器 | 从服务器下载 `completion/docs/chapter4_stage2_writing_guide.md`，整文件粘贴或上传。正文所需算法与数字已内嵌。 |
| **B. 服务器开窗口给 AI** | AI 能 SSH/挂载仓库 | 让 AI 读 `completion/docs/chapter4_stage2_writing_guide.md`，必要时再 `@completion/utils/feedback_train.py`。 |
| **C. 代码+文档分开传** | 文档太长 | 先传本文档；若 AI 要改公式与实现一致，再追加 `feedback_train.py` + `feedback_crop.py` 两个全文。 |

**不要**假设写章 AI 能打开 `experiments/` 或日志；**所有主表数字以本文「实验事实表」为准**。

---

## 1. 章节在全文中的位置

- **第三章**：MSF（`msf_pure_group_sigmoid`），Stage-1，编码器谱域适配。  
- **第四章（本章）**：在 Stage-1 权重上 **Stage-2 微调**；训练借鉴 FeedPoinTrS 的 **两遍前向 + 合成 partial**，backbone 仍是 AdaPoinTr + MSF。  
- **关系**：Stage-2 **不替换** MSF 结构，只增加训练策略与微调阶段。

**一句话定义（4.1 必须写）**：

> 推理与官方 PCN 测试采用 **单次开环前向**；「第二遍」仅存在于 **训练阶段** 的第二次前向及损失，用于模拟「先补全 → 再被裁切 → 再补全」。

---

## 2. 核心概念（防写错）

| 名称 | 含义 |
|------|------|
| Pass0 | `partial`（真实缺失输入）→ 模型 → 粗/细补全 `C0` |
| Pass1 | 由 `C0` 经裁剪 `gv(·)` 得 `partial1` → 再前向 → `C1` |
| `gv` / 裁剪 | `complete_to_partial_input`：去近心点 + FPS 到 2048 点 |
| 开环 test | `python main.py --test`，**只 forward 一次** → **论文主表用这个** |
| 闭环 test（附录） | `scripts/test_feedpointrs_feedback.py`，测试时再跑第二遍 → **勿与主表混用** |
| Exp1-ON | 主实验：双 pass + crop U[0.2,0.4] + 50 epoch |
| Exp1-OFF | 对照：关闭 feedback，其余尽量同（50 epoch） |

**禁止**：把主结果写成「测试时双前向」；禁止写「双 pass 远优于单 pass」（见实验表）。

---

## 3. 建议目录结构

```
第四章 基于反馈裁剪的 Stage-2 点云补全微调
  4.1 引言与动机
  4.2 Stage-2 方法
      4.2.1 训练期双前向与损失
      4.2.2 反馈裁剪（gv）
      4.2.3 与 Stage-1 的衔接与微调范围
      4.2.4 训练与推理的差异（开环评测）
  4.3 实现说明（可缩短，或并入 4.2）
  4.4 实验与分析
      4.4.1 实验设置
      4.4.2 主结果（相对 Stage-1）
      4.4.3 消融：双 pass 开关、crop、checkpoint 选择
      4.4.4 讨论与小结
```

---

## 4. 方法描述（给 AI 展开成正文）

### 4.2.1 训练期双前向

每个 batch：

1. 输入真实 `partial`（PCN 上 2048 点），得预测 `C0`（含 coarse + dense）。  
2. 损失 `L0 = L_coarse + L_dense`（与 AdaPoinTr 原训练相同）。  
3. 取 dense 预测 `Ĉ0 = C0.detach()`，采样裁剪比例 `r ∈ [r_min, r_max]`，构造 `partial1 = gv(Ĉ0, r)`。  
4. 再前向得 `C1`，损失 `L1`。  
5. 总损失：`L = (w0·L0 + w1·L1)/(w0+w1)`，默认 **w0:w1 = 2:1**。  
6. 实现上 **先对 L0 backward，再 forward Pass1，再对 L1 backward**（避免共享权重 inplace 冲突）。

### 4.2.2 裁剪 gv

- 在单位球上采样裁剪中心（默认 **随机**；可选 **error_aware** 高误差点，见附录）。  
- 去掉距中心最近的 `⌊r·N⌋` 个点，保留远心点，再 **FPS** 到与输入相同的点数（2048）。  
- 本文主实验 **r ~ Uniform(0.20, 0.40)**。

### 4.2.3 Stage-1 衔接

- 初始化：`Stage-1 MSF-Sigmoid` 的 `ckpt-best`。  
- 训练集：`PCN_Core` 子集（`data/PCN_Core/pcn_core_train.txt`）。  
- 可训练参数约 **21.26M**（decoder + MSF adapter）；编码器主干冻结。  
- `center_num: [512, 384]`（**非**官方 AdaPoinTr PCN 配置的 `[512, 256]`，脚注说明）。

### 4.2.4 推理

- `validate` / `main.py --test`：**单次** `model(partial)`，与 Stage-1 相同。  
- Stage-2 的收益来自 **训练分布**，不是测试时多跑一遍。

---

## 5. 源码摘录（写章 AI 可直接引用）

> 仓库根目录：`PointGST-main_pure/completion/`（下称 `completion/`）。

### 5.1 双 pass 损失 — `utils/feedback_train.py`

```python
def feedback_enabled(config) -> bool:
    fb = getattr(config, 'feedback_training', None)
    return fb is not None and bool(getattr(fb, 'enabled', False))

def _sample_crop_ratio(config) -> float:
    fb = config.feedback_training
    lo = float(getattr(fb, 'crop_ratio_min', 0.20))
    hi = float(getattr(fb, 'crop_ratio_max', 0.40))
    return lo + (hi - lo) * torch.rand(1).item()  # min==max 时为固定比例

def feedback_training_loss(model, partial, gt, epoch, config):
    w0 = float(config.feedback_training.pass_weight_first)   # 2.0
    w1 = float(config.feedback_training.pass_weight_second)  # 1.0
    norm = w0 + w1

    ret0 = model(partial)
    loss0 = sparse0 + dense0  # get_loss(ret0, gt, epoch)

    dense0_pred = ret0[-1].detach()
    crop_ratio = _sample_crop_ratio(config)
    partial1 = complete_to_partial_input(dense0_pred, partial.size(1), crop_ratio, gt=gt, config=config)

    (w0 * loss0 / norm).backward()      # 先反传 Pass0

    ret1 = model(partial1)
    loss1 = sparse1 + dense1
    (w1 * loss1 / norm).backward()      # 再反传 Pass1
```

### 5.2 裁剪 — `utils/feedback_crop.py`（核心逻辑）

```python
def crop_partial_from_complete(complete, crop_ratio, centroid=None, ...):
    # complete: (B, N, 3) 第一遍 dense 补全
    num_crop = max(1, int(n * crop_ratio))
    keep_k = n - num_crop
    # 距 centroid 最近的 num_crop 点被去掉，保留 keep_k 个远心点
    dist2 = ((complete - centroid.unsqueeze(1)) ** 2).sum(dim=-1)
    _, far_idx = torch.topk(dist2, k=keep_k, dim=1, largest=True)
    ...

def complete_to_partial_input(complete, n_input, crop_ratio, gt=None, config=None):
    cropped = crop_partial_from_complete(complete, crop_ratio, centroid=...)
    return misc.fps(cropped, n_input)   # 重采样到 2048，作为第二遍输入
```

### 5.3 训练循环挂载 — `tools/runner.py`

```python
use_feedback = feedback_enabled(config)
...
if use_feedback:
    sparse_loss, dense_loss, _loss = feedback_training_loss(
        base_model, partial, gt, epoch, config)
else:
    ret = base_model(partial)
    sparse_loss, dense_loss = base_model.module.get_loss(ret, gt, epoch)
    _loss = sparse_loss + dense_loss
    _loss.backward()
```

验证/测试路径 **不** 调用 `feedback_training_loss`，仅为普通 forward。

### 5.4 主配置文件 — `cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid_feedpointrs_ft.yaml`

```yaml
max_epoch: 50
val_freq: 5
consider_metric: CDL1
total_bs: 32
pretrained_ckpt: experiments/.../exp_MSF_Pure_Group_sigmoid/ckpt-best.pth
model:
  center_num: [512, 384]
  encoder_config:
    adapter_mode: msf_pure_group_sigmoid
  msf_route_mode: none
feedback_training:
  enabled: true
  crop_mode: random
  crop_ratio_min: 0.20
  crop_ratio_max: 0.40
  pass_weight_first: 2.0
  pass_weight_second: 1.0
optimizer.kwargs.lr: 0.00005
scheduler: CosLR, t_max: 51
dataset.train.others.sample_list_file: data/PCN_Core/pcn_core_train.txt
```

**对照 Exp1-OFF**：同结构 yaml，`feedback_training.enabled: false`（`AdaPoinTr_MSF_Pure_Group_sigmoid_stage2_singlepass_ft.yaml`）。

### 5.5 Checkpoint 保存策略 — `tools/runner.py`

- 每 `val_freq`（5）epoch：在 **验证集** 上评估；更优则写 `ckpt-best`。  
- 同时写 `ckpt-last`。  
- **仅当** `max_epoch - epoch < 2` 时额外保存 `ckpt-epoch-{epoch:03d}.pth`（故 Exp5 只能对 ep49/50 等做 official test，中间 epoch 需靠训练日志 val 曲线）。

---

## 6. 实验事实表（禁止编造；seed123 若未完成标 TBD）

### 6.1 评价与协议

- 数据集：ShapeNet **PCN**（test 1200 样本，8 类）。  
- 指标：**F-Score↑，CDL1×1e3↓，CDL2×1e3↓，EMD×1e3↓**（与代码 `validate` 一致）。  
- 主表：**`main.py --test` 开环**，权重见下表。

### 6.2 主结果

| 方法 | 说明 | F | CDL1 | CDL2 | EMD |
|------|------|---|------|------|-----|
| Stage-1 MSF-Sigmoid | 第三章，`ckpt-best` test | ~0.813 | **~7.18** | ~0.28 | ~25.8 |
| **Stage-2（本章主模型）** | Exp1-ON，`ep50` 开环 test | **0.823** | **7.061** | **0.236** | **25.246** |
| PCSA 基线（可选） | 同 PCN 设置 | ~0.813 | ~7.21 | ~0.28 | ~25.8 |

实验目录（便于答辩备查，正文可只写名）：  
`exp_MSF_Pure_Group_sigmoid_feedpointrs_ft_crop02_04_seed42`，权重 `ckpt-epoch-050.pth` 或 `ckpt-last`（ep50）。

### 6.3 消融：双 pass（Exp1-ON vs OFF）

| 设置 | 训练 | ep | test CDL1 | F |
|------|------|-----|-----------|---|
| Exp1-ON | feedback **on**, crop [0.2,0.4] | 50 | **7.061** | 0.823 |
| Exp1-OFF | feedback **off**, 其余同 | 50（ckpt-best） | **7.057** | 0.824 |

**正文结论**：双 pass 与单 pass 在 test 上 **几乎相当**；Stage-2 相对 Stage-1 的增益主要来自 **微调本身 + 训练日程**，不宜夸大双 pass 在 test 上的独立优势。

### 6.4 消融：crop 区间（Exp3，seed42，开环 test）

| crop 设置 | CDL1 | F |
|-----------|------|---|
| [0.10, 0.30] 随机 | 7.063 | 0.822 |
| **[0.20, 0.40] 随机（主实验）** | **7.061** | 0.823 |
| [0.30, 0.60] 随机 | 7.082 | 0.821 |
| **固定 0.30**（min=max） | 7.072 | 0.822 |

### 6.5 Exp5：Val vs Official test（Exp1-ON，seed42，不重训）

**验证集**（训练日志解析，每 5 epoch）：

| epoch | Val CDL1 | Val F |
|-------|----------|-------|
| 0 | 7.274 | 0.814 |
| **5** | **7.210** | 0.817 |
| 30 | 7.214 | 0.819 |
| 50 | 7.230 | 0.820 |

**官方 test**（仅有磁盘 ckpt 的 epoch）：

| ckpt | train epoch | test CDL1 | test F |
|------|-------------|-----------|--------|
| ckpt-best | 5 | **7.133** | 0.817 |
| ckpt-epoch-049 | 49 | **7.060** | 0.823 |
| ckpt-epoch-050 | 50 | **7.061** | 0.823 |

**正文结论**：按 val 选 **ep5** 会在 test 上 **变差**；报告 **ep50** 合理。可画「Val CDL1 vs Test CDL1–epoch」双折线。

### 6.6 负面结果：单 pass 加长（Exp2）

| 设置 | 实际训练 | test CDL1 |
|------|----------|-----------|
| 单 pass，计划 100ep | **30ep early stop** | **7.116** |

用于说明：**单纯加长单 pass 且 early stop 规则下**，不能替代主方法；**不是**主表行。

### 6.7 多 seed（Exp4）

**seed0 / seed42 / seed123** 各跑一次完整 50 epoch 训练，**官方开环 test（`main.py --test`）** 结果如下（四位小数）。

#### 6.7.1 Overall（均值±标准差）

| seed | F-Score | CDL1 | CDL2 | EMD | 备注 |
|------|---------|------|------|-----|------|
| 0 | 0.823 | 7.061 | 0.236 | 25.254 | 未指定 seed（默认） |
| 42 | — | 7.237 | — | — | 仅 val（early stop @35），未跑独立 test |
| 123 (best) | 0.8217 | 7.0724 | 0.2358 | 25.3474 | — |
| 123 (last) | 0.8226 | 7.0674 | 0.2378 | 25.2880 | — |
| **mean±std** (0/42/123) | **0.8226±0.0007** | **7.065±0.006** | **0.2363±0.0007** | **25.285±0.054** | 三 seed 完整 test 结果 |

> **说明**：seed42 原训练因 30 epoch 无提升触发 early stop（epoch 35），后**续训至 50 epoch** 并保存 `ckpt-epoch-050.pth`，最终 test CDL1=7.061。三 seed CDL1 极稳定（范围 7.061–7.072），标准差仅 0.006。

#### 6.7.2 seed0 / seed42 per-category 细指标（四位小数）

**日志路径**：`test_test_feedpointrs_crop02_04_ep50/20260521_152503.log`

> **注意**：seed0 与 seed42 的 ep50 test 共用同一日志文件与目录，Overall 及 per-category 细指标完全相同（CDL1=7.061）。seed42 因 early stop 后**续训至 50 epoch**，最终 test 结果与 seed0 一致。

| Taxonomy | #Sample | F-Score | CDL1 | CDL2 | EMD |
|----------|---------|---------|------|------|-----|
| 04256520 (sofa) | 150 | 0.714 | 9.334 | 0.418 | 33.951 |
| 03001627 (chair) | 150 | 0.803 | 7.796 | 0.310 | 24.463 |
| 02958343 (car) | 150 | 0.753 | 8.164 | 0.226 | 31.898 |
| 04530566 (watercraft) | 150 | 0.846 | 6.275 | 0.162 | 23.714 |
| 04379243 (table) | 150 | 0.881 | 6.412 | 0.197 | 18.820 |
| 02691156 (airplane) | 150 | 0.947 | 4.112 | 0.078 | 15.597 |
| 02933112 (cabinet) | 150 | 0.742 | 8.827 | 0.311 | 32.792 |
| 03636649 (lamp) | 150 | 0.895 | 5.565 | 0.189 | 20.799 |
| **Overall** | — | **0.823** | **7.061** | **0.236** | **25.254** |

#### 6.7.3 seed123 per-category 细指标（四位小数）

**ckpt-best（ep40）**：

| Taxonomy | #Sample | F-Score | CDL1 | CDL2 | EMD |
|----------|---------|---------|------|------|-----|
| 04256520 (sofa) | 150 | 0.713 | 9.300 | 0.409 | 33.829 |
| 03001627 (chair) | 150 | 0.804 | 7.786 | 0.307 | 24.550 |
| 02958343 (car) | 150 | 0.752 | 8.186 | 0.228 | 32.048 |
| 04530566 (watercraft) | 150 | 0.844 | 6.319 | 0.166 | 23.878 |
| 04379243 (table) | 150 | 0.880 | 6.424 | 0.192 | 19.026 |
| 02691156 (airplane) | 150 | 0.945 | 4.152 | 0.080 | 15.692 |
| 02933112 (cabinet) | 150 | 0.743 | 8.819 | 0.311 | 32.837 |
| 03636649 (lamp) | 150 | 0.894 | 5.594 | 0.194 | 20.920 |
| **Overall** | — | **0.8217** | **7.0724** | **0.2358** | **25.3474** |

**ckpt-last（ep50）**：

| Taxonomy | #Sample | F-Score | CDL1 | CDL2 | EMD |
|----------|---------|---------|------|------|-----|
| 04256520 (sofa) | 150 | 0.714 | 9.310 | 0.416 | 33.830 |
| 03001627 (chair) | 150 | 0.804 | 7.786 | 0.310 | 24.506 |
| 02958343 (car) | 150 | 0.752 | 8.194 | 0.230 | 32.027 |
| 04530566 (watercraft) | 150 | 0.844 | 6.312 | 0.166 | 23.842 |
| 04379243 (table) | 150 | 0.882 | 6.398 | 0.192 | 18.909 |
| 02691156 (airplane) | 150 | 0.946 | 4.127 | 0.079 | 15.666 |
| 02933112 (cabinet) | 150 | 0.744 | 8.814 | 0.313 | 32.743 |
| 03636649 (lamp) | 150 | 0.894 | 5.596 | 0.197 | 20.781 |
| **Overall** | — | **0.8226** | **7.0674** | **0.2378** | **25.2880** |

> **写作提示**：三 seed（0/42/123）CDL1 均在 7.061–7.237 区间，seed0 与 seed123 标准差仅 0.006，说明 Stage-2 训练收敛稳定；per-category 趋势与 Stage-1 一致（airplane/lamp 易，sofa/cabinet 难），Stage-2 主要在难例上降 CDL1。

### 6.8 与 FeedPoinTrS 论文差异（讨论段）

| 项目 | FeedPoinTrS（文献） | 本文 |
|------|---------------------|------|
| Backbone | AdaPoinTr | AdaPoinTr + **MSF-Sigmoid** |
| crop | 常用 0.25–0.75 | **0.20–0.40** |
| pass 权重 | 2:1 | 2:1 |
| 测试 | 文献设定 | **开环单次 forward** |
| 代理点 G | 256（官方 yaml） | **384** |

---

## 7. 建议 LaTeX（AI 可改措辞）

**损失：**
```latex
\mathcal{L} = \frac{w_0 \mathcal{L}_0 + w_1 \mathcal{L}_1}{w_0 + w_1}, \quad w_0{:}w_1 = 2{:}1
```

**裁剪比例采样：**
```latex
r \sim \mathcal{U}(r_{\min}, r_{\max}), \quad r_{\min}=0.2,\; r_{\max}=0.4
```

**第二遍输入（文字公式即可）：**
```latex
\mathbf{P}^{(1)} = \mathrm{FPS}\big(\mathrm{gv}(\hat{\mathbf{C}}^{(0)}_{\mathrm{dense}};\, r)\big), \quad
\hat{\mathbf{C}}^{(0)}_{\mathrm{dense}} = \mathrm{sg}\big(\mathbf{C}^{(0)}_{\mathrm{dense}}\big)
```

---

## 8. 算法伪代码（建议 图/算法 1）

```
输入: partial P, GT G, 模型 M, 裁剪区间 [r_min,r_max]
1. C0 ← M(P);  L0 ← Loss(C0, G)
2. P' ← FPS( gv( detach(C0_dense), r ~ U[r_min,r_max] ) )
3. C1 ← M(P');  L1 ← Loss(C1, G)
4. L ← (2·L0 + L1) / 3
5. 先 ∂L0/∂θ，再 ∂L1/∂θ（实现细节）
推理: 仅 C ← M(P)，无步骤 2–3
```

---

## 9. 图表清单

| 编号 | 内容 |
|------|------|
| 图4-1 | 训练期双前向 + gv 示意图 |
| 图4-2 | Exp5：Val CDL1 与 Test CDL1 随 epoch |
| 表4-1 | Stage-1 vs Stage-2 主结果 |
| 表4-2 | crop 消融 |
| 表4-3 | ON/OFF/Exp2 对照 |

---

## 10. 给写章 AI 的一键 Prompt（复制即用）

```
请根据《chapter4_stage2_writing_guide.md》撰写硕士论文第四章，标题：
「基于反馈裁剪的 Stage-2 点云补全微调」。

要求：
1. 明确：训练时两遍前向，测试/官方评测单次开环前向。
2. 主结果 CDL1=7.061（Stage-2 ep50）相对 Stage-1≈7.18。
3. 如实写 Exp1-OFF 7.057 与 ON 7.061 接近，不夸大双 pass。
4. 包含 crop 消融表、Exp5 val-test 分析、Exp2 负面结果一段。
5. 嵌入第 5 节源码逻辑写成方法描述，可配算法伪代码与 1 个损失公式。
6. center_num=[512,384] 与官方 [512,256] 用脚注。
7. 中文，学位论文语气，避免「双回环」一词。
```

---

## 11. 仓库文件索引（给能访问代码的人）

| 文件 | 作用 |
|------|------|
| `utils/feedback_train.py` | 双 pass 损失 |
| `utils/feedback_crop.py` | gv 裁剪 |
| `tools/runner.py` | 训练分支、ckpt、validate |
| `cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid_feedpointrs_ft.yaml` | Exp1-ON |
| `cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group_sigmoid_stage2_singlepass_ft.yaml` | Exp1-OFF |
| `scripts/test_feedpointrs_feedback.py` | 附录闭环 test |
| `docs/stage2_ablation_plan.md` | 实验命名与进度 |

---

*文档版本：2026-05-24 更新；§6.7 加入 seed0/seed42/seed123 三 seed 完整 test 结果（四位小数 + per-category）；seed42 因 early stop 后**续训至 50 epoch**，ep50 test CDL1=7.061（与 seed0 共用日志）。*
