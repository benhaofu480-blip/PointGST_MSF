# Stage-2 补充实验计划

> 主线结论：Stage-1 MSF-Sigmoid（test CDL1=7.180）→ Stage-2 FeedPoinTrS 式双 pass 微调 ep50（test CDL1=7.061）。  
> 本章最终权重：**Stage-2 ep50**。Official test 均为 **开环单次前向**。

## 统一约定

| 项 | 约定 |
|----|------|
| 起点 | Stage-1 `exp_MSF_Pure_Group_sigmoid/ckpt-best.pth`（除非 Exp8 另有说明） |
| 训练集 | `data/PCN_Core/pcn_core_train.txt` |
| 默认 lr | 5e-5，AdamW，CosLR t_max=51 |
| 默认 epoch | max 50，val_freq=5，early_stop patience=30 |
| 默认 seed | 42（Exp4 扩展多种子） |
| Official test | `main.py --test`，开环；主表取 **ep50**（Exp5 解释 val-best 与 ep50 差异） |
| GPU | 训练 GPU1；测试 GPU0→GPU1 并行（best 到 200/1200 启 last） |

---

## Exp1 Feedback 开关（进行中）

**目的**：证明增益来自双 pass，而非「同设定下多微调 50 epoch」。

| 组 | feedback | 配置 | 实验名 | 状态 |
|----|----------|------|--------|------|
| ON | enabled | `AdaPoinTr_MSF_Pure_Group_sigmoid_feedpointrs_ft.yaml` | `exp_MSF_Pure_Group_sigmoid_feedpointrs_ft_crop02_04_seed42` | ✅ 已完成（ep50 test 7.061） |
| OFF | disabled | `AdaPoinTr_MSF_Pure_Group_sigmoid_stage2_singlepass_ft.yaml` | `exp_stage2_exp1_feedback_off_seed42` | 🔄 训练中 |

**脚本**

```bash
# OFF
GPU=1 bash scripts/run_stage2_exp1_feedback_off.sh

# ON（已跑过，复用结果；如需严格重跑）
GPU=1 bash scripts/run_feedpointrs_ft_seed42.sh
```

**日志**

- ON train: `logs/feedpointrs_ft_crop02_04_seed42.log`
- OFF train: `logs/stage2_exp1_feedback_off_seed42.log`
- test: `logs/test_stage2_exp1_{on,off}_{best,last,ep50}.log`

**判据**：OFF 的 test CDL1 应显著高于 ON（期望 OFF ≈ 7.12–7.15，ON ≈ 7.06）。

---

## Exp2 算力公平：单 pass 100 epoch

**目的**：回应「双 pass 50ep 不如单 pass 多训一倍」。

| 组 | 训练 | epoch |
|----|------|-------|
| A | 双 pass（Exp1-ON） | 50 |
| B | 单 pass | **100** |

**配置**：`AdaPoinTr_MSF_Pure_Group_sigmoid_stage2_singlepass_100ep.yaml`（crop 0.2–0.4 同 ON，`feedback.enabled=false`，`max_epoch:100`，`t_max:101`，lr=5e-5）。

**实验名**：`exp_stage2_exp2_singlepass_100ep_seed42`

**启动**：`GPU=1 bash scripts/run_stage2_exp2_singlepass_100ep.sh`

**判据**：B 的 test CDL1 仍不如 A（ep50），则双 pass 非单纯算力等价替换。

---

## Exp3 Crop 比例消融

**目的**：说明 0.2–0.4 相对原论文 0.25–0.75 的选取合理。

| 组 | crop_ratio_min | crop_ratio_max | 状态 |
|----|----------------|----------------|------|
| A（现用） | 0.20 | 0.40 | ✅ Stage-2 主实验 (test ep50 **7.061**) |
| B | 0.30 | 0.60 | ✅ best ep5 **7.152** / last ep35 **7.082** |
| C | 0.10 | 0.30 | 🔄 双卡训练中 |
| D（论文） | 0.25 | 0.75 | 待做 |
| E（固定） | 0.30 | 0.30 | 🔄 `exp_stage2_exp3_crop030_fixed_seed42` |

**配置**

- B: `AdaPoinTr_MSF_Pure_Group_sigmoid_feedpointrs_ft_crop030_060.yaml`
- C/D: 待复制 yaml

**实验名**：`exp_stage2_exp3_crop030_060_seed42`（B）

**判据**：A 或 C 在 test 上不差于 B；写入方法节「微调阶段 crop 区间适配」。

---

## Exp4 多 seed 稳定性

**目的**：7.061 非单 seed 偶然。

| seed | 实验名后缀 |
|------|------------|
| 42 | 已有（Exp1-ON） |
| 123 | `_seed123` |
| 456 | `_seed456` |

**每组**：完整 Stage-2 双 pass 50ep → official test ep50。

**汇报**：test CDL1 **均值 ± 标准差**（F/CDL2/EMD 可选）。

---

## Exp5 Val–Test 错位 / checkpoint 选择

**目的**：解释为何取 ep50 而非 val-best；量化 val 与 official test 差异。

**做法**（不重训）：
- **Val 曲线**：从训练日志解析 ep0,5,…,50 → `python scripts/parse_exp5_val_from_log.py`
- **Official test**：仅磁盘存在的权重（`runner` 只在末 2 epoch 存 `ckpt-epoch-*`）  
  `ckpt-best`≈ep5、`ckpt-epoch-049`、`ckpt-epoch-050`

**脚本**：`bash scripts/run_stage2_exp5_epoch_sweep_tests.sh` → `logs/exp5_epoch_sweep_summary.txt`

**已知**：val-best **ep5 CDL1=7.210**；ep50 val **7.230**；test ep50 **7.061**（已有）

## Exp4 多 seed

| seed | 实验名 | 状态 |
|------|--------|------|
| 42 | `exp_..._crop02_04_seed42` | ✅ |
| 123 | `exp_..._crop02_04_seed123` | 🔄 `scripts/run_feedpointrs_ft_seed123.sh` |

---

## Exp6 PCN 八类 per-category 分解

**目的**：解释 Stage-2 在哪些类别真正变好。

**对比对**

- Stage-1 sigmoid ckpt-best（或 test 7.180 那次权重）
- Stage-2 ep50（7.061）

**产物**：8 类 CDL1 表 + Δ 柱状图 + 2–3 个典型可视化样例。

**脚本**：从 official test 日志解析 per-taxonomy，或单独跑带 taxonomy 汇总的 test。

---

## 可选扩展（时间允许）

| ID | 内容 | 优先级 |
|----|------|--------|
| Exp7 | pass 权重 2:1 vs 1:1 | 中 |
| Exp8 | 起点消融（sigmoid best / last / ps55） | 中 |
| Exp9 | ep50 开环 vs 第二 pass C1（`test_feedpointrs_feedback.py`） | 低 |
| Exp10 | 无 MSF adapter + 同 feedback 微调 | 低（成本高） |

---

## 论文章节映射

| 实验 | 章节 |
|------|------|
| Exp1–2 | 4.x 有效性 + 归因 |
| Exp3–4 | 4.x 消融与稳定性 |
| Exp5 | 4.x 训练策略 / 4.x 讨论 |
| Exp6 | 4.x 定性分析 |

---

## 进度追踪

- [x] Exp1-ON（Stage-2 主实验，ep50 test 7.061）
- [ ] Exp1-OFF 训练 + 双权重 official test
- [ ] Exp2–Exp6
- [ ] 汇总表写入 `docs/stage2_ablation_results.md`（待实验完成后）
