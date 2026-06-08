# Hard FT 排错矩阵（实验编号 ↔ 假设 ↔ 指标）

## 不可破坏的基线

- 配置：`AdaPoinTr_MSF_Pure_Group_sigmoid_hard_ft_cov005.yaml`
- 条件：`ohem.enabled=false`, `cover_critical_ratio=0.0`, `cover_weight=0.05`
- 复现：seed42 rerun test CDL1 **7.122**（2026-05-20）

## 假设编号

| ID | 假设 | 若成立时现象 |
|----|------|----------------|
| H1 | 微调过长，最优在极早 epoch | val CDL1：ep5 最好，ep10 变差（短训）；长训 best 也在 ep5 附近 |
| H2 | L_cover 与全点 CDL1 错位 | 有 cover 时中后期比无 cover 跌更多 |
| H3 | 去掉 cover 后曲线形态不变 | 无 cover 仍 ep5 最好、ep10 反弹；主因不单是 cover |
| H4 | critical top-30% 加重错位 | critical30 val ep0 7.22，后期更差；test 7.13 vs static 7.12 |
| H5 | 动态换表 / OHEM | 已负向，不重复 |

## 本轮短训（max_epoch=10, seed=42, val_freq=5）

| Run | 配置 | val CDL1 (ep0 / ep5 / ep10) | test CDL1 (official) | 结论 |
|-----|------|------------------------------|----------------------|------|
| D1 | `hard_ft_diag_static_ep10.yaml`（=cov005 损失，10ep） | 7.216 / **7.189** / 7.219 | **7.113** (ep5 ckpt) | **支持 H1**；短训 test 优于 50ep 7.122 |
| D2 | `hard_ft_diag_nocover_ep10.yaml`（cover_weight=0） | 7.207 / **7.188** / 7.219 | **7.111** (ep5 ckpt) | **支持 H3**；test 与 D1 差 0.002，cover 对最终 test 可忽略 |

### D1 vs D2 对比（val + official test 1200）

| 指标 | D1（cover=0.05） | D2（cover=0） |
|------|------------------|---------------|
| val ep5 | 7.189 | 7.188 |
| val ep10 | 7.219 | 7.219 |
| **test CDL1** | **7.113** | **7.111** |
| test F | 0.818 | 0.818 |

- test 差 **0.002**（小于 run 间波动），**H2 不成立**：cover 不是瓶颈
- 二者均略优于 cov005 长训 seed42 rerun **7.122**（同 seed、仅 10ep）→ **H1 强化**：不必训满 50ep

### 对照基线（official test CDL1）

| 配置 | test CDL1 |
|------|-----------|
| 论文静态 cov005（历史） | 7.118 |
| cov005 seed42 rerun 50ep | 7.122 |
| D1 短训 10ep + ep5 ckpt | 7.113 |
| D2 短训 10ep + ep5 ckpt | 7.111 |

## 指标读法

| 指标 | 反映问题 |
|------|----------|
| val CDL1 @ ep0/5/10 | H1 选模 epoch |
| train Dense loss 平台 | 非主因；与 val CD 可脱钩 |
| test CDL1 (official 1200) | 最终论文数字 |
| F vs CD 最佳 epoch | 选模指标错位 |

## 推荐动作（不改 cov005 默认行为）

1. 论文主结果仍用静态 cov005 **7.118 / 7.122**
2. 若继续改进：优先 **缩短 hard_ft epoch（5–10）** 或 **val 选 ep5 ckpt**，而非 OHEM/critical/动态表
3. 新功能保持默认关闭，独立 yaml 做实验
