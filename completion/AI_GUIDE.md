# AI 执行指南：AdaPoinTr / PGST 训练启动

> **本文件用途**：给后续 AI 看，减少「改代码 5 分钟、启动折腾 20 分钟」。  
> 用户约束：**不随意删工程文件**（仅可删缓存）；**不干扰正在运行的训练**；服务器常卡顿，**启动后需耐心等 1～3 分钟**才有 Epoch 日志。

---

## 一、为什么启动特别难？（先建立预期）

改 `PGST.py` 只动逻辑；**每次启动都要重新走整条冷启动链**，与改代码耗时无关：

| 阶段 | 典型耗时 | 说明 |
|------|----------|------|
| 指定 `pgst` 环境 + `LD_LIBRARY_PATH` | — | 缺则 `chamfer` 找不到 `libc10.so`，进程挂住或日志长期为空 |
| `import torch` / CUDA 初始化 | 10～30s+ | 共享机 load 高、swap 满时更慢 |
| `import` PGST、建模型、读配置 | 数十秒 | |
| 构建 DataLoader / PCN 索引 | 数十秒～数分钟 | 磁盘缓存存在后，**epoch 内** DataTime 可 ~0.001s |
| **DDP 双卡** | 两个进程各做一遍 | 冷启动时间近似 ×2 |
| `num_workers>0` | 每 rank 再 fork worker | 第一个 batch 的 DataTime 可达 60～90s，**属正常** |

**误判高发**：日志 1～3 分钟无输出 ≠ 没启动；不要用 `timeout 20s` 判断卡死。

**两个日志文件**（正常现象）：
- `nohup` 重定向的 `.log`（stdout/stderr）
- `get_root_logger` 创建的 `experiments/.../YYYYMMDD_HHMMSS.log`

**只看其一**，不要因另一个为空就认为失败。监控方式见 **§1.5**（**禁止叠多个 `tail -f`**）。

---

## 1.5 日志监控：禁止叠多个 `tail -f`（inotify，非常重要）

### 现象

```text
tail: inotify 资源耗尽
tail: 无法使用 inotify 机制，回退为轮询（polling）机制
```

### 根因

- 每个 **`tail -f`** 会对目标文件占一个 **inotify watch**。
- AI / 用户 / 多个 SSH 窗口若**同时**对多个日志 `tail -f`（`train.log`、`logs/...log`、orchestrator、各实验目录……），极易打满系统 `fs.inotify.max_user_watches`，**整机**后续 `tail -f` 都异常。
- 回退为 polling 后仍可用，但会**额外占 CPU**；更严重的是会让人误以为「训练没输出」。

### 硬性规则（AI 必须遵守）

1. **同一时刻最多 1 个** `tail -f`（全用户、全日志合计）。盯训练时**只选一个路径**（推荐 `logs/msf_sigmoid_rebuild_train.log` 或当前实验的 `train.log` 二选一）。
2. **不要用 `tail -f` 做轮询监控**：Agent 反复「sleep + tail」应改用 **`tail -n N`（不带 `-f`）** 或 **`grep Epoch`**，避免每次起新 watch。
3. **不要**在后台为用户长期挂 `tail -f`；需要持续看日志时，告诉用户用下面「推荐命令」自行开一个终端。
4. 若 inotify 已耗尽：让用户 `pkill -f "tail -f"` 关掉多余 tail，或 `watch` / 定期 `cat`，**不要**再叠新的 `tail -f`。

### 推荐命令（替代 `tail -f`）

```bash
# 推荐：每 10 秒刷新最后 30 行，只占一个进程、不爆 inotify
watch -n 10 'tail -30 logs/msf_sigmoid_rebuild_train.log'

# 或偶尔手动看（AI 轮询用这个）
tail -30 /path/to/train.log
grep -E "Epoch|Error|Traceback" /path/to/train.log | tail -5
```

### 与「训练是否在跑」的配合

- **是否训练**：以 `nvidia-smi` 是否有 **python** 且显存 **~7–8GB/卡** 为准，**不要**以 `tail -f` 是否卡住为准。
- 日志长期 0 字节可能是 **仍在 `import torch`**（见 §一），不是 tail 坏了。

---

## 1.6 脚本日志目录：只用 `completion/logs/`，禁止堆 `/tmp`

### 约定

| 类型 | 路径 | 说明 |
|------|------|------|
| **脚本 nohup 日志** | `completion/logs/*.log` | 训练/测试/watch 脚本统一写这里 |
| **PyTorch 文件日志** | `experiments/.../YYYYMMDD_HHMMSS.log` | `get_root_logger` 自动生成 |
| **实验目录 train.log** | `experiments/.../train.log` | 部分 DDP 脚本直接重定向 |

- 所有 `scripts/*.sh` 通过 `source scripts/_logs_dir.sh` 得到 **`LOG_ROOT`**（默认 `completion/logs`），**禁止**再写 `/tmp/xxx.log`。
- 可用环境变量覆盖：`LOG_ROOT=/path/to/logs bash scripts/run_xxx.sh`
- `logs/` 下 `*.log` 已被 `.gitignore` 忽略，不会进 git。

### 脚本模板

```bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$(dirname "$0")/_logs_dir.sh"
LOG="${LOG_ROOT}/stage2_exp1_on_lr1e4_seed42.log"
```

### 清理 `/tmp` 旧日志（AI 在用户要求或迁移后执行）

1. **先查正在跑的进程占用的日志**（这些不能删）：

```bash
pgrep -af 'main.py|torch.distributed.run'
lsof -p <pid> 2>/dev/null | grep '/tmp/.*\.log'
```

2. **只删本项目相关、且未被进程打开的文件**（示例）：

```bash
# 保留正在写的唯一文件，例如：
KEEP="/tmp/stage2_exp1_on_lr1e4_seed42.log"
for f in /tmp/stage2_*.log /tmp/test_stage2_*.log /tmp/feedpointrs*.log /tmp/msf_*.log; do
  [[ -f "$f" && "$f" != "$KEEP" ]] && rm -f "$f"
done
```

3. 历史内容已复制到 `completion/logs/` 的，可安全删 `/tmp` 副本。
4. **不要**删除 `/tmp` 下与 completion 无关的系统/其他用户文件。

### 监控命令（相对 completion 目录）

```bash
watch -n 10 'tail -30 logs/stage2_exp1_on_lr1e4_seed42.log'
grep -E 'Epoch|Validation|Early Stop|Overall' logs/your_exp.log | tail -5
```

---

## 二、AI 启动训练标准流程（按顺序做）

### 步骤 0：工作目录与环境

```text
工作目录: /home/fubenhao/data/fubenhao_data/PointGST-main_pure/completion
Python:   /home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python
```

**必须**设置（每次启动前）：

```bash
export LD_LIBRARY_PATH=/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/lib/python3.9/site-packages/torch/lib:$LD_LIBRARY_PATH
```

### 步骤 1：检查是否已有训练在跑

```bash
ps aux | grep "main.py" | grep -v grep
nvidia-smi
```

- 若用户未要求停止，**不要** `pkill` 正在跑的 `main.py`。
- 若需新实验且 GPU 被占用，先与用户确认。

### 步骤 2：选择启动模式

| 场景 | 用法 | 说明 |
|------|------|------|
| **推荐：从头/预训练微调** | 不加 `--resume` | 自动加载 yaml 里 `pretrained_ckpt: ckpt/AdaPoinTr_ps55.pth` |
| 同实验续训 | 加 `--resume` | 只读 `exp_xxx/ckpt-last.pth`；**文件损坏会秒崩** |
| 单卡 | `CUDA_VISIBLE_DEVICES=1` + `--launcher none`（默认） | 内部 `DataParallel`，但只见 1 卡 |
| **推荐：双卡** | `CUDA_VISIBLE_DEVICES=0,1` + DDP（见下） | 每卡 batch = total_bs / 2 |

**不要轻易 `--resume`**：用户强杀训练时 `ckpt-last.pth` 可能写一半（`PytorchStreamReader failed`）。  
续训前应用 python 验证：

```bash
/home/fubenhao/data/.../pgst/bin/python -c "import torch; torch.load('experiments/.../ckpt-last.pth', map_location='cpu'); print('OK')"
```

损坏则用 `ckpt-best.pth` 覆盖 `ckpt-last.pth`，或**去掉 `--resume` 从 ps55 重训**。

### 步骤 3：必选命令行参数（缺一常失败）

| 参数 | 必须？ | 说明 |
|------|--------|------|
| `--model pgst` | **是** | 否则 `main.py` 把 NAME 改成 `AdaPoinTr` → `KeyError` |
| `--config cfgs/PCN_models/<xxx>.yaml` | 是 | |
| `--exp_name <name>` | 是 | 决定 `experiments/.../exp_name/` |
| `--num_workers N` | 建议 4（DDP） | 0 更稳但 DataTime 慢；测试曾用 0 防卡死 |
| `--launcher pytorch` | 双卡必须 | 单卡用默认 `none` |
| `--resume` | 仅续训 | 见上 |
| `--val_freq` | **不必传** | `runner.py` 已写死 `val_freq = 10` |
| `--gpu` | **不存在** | 用 `CUDA_VISIBLE_DEVICES` |

yaml 里的 `val_freq: 10` **不会生效**（只认 `args.val_freq`，已由代码写死覆盖）。

### 步骤 4：推荐启动命令（双卡 DDP + 从 ps55 预训练，当前 MSF_pure_Group）

```bash
cd /home/fubenhao/data/fubenhao_data/PointGST-main_pure/completion

export CUDA_VISIBLE_DEVICES=0,1
export LD_LIBRARY_PATH=/home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/lib/python3.9/site-packages/torch/lib:$LD_LIBRARY_PATH

nohup /home/fubenhao/data/fubenhao_data/miniforge3/envs/pgst/bin/python -u -m torch.distributed.run \
  --nproc_per_node=2 --master_port=29506 \
  main.py --launcher pytorch \
  --config cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group.yaml \
  --exp_name exp_MSF_Pure_Group \
  --num_workers 4 \
  --model pgst \
  > experiments/AdaPoinTr_MSF_Pure_Group/PCN_models/exp_MSF_Pure_Group/train.log 2>&1 &
```

- **不要加 `--resume`** 除非已确认 `ckpt-last.pth` 可读。
- `master_port` 冲突时改成 29507、29508 等。
- `total_bs: 32` → 每卡 batch 16。

### 步骤 5：启动后如何确认成功（等 1～3 分钟）

```bash
# 进程
ps aux | grep "main.py" | grep -v grep

# 双卡显存（应有约 7～8GB/卡）
nvidia-smi

# 日志：用 tail -n 或 watch（勿叠多个 tail -f，见 §1.5）
tail -30 experiments/AdaPoinTr_MSF_Pure_Group/PCN_models/exp_MSF_Pure_Group/train.log
# 用户若要持续刷新，只开一个终端：
# watch -n 10 'tail -30 experiments/.../train.log'
```

日志中应出现：

```text
Loading weights from ckpt/AdaPoinTr_ps55.pth...   # 未 --resume 时
args.resume : False
args.distributed : True
args.num_workers : 4
encoder_config.adapter_mode : msf_pure_group
[Epoch 0/150][Batch 1/156] ...
```

验证轮结束后（每 10 epoch）会有 `[Gate Monitor]`（`MSF_pure_Group.flush_gate_stats` 已在 `runner.py` validate 中调用）。

### 步骤 6：单卡备用模板（仅 GPU1）

```bash
export CUDA_VISIBLE_DEVICES=1
export LD_LIBRARY_PATH=.../torch/lib:$LD_LIBRARY_PATH

nohup .../pgst/bin/python -u main.py \
  --config cfgs/PCN_models/AdaPoinTr_MSF_Pure_Group.yaml \
  --exp_name exp_MSF_Pure_Group \
  --num_workers 0 \
  --model pgst \
  > experiments/.../train_single.log 2>&1 &
```

---

## 三、启动失败速查表

| 现象 | 常见原因 | 处理 |
|------|----------|------|
| 日志长期 0 字节 | 缺 `LD_LIBRARY_PATH`；或仍在 import | 加上环境变量；等 2～3 分钟 |
| `ModuleNotFoundError: torch` | 用了系统 `python` | 用 `pgst/bin/python` 全路径 |
| `AdaPoinTr is not in the models registry` | 未加 `--model pgst` | 加上 |
| `unrecognized arguments: --gpu` | 无此参数 | 改 `CUDA_VISIBLE_DEVICES` |
| DDP `AssertionError` local_rank | 已修：`main.py` 读 `LOCAL_RANK` | 确保代码未回退 |
| DDP `invalid device ordinal` | 只可见 1 卡或 `dist_utils` 未用 LOCAL_RANK | `CUDA_VISIBLE_DEVICES=0,1` |
| `PytorchStreamReader failed` | `ckpt-last.pth` 损坏 | 去掉 `--resume` 或换 `ckpt-best.pth` |
| 每轮都验证 | 旧版只认 `args.val_freq` | 确认 `runner.py` 内 `val_freq = 10` |
| 配置写了 val_freq 无效 | yaml 不生效 | 以代码为准 |
| `tail: inotify 资源耗尽` | 多个 `tail -f` 同时盯着多份日志 | **关掉多余 `tail -f`**；改用 `watch -n 10 'tail -30 ...'` 或单次 `tail -n`（§1.5） |
| `nvidia-smi` 无 python、log 空 | 进程卡在 `import torch` 僵尸占坑 | `pkill` 后 `bash scripts/run_rebuild_dual_gpu.sh`；勿叠 tail 误判（§1.5） |

---

## 四、代码侧已修复项（AI 勿重复改坏）

1. `main.py`：`LOCAL_RANK` 从环境变量覆盖 `args.local_rank`。
2. `dist_utils.py`：`torch.cuda.set_device(int(os.environ['LOCAL_RANK']))`。
3. `runner.py`：训练循环内 **`val_freq = 10`**（写死，勿依赖命令行）。
4. `runner.py`：`validate()` 末尾调用 `MSF_pure_Group` / `MSF_pure_Group_tanh` / `MSF_pure_Group_sigmoid` 的 `flush_gate_stats`。
5. `MSF_pure_Group` 系列：基类 `_MSF_pure_GroupBase` + 子类 `_spatial_gates`（softmax / tanh / sigmoid）。

### 门控消融（第一步）

| 变体 | adapter_mode | config | exp_name |
|------|--------------|--------|----------|
| A Softmax | `msf_pure_group` | `AdaPoinTr_MSF_Pure_Group.yaml` | `exp_MSF_Pure_Group` |
| B Tanh | `msf_pure_group_tanh` | `AdaPoinTr_MSF_Pure_Group_tanh.yaml` | `exp_MSF_Pure_Group_tanh` |
| C Sigmoid | `msf_pure_group_sigmoid` | `AdaPoinTr_MSF_Pure_Group_sigmoid.yaml` | `exp_MSF_Pure_Group_sigmoid` |

A 可与正在跑的 Softmax 实验复用；**B/C 须在 A 释放双卡后**再启动（勿与 A 同时占 GPU）：

**推荐（夜间自动：等 A 结束 → B → C → 测 A/B/C）**

用 **nohup**（已采用），SSH 断开不会杀进程；需要随时 attach 可用 `screen -S ablation` 再跑同样命令。

```bash
cd /home/fubenhao/data/fubenhao_data/PointGST-main_pure/completion
nohup bash scripts/run_msf_ablation_overnight.sh >> experiments/overnight_orchestrator.log 2>&1 &
# 只看编排日志时开一个 watch 即可，勿同时对多文件 tail -f（§1.5）
watch -n 30 'tail -20 experiments/overnight_orchestrator.log'
```

- B/C 训练目录：`exp_MSF_Pure_Group_tanh` / `exp_MSF_Pure_Group_sigmoid`（**不要**用 `exp_MSF_Pure_Group`，否则会覆盖 A 的 ckpt）。
- 评测日志：`experiments/MSF_ablation_eval/{A-softmax,B-tanh,C-sigmoid}/test.log`；A 的权重**只读** `exp_MSF_Pure_Group/ckpt-best.pth`。
- 双卡编排默认 `export CUDA_VISIBLE_DEVICES=0,1`（变量名 `MSF_DDP_GPUS`）。若用带 `CUDA_VISIBLE_DEVICES=0` 的 nohup 启动父 shell，**不要**继承该值，否则 B/C 会 `invalid device ordinal`。
- **启动失败可自动重试**：`run_msf_ablation_overnight.sh` 对 B/C 会在日志出现典型崩溃（如 `invalid device ordinal`、`ChildFailedError`、端口占用等）或超时后，退避再启，默认最多 `MAX_TRAIN_START_ATTEMPTS=5` 次；`STARTUP_MAX_WAIT=1800`（30 分钟）等首条 Epoch。B 失败仍会尝试 C；缺 ckpt 的测试会 SKIP。
- **评测交错并行**：A/B/C 测试后台跑；日志出现 `Test[200/`（`TEST_PROGRESS_RE`）后启动下一个；默认 GPU `MSF_TEST_GPUS=0,1,0`；全部结束后 `wait` 收束。

轮询默认 30 分钟（`POLL_TRAIN_SEC=1800`）；启动后等 Epoch 行每 2 分钟（`POLL_STARTUP_SEC=120`）。

**手动**

```bash
bash scripts/train_msf_ablation_bc.sh tanh      # B，占满双卡
bash scripts/train_msf_ablation_bc.sh sigmoid   # C（须 B 结束）
```

日志：`experiments/AdaPoinTr_MSF_Pure_Group_tanh/.../train.log` 与 `..._sigmoid/.../train.log`。Gate Monitor 标签为 `[tanh]` / `[sigmoid]`。

---

## 五、历史排障记录（备查）

### 2026-05-14：`chamfer` / `libc10.so`

- 根因：`import chamfer` 找不到 `libc10.so`，在写日志前阻塞。
- 修复：`LD_LIBRARY_PATH` 指向 `.../envs/pgst/.../torch/lib`。

### 2026-05-15：`MSF_pure_Group` 与双卡

- 非挖矿；慢因 load/swap/多用户 CPU。
- DDP 需 `CUDA_VISIBLE_DEVICES=0,1` + 上述 LOCAL_RANK 修复。
- 强杀导致 `ckpt-last` 损坏 → **默认从 `AdaPoinTr_ps55.pth` 重训更省心**。
- `num_workers=4` + DDP 可用；首 batch DataTime 大属正常。

### 2026-05-17：`tail -f` / inotify 与 F2 rebuild 启动

- 多窗口 + AI 反复 `tail -f` → **`inotify 资源耗尽`**；见 **§1.5**。
- 旧进程卡在 `import torch` 时 log 空、`nvidia-smi` 无 python → 非训练失败；用 `scripts/run_rebuild_dual_gpu.sh`（先打 log 再训，DataParallel 双卡）。

### F2：MSF rebuild 注入（Phase 1）

```bash
nohup bash scripts/run_rebuild_dual_gpu.sh >/dev/null 2>&1 &
watch -n 10 'tail -30 logs/msf_sigmoid_rebuild_train.log'   # 勿叠 tail -f
```

- 配置：`AdaPoinTr_MSF_Pure_Group_sigmoid_rebuild.yaml`（`msf_route_mode: rebuild`）。
- Phase 0 回退：`AdaPoinTr_MSF_Pure_Group_sigmoid.yaml`（`msf_route_mode: none`）。

---

## 六、给 AI 的一句话原则

1. **先凑齐环境变量和 `--model pgst`，再 nohup。**  
2. **默认双卡 DDP + ps55 预训练，不要 `--resume`。**  
3. **启动后等日志，不要 30 秒就判定失败。**  
4. **监控日志：最多 1 个 `tail -f`；AI 用 `tail -n` / `watch`；以 `nvidia-smi` 判断是否在训（§1.5）。**  
5. **停训前勿强杀正在 save checkpoint 的进程。**

---

## 七、性能 Profiling 与部分参数微调（补充，2026-06）

> **用途**：解释「只训 MSF / 只训 block7 / decoder 全冻」对 **耗时** 与 **精度** 各自意味着什么；避免把「少解冻层」误当成「训练加速」。

### 7.1 脚本与输出路径

工作目录仍为 `completion/`：

```bash
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=.../envs/pgst/.../torch/lib:$LD_LIBRARY_PATH

# 完整 train step：forward + loss + backward + AdamW
python scripts/profile_train_step.py \
  --config cfgs/PCN_models/AdaPoinTr_MSF_single_MSF_trial.yaml \
  --ckpt ckpt/AdaPoinTr_ps55.pth \
  --tag single_MSF_trial \
  --batch-size 8

# 仅 forward + loss（无 backward / optimizer）
python scripts/profile_train_step.py \
  --config cfgs/PCN_models/AdaPoinTr_MSF_single_MSF_trial.yaml \
  --ckpt ckpt/AdaPoinTr_ps55.pth \
  --tag single_MSF_trial_fwd \
  --forward-only \
  --batch-size 8
```

| 输出 | 路径 |
|------|------|
| 文本摘要 | `logs/complete/profile_<tag>.txt` |
| Chrome trace | `logs/complete/profile_<tag>.chrome.json` |

脚本会打 **record_function** 标签：`forward_total`、`loss_total`、`backward_total`、`optimizer_step`，以及 `encoder.block{i}` / `decoder.block{i}` 逐层 forward 耗时。

### 7.2 Forward-only 实测（single_MSF_trial，bs=8，双 step 平均）

| 模块 | CUDA 耗时 | 占 forward 比例 |
|------|-----------|-----------------|
| decoder 8 层合计 | ~96 ms | **~59%** |
| encoder + 6×MSF | ~29 ms | ~18% |
| grouper + 排序/FPS | ~28 ms | ~17% |
| coarse/query 栈 | ~4 ms | ~3% |
| 重建头 reduce/decode | ~2 ms | ~1% |
| Chamfer loss | ~23 ms | （`loss_total` 阶段）|
| **forward + loss** | **~188 ms/step** | — |

Decoder 内：**block0（graph-attn）最重 ~25 ms**，block5 次之 ~20 ms；其余层各 ~6–14 ms。

**Full step 对比**（`single_MSF_trial` vs 全量 `gft`，同 bs=8）：每 step ~250 ms 量级，**epoch 时间几乎相同**（~266 s vs ~287 s）。少解冻层 **不显著省 wall-clock**。

### 7.3 「Decoder 全冻」到底改变什么？

| 维度 | 全冻 decoder 权重时 | 说明 |
|------|---------------------|------|
| **Forward** | **不变** | query 仍须过 8 层 decoder 才得到 `q`；冻权重 ≠ 跳过 matmul |
| **Backward** | **几乎不变** | MSF / 重建头要训时，梯度仍须 `loss → pred → q → mem → encoder(MSF)` 回传；frozen 层不算 param grad，但 **对输入的 backward GEMM 仍在** |
| **Optimizer** | **略减** | 不更新 decoder 参数；AdamW state 更小，但占比可忽略 |
| **表达能力** | **可能明显变弱** | query 表征锁死在 ps55；仅靠 MSF 改 `mem` + 重建头，通常弱于「MSF + 末层 decoder + head」 |

可选 freeze 策略（`tools/freeze_policy.py` + `tools/builder.py`）：

| `optimizer.part` | 可训范围 |
|------------------|----------|
| `gft` | MSF + decoder 全层 + coarse/query + 重建头（默认全量微调） |
| `gft_single_decoder` | 6×MSF + **指定** `trainable_decoder_block`（默认 7）+ 重建头 |
| `gft_msf_head_only` | 6×MSF + 重建头，**零 decoder block** |

配置示例：`cfgs/PCN_models/AdaPoinTr_MSF_single_MSF_trial.yaml`（`gft_single_decoder` + `trainable_decoder_block: 7`）。

### 7.4 部分参数微调的意义（AI 回答用户时的框架）

**结论先行**：在本仓库的 AdaPoinTr + MSF 补全管线里，部分微调的主价值是 **精度 / 样本效率 / 存储**，**不是** 单 step 训练加速。

#### 有意义的部分（补全领域内）

1. **精度**：Stage-1 已验证——在 ps55 上只训 MSF（~13% 参数）即可把 PCN CD 从 init ~8.8 拉到表 3.10 量级（MSF-Sigmoid CDL2≈0.210，PCSA≈0.212；相对 ps55 的 CDL2≈0.327 是实质性提升）。`single_MSF_trial` ep0 val CDL1 8.712 → ep10 7.907 亦在收敛。
2. **Stage-2 策略**：FeedPoinTrS 双 pass 改的是 **训练分布**（反馈裁剪），与「训多少层」正交；Stage-2 在 Stage-1 上 CDL1 可进一步到 ~6.55 量级。
3. **Decoder 末层**：profiler 证明 block7 **不省时间**，但若 block7 可训，仍可能带来 **query 表征** 的额外自由度（精度 ablation 待 `gft_msf_head_only` vs `gft_single_decoder` 对比）。
4. **工程侧**：可只存/发 **4.44M adapter+head** 而非 32M+ 全模型；多任务可挂不同 MSF adapter；optimizer 状态更小（仍非瓶颈）。

#### 意义有限的部分

1. **Wall-clock 训练**：forward 占 ~65%+，decoder forward 必跑；backward 穿过 frozen decoder 仍贵 → **解冻 13% vs 100% 参数，epoch 时间几乎一样**。
2. **推理延迟**：微调策略不改变推理图；推理仍是完整 8 层 decoder。
3. **「只有补全才有意义」**：本 repo 的 MSF 嵌在 **completion encoder** 上，实验与 loss 均围绕 Chamfer 补全；**未验证** 检测/分割等下游。谈「跨任务迁移」需另做实验，不能从 profiler 推出。

#### 一句话给论文 / 讨论

> 部分参数微调是一种 **在强预训练补全骨干上、用少量谱域 adapter 与可选末层 decoder 做任务适配** 的手段；其收益体现在 **CD/EMD 与 checkpoint 体积**，而非训练或推理的算力节省。若目标是 **加速**，应改结构（蒸馏、剪层、早退）或 batch/IO，而不是指望 `requires_grad=False`。

### 7.5 给 AI 的操作提示

- 用户问「冻 decoder 能快多少」→ 先跑 `--forward-only`，引用 §7.2–7.3，强调 **forward 省不下来**。
- 用户问「少训几层有啥用」→ 区分 **精度 ablation** vs **速度**；速度看 §7.2 full step 对比。
- 要做 `gft_msf_head_only` ablation → 复制 `AdaPoinTr_MSF_single_MSF_trial.yaml`，改 `optimizer.part: gft_msf_head_only`，删 `trainable_decoder_block`。

### 7.6 文献与「部分微调 vs 全量」——何时有意义？（2026-06 补充）

> **结论先行**：部分微调 **不是没用**，但 **价值主张因任务而异**。在 **补全（生成）** 上，纯 adapter 往往 **略低于** 解冻 decoder/全量；在 **分类/检测** 上，3D PEFT 文献多次报告 **用 <1%～3% 参数接近或超过 FFT**。

#### 7.6.1 本仓库 seed42 官方 test（PCN，1200 test，ep150 best 量级）

| 设置 | 可训参数占比 | CDL1 | CDL2 | 备注 |
|------|-------------|------|------|------|
| ps55 初始化 | 0%（仅加载） | 8.808 | 0.327 | 未训 |
| MSF-Sigmoid + `gft` | **~65%**（MSF+decoder+head，encoder 主干冻） | **6.654** | **0.210** | Stage-1 主结果 |
| PCSA + `gft` | ~65% | 6.715 | 0.212 | Stage-1 对照 |
| MSF + FeedPoinTrS Stage-2 | ~65% | **6.568** | 0.208 | 训练策略增益，非 PEFT 参数量 |
| `single_MSF_trial` | **~13.5%** | 训练中（ep10 val 7.907） | — | 预期低于 65% gft |
| AdaPoinTr 官方 PCN（文献） | 100% 量级 | **~6.53** | — | [AdaPoinTr TPAMI](https://arxiv.org/abs/2301.04545) |

**要点**：你现在的 `optimizer.part: gft` **不是** PointGST 论文里的「0.6M 纯 adapter」；它仍训 **8 层 decoder + coarse/query + 重建头**，已是 **任务导向的部分全量**。再砍到 13%（single_MSF_trial）几乎必然掉点——这与文献一致，不是 MSF 方向「失败」。

#### 7.6.2 PointGST 原文 Table 8（PCN 补全，adapter-only）

| 方法 | 可训 Params | Avg CD-L1 ↓ | F@1% ↑ |
|------|------------|------------|--------|
| AdaPoinTr baseline（全量微调对照） | 17.1M | **6.45** | **0.844** |
| PointGST / PCSA（纯 adapter） | **0.6M** | 6.64 | 0.836 |
| Point-PEFT | 0.6M | 6.74 | 0.828 |

来源：[PointGST TPAMI / arXiv:2410.08114](https://arxiv.org/abs/2410.08114) §6.6。**补全任务上 PEFT 略逊于 FFT ~0.15–0.3 CD-L1**，但用 **~1/30 可训参数** 达到接近性能。论文「超越 FFT」的主证据在 **ScanObjectNN 分类**（0.67% 参数），不是 PCN 补全。

#### 7.6.3 其它 3D PEFT 文献（简要）

| 工作 | 会议 | 核心 | 与 FFT 关系 |
|------|------|------|------------|
| [Point-PEFT](https://arxiv.org/abs/2310.03059) | AAAI'24 | Point-prior Prompt + Geometry-aware Adapter | 分类上 **优于** FFT；3D 专用设计 |
| [PointLoRA](https://arxiv.org/abs/2504.16023) | CVPR'25 | LoRA + multi-scale token selection | **~3.43%** 参数，多数据集 competitive |
| [PointGST](https://arxiv.org/abs/2410.08114) | TPAMI | 谱域 PCSA | 分类 **超** FFT；补全 **近** FFT 但略低 |
| PEFT Survey | [arXiv:2403.14608](https://arxiv.org/pdf/2403.14608) | 综述 | PEFT 省 **存储/多任务 adapter**；训练内存仍常需 backward 穿 frozen 层 |
| Look Within or Look Beyond | [arXiv:2505.22355](https://arxiv.org/html/2505.22355) | 理论 | **PEFT ⊂ FFT**；复杂生成/推理任务 FFT 上限更高 |

#### 7.6.4 方向有没有用？——换一个问题

不要问「13% 能否打败 100%」（多数情况不能），应问：

1. **任务是什么？**  
   - **补全/生成**：decoder + 重建头对 CD 敏感；纯 adapter 有 **~0.2 CD 级 gap**（PointGST Tab.8 + 你的 gft 实验）。  
   - **分类/检测**：谱域/几何 adapter 更有故事，文献支持 **参数少、精度够**。

2. **你要优化什么资源？**  
   - **Wall-clock**：在本 AdaPoinTr 实现里 **几乎无效**（§7.2–7.3）。  
   - **Checkpoint / 多数据集 adapter / 防遗忘**：PEFT 仍成立。  
   - **精度 SOTA**：Stage-2 训练策略（FeedPoinTrS）+ 合理解冻（gft 65%）比再砍参数更划算。

3. **你的 MSF 相对 PCSA 的意义**（同 65% gft）：CDL1 **6.654 vs 6.715**、EMD **24.32 vs 24.59**——**同参数量下 adapter 设计仍有增益**，这不是「部分微调没意义」，而是 **「adapter 结构创新」** 的意义。

4. **论文怎么写才诚实**  
   - _claim_：谱域/group MSF 在 **参数效率–精度** 曲线上优于 PCSA / 空间 PEFT（Pareto 更好）。  
   - _不 claim_：13% 参数打败全量 gft 或 AdaPoinTr 官方 6.53（除非有实验支撑）。  
   - _可 claim_：Stage-2 反馈裁剪与 Stage-1 adapter **正交**，合计 CDL1 ~6.55 接近全量微调区间。

#### 7.6.5 建议实验（若需补强「方向有用」）

| 实验 | 目的 |
|------|------|
| `gft` vs `gft_msf_head_only` vs `gft_single_decoder` | 精度–参数量 Pareto 曲线 |
| 同 epoch 预算下 MSF vs PCSA（已有 seed42） | adapter 结构对比 |
| 与 PointGST Tab.8 对齐：仅 adapter 0.6M 配置 | 和原文公平对比 |
| Stage-2 on/off（同 Stage-1 ckpt） | 分离「参数」与「训练策略」贡献 |
