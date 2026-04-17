#!/usr/bin/env python
"""分析 adaptive pooling gate_net 的权重，判断训练是否有效"""
import sys
sys.path.insert(0, '/home/fubenhao/data/fubenhao_data/PointGST-main_pure/completion')

import torch
import numpy as np
from utils.config import cfg_from_yaml_file
from tools import builder

def main():
    config = cfg_from_yaml_file('cfgs/PCN_models/AdaPoinTr_core_new.yaml')
    config.model.NAME = "AdaPoinTr_PGST"
    model = builder.build_model_from_cfg(config.model)

    # 加载 ckpt-best
    ckpt_path = 'experiments/AdaPoinTr_core_new/PCN_models/exp2_adaptive_pooling/ckpt-best.pth'
    builder.load_model(model, ckpt_path)
    model.cuda()
    print(f"Loaded checkpoint: {ckpt_path}")
    print()

    # ========== 1. 静态权重分析 ==========
    print("=" * 70)
    print("【1】gate_net 静态权重分析（网络参数本身）")
    print("=" * 70)

    gate = model.base_model.gate_net  # nn.Sequential: Linear(1024,128) -> GELU -> Linear(128,1024)

    W1 = gate[0].weight.data   # (128, 1024)
    b1 = gate[0].bias.data     # (128,)
    W2 = gate[2].weight.data   # (1024, 128)
    b2 = gate[2].bias.data     # (1024,)

    print(f"  Layer1 (1024->128):")
    print(f"    W1 shape={W1.shape}, mean={W1.mean():.6f}, std={W1.std():.6f}")
    print(f"    b1 mean={b1.mean():.6f}, std={b1.std():.6f}")

    print(f"\n  Layer2 (128->1024): [零初始化层]")
    print(f"    W2 shape={W2.shape}, mean={W2.mean():.8f}, std={W2.std():.8f}")
    print(f"    b2 mean={b2.mean():.8f}, std={b2.std():.8f}")
    print(f"    b2 L2 norm = {b2.norm():.6f}")
    print(f"    b2 |min| = {b2.abs().min():.8f} |max| = {b2.abs().max():.8f}")
    print(f"    |b2| > 0.01 的通道数: {(b2.abs() > 0.01).sum().item()} / 1024")
    print(f"    |b2| > 0.001 的通道数: {(b2.abs() > 0.001).sum().item()} / 1024")
    print(f"    |b2| > 0.0001 的通道数: {(b2.abs() > 0.0001).sum().item()} / 1024")

    print(f"\n  关键指标 — b2 偏离零的程度:")
    print(f"    L1 norm(b2)   = {b2.abs().sum():.6f}")
    print(f"    L-inf norm(b2)= {b2.abs().max():.6f}")
    print(f"    非零通道占比   = {(b2.abs() > 1e-7).sum().item() * 100 / 1024:.1f}%")
    print(f"    显著通道占比(|>0.005) = {(b2.abs() > 0.005).sum().item() * 100 / 1024:.2f}%")

    # b2 直方图
    b2_np = b2.cpu().numpy()
    bins = [-np.inf, -0.05, -0.01, -0.001, -1e-6, 1e-6, 0.001, 0.01, 0.05, np.inf]
    hist, _ = np.histogram(b2_np, bins=bins)
    labels = ["<-0.05", "-0.05~-0.01", "-0.01~-0.001", "~-0", "~+0",
              "0.001~0.01", "0.01~0.05", ">0.05"]
    print(f"\n  b2 分布直方图:")
    for l, h in zip(labels, hist):
        if h > 0:
            bar = '#' * max(1, h // 3)
            print(f"    {l:>15s}: {h:>4d} {bar}")

    # 正负通道统计
    pos_count = (b2 > 0).sum().item()
    neg_count = (b2 < 0).sum().item()
    print(f"\n  方向分布: 正向(mean注入) {pos_count} 通道 ({pos_count*100/1024:.1f}%) | "
          f"反向(减去mean) {neg_count} 通道 ({neg_count*100/1024:.1f}%)")

    # ========== 2. 动态门控值分析（在真实数据上跑） ==========
    print("\n" + "=" * 70)
    print("【2】动态门控值分析（真实数据上的 dynamic_gate 输出）")
    print("=" * 70)

    from datasets.build import build_dataset_from_cfg
    from torch.utils.data import DataLoader
    from easydict import EasyDict

    test_cfg = EasyDict()
    for k, v in config.dataset.test["_base_"].items():
        test_cfg[k] = v
    for k, v in config.dataset.test["others"].items():
        test_cfg[k] = v
    test_cfg["NAME"] = test_cfg.get("NAME", "PCN")
    test_cfg["subset"] = "test"

    val_dataset = build_dataset_from_cfg(test_cfg)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    all_gates = []
    all_gate_norms = []
    category_gates = {}

    SYNSET_TO_NAME = {
        "04256520": "sofa",
        "03001627": "chair",
        "02958343": "car",
        "04530566": "watercraft",
        "04379243": "table",
        "02691156": "airplane",
        "02933112": "cabinet",
        "03636649": "lamp",
    }

    from models.PGST import xyz2key, sort, get_basis

    # 获取内部 PCTransformer
    base_model = model.base_model

    model.eval()
    with torch.no_grad():
        for taxonomy_id, model_id, data in val_loader:
            partial, gt = data
            partial = partial.cuda()

            # forward 到 gate_net 之后截取中间值
            bs = partial.size(0)
            coor, f = base_model.grouper(partial, base_model.center_num)
            pe = base_model.pos_embed(coor)
            x = base_model.input_proj(f)
            B, G, _ = coor.shape
            c = coor * 100
            key = xyz2key(c[:, :, 1], c[:, :, 0], c[:, :, 2])
            _, idx0 = torch.sort(key)
            _, idx1 = torch.sort(idx0)
            sub_center = sort(coor, idx0)
            sub_U0 = get_basis(sub_center.reshape(B * (G // 16), 16, 3)).reshape(B, G // 16, 16, 16)
            sub_U1 = get_basis(sub_center.reshape(B * (G // 32), 32, 3)).reshape(B, G // 32, 32, 32)

            x = base_model.encoder(x + pe, coor, [sub_U0, sub_U1], [idx0, idx1])
            global_feature = base_model.increase_dim(x)  # (B, N, C)

            max_feat = torch.max(global_feature, dim=1)[0]   # (B, C)
            mean_feat = torch.mean(global_feature, dim=1)    # (B, C)
            dynamic_gate = base_model.gate_net(max_feat)      # (B, C)

            g_np = dynamic_gate[0].cpu().numpy()  # (C,)
            all_gates.append(g_np)
            all_gate_norms.append(np.linalg.norm(g_np))

            tax_name = SYNSET_TO_NAME.get(taxonomy_id[0], str(taxonomy_id[0]))
            if tax_name not in category_gates:
                category_gates[tax_name] = []
            category_gates[tax_name].append(g_np)

    all_gates = np.stack(all_gates, axis=0)  # (N_samples, C)

    print(f"  样本数: {all_gates.shape[0]}")
    print(f"  通道数: {all_gates.shape[1]}")
    print()

    # 全局统计
    print(f"  dynamic_gate 统计 (所有样本 x 所有通道):")
    print(f"    mean  = {all_gates.mean():.6f}")
    print(f"    std   = {all_gates.std():.6f}")
    print(f"    min   = {all_gates.min():.6f}")
    print(f"    max   = {all_gates.max():.6f}")
    print(f"    median= {np.median(all_gates):.6f}")

    print(f"\n  per-sample L2 norm of gate:")
    gate_norms = np.array(all_gate_norms)
    print(f"    mean  = {gate_norms.mean():.4f}")
    print(f"    std   = {gate_norms.std():.4f}")
    print(f"    min   = {gate_norms.min():.4f}")
    print(f"    max   = {gate_norms.max():.4f}")
    zeroish = (gate_norms < 0.1).sum()
    active = len(gate_norms) - zeroish
    print(f"    接近零的样本(norm<0.1): {zeroish}/{len(gate_norms)} ({zeroish*100/len(gate_norms):.1f}%)")
    print(f"    活跃的样本(norm>=0.1):   {active}/{len(gate_norms)} ({active*100/len(gate_norms):.1f}%)")

    # per-channel 统计
    print(f"\n  per-channel gate 统计 (跨样本平均):")
    ch_mean = all_gates.mean(axis=0)  # (C,)
    ch_std = all_gates.std(axis=0)
    ch_max_abs = np.abs(all_gates).max(axis=0)

    print(f"    通道均值范围: [{ch_mean.min():.4f}, {ch_mean.max():.4f}]")
    print(f"    通道标准差范围: [{ch_std.min():.4f}, {ch_std.max():.4f}]")
    print(f"    |mean|>0.01 的通道数: {(np.abs(ch_mean)>0.01).sum()} / 1024")
    print(f"    |mean|>0.001 的通道数: {(np.abs(ch_mean)>0.001).sum()} / 1024")
    print(f"    max_abs>0.05 的通道数: {(ch_max_abs>0.05).sum()} / 1024")

    # Top10 最大激活通道
    top_idx = np.argsort(ch_std)[::-1][:20]
    print(f"\n  Top-20 最活跃通道 (按 std 排序):")
    print(f"    {'Ch#':>5s} {'Mean':>8s} {'Std':>8s} {'MaxAbs':>8s}")
    for i in top_idx:
        print(f"    {i:>5d} {ch_mean[i]:>8.4f} {ch_std[i]:>8.4f} {ch_max_abs[i]:>8.4f}")

    # per-category 分析
    print(f"\n  per-category gate L2 norm (平均):")
    for cat in sorted(category_gates.keys()):
        cat_gates = np.array(category_gates[cat])  # (n_cat_samples, C)
        norms = np.linalg.norm(cat_gates, axis=1)
        print(f"    {cat:>12s}: mean_norm={norms.mean():.4f} ± {norms.std():.4f} "
              f"| gate_mean={cat_gates.mean():.4f} | samples={cat_gates.shape[0]}")

    # ========== 3. 结论判断 ==========
    print("\n" + "=" * 70)
    print("【3】训练有效性结论")
    print("=" * 70)

    b2_l1 = b2.abs().sum().item()
    b2_linf = b2.abs().max().item()
    significant_ch = (b2.abs() > 0.001).sum().item()
    avg_gate_norm = gate_norms.mean()

    print(f"  判断依据:")
    print(f"    ① b2 L1-norm      = {b2_l1:.4f}")
    print(f"    ② b2 L-inf-norm   = {b2_linf:.6f}")
    print(f"    ③ 显著通道数(|>0.001) = {significant_ch}/1024")
    print(f"    ④ 平均动态gate L2-norm = {avg_gate_norm:.4f}")

    if avg_gate_norm > 0.5 or significant_ch > 100:
        verdict = "✅ 训练有效 — 门控已明显偏离初始值，产生了有意义的差异化"
    elif avg_gate_norm > 0.1 or significant_ch > 50:
        verdict = "⚠️ 训练有一定效果 — 门控发生了偏移但幅度有限"
    elif significant_ch > 10:
        verdict = "🔶 微弱效果 — 仅少数通道学到了非零门控"
    else:
        verdict = "❌ 几乎无效 — 门控基本停留在初始化附近"
    print(f"\n  结论: {verdict}")


if __name__ == '__main__':
    main()
