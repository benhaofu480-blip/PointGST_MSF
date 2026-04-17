import torch
import numpy as np

def load_state(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    return ckpt['base_model']

state_best = load_state('experiments/AdaPoinTr_core_new/PCN_models/exp3_adaptive_pooling_fixed/ckpt-best.pth')
state_last = load_state('experiments/AdaPoinTr_core_new/PCN_models/exp3_adaptive_pooling_fixed/ckpt-last.pth')

print('=' * 90)
print('exp3 gate_net 全层级分析: ckpt-best vs ckpt-last (训练140轮)')
print('=' * 90)

# ========== Part 1: 池化用的 gate_net (顶层) ==========
print('\n' + '=' * 90)
print('[Part 1] 池化 gate_net (global_feature 混合门控)')
print('结构: Linear(1024->128) -> GELU -> Linear(128->1024) -> sigmoid')
print('=' * 90)

gate_layers = [
    ('gate_net.0.weight', '第1层 Weight (1024->128)'),
    ('gate_net.0.bias',   '第1层 Bias   (128,)'),
    ('gate_net.2.weight', '第2层 Weight (128->1024, 初始化=0)'),
    ('gate_net.2.bias',   '第2层 Bias   (1024, 初始化=0)'),
]

for key_suffix, desc in gate_layers:
    full_key = f'module.base_model.{key_suffix}'
    v_b = state_best[full_key]
    v_l = state_last[full_key]
    
    print(f'\n--- {desc} ---')
    print(f'  Key: {key_suffix}')
    print(f'  Shape: {list(v_b.shape)}  Params: {v_b.numel():,}')
    
    flat_b, flat_l = v_b.flatten(), v_l.flatten()
    
    # 判断初始化方式
    if 'gate_net.2' in key_suffix:
        init_info = "全零初始化"
    elif '.bias' in key_suffix:
        init_info = "默认零初始化"
    else:
        import math
        fan_in = v_b.shape[1]
        std_init = math.sqrt(2.0 / fan_in)
        init_info = f"kaiming (std≈{std_init:.4f})"
    print(f'  初始化: {init_info}')
    
    print(f'  {"":8s} {"Mean":>10s} {"Std":>10s} {"Min":>12s} {"Max":>12s} {"|x|<0.01":>10s} {"|x|>0.5":>10s}')
    print(f'  {"best":8s} {flat_b.mean():10.5f} {flat_b.std():10.5f} '
          f'{flat_b.min():12.5f} {flat_b.max():12.5f} {(flat_b.abs()<0.01).sum():10d} {(flat_b.abs()>0.5).sum():10d}')
    print(f'  {"last":8s} {flat_l.mean():10.5f} {flat_l.std():10.5f} '
          f'{flat_l.min():12.5f} {flat_l.max():12.5f} {(flat_l.abs()<0.01).sum():10d} {(flat_l.abs()>0.5).sum():10d}')
    
    l1_change = (v_b - v_l).abs().mean().item()
    l2_change = ((v_b - v_l)**2).mean().item()**0.5
    ratio = l1_change / (v_b.abs().mean().item() + 1e-8)
    print(f'  best→last L1变化={l1_change:.6f}, L2变化={l2_change:.6f}, 相对变化率={ratio*100:.2f}%')

# 模拟前向输出
print('\n--- gate_net 实际输出模拟 ---')
import torch.nn as nn
class GateNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1024,128), nn.GELU(), nn.Linear(128,1024))
    def forward(self, x): return torch.sigmoid(self.net(x))

gate = GateNet()
sd = {}
for ks in ['0.weight','0.bias','2.weight','2.bias']:
    sd[ks] = state_best[f'module.base_model.gate_net.{ks}']
gate.net.load_state_dict(sd)
gate.eval()

torch.manual_seed(42)
test_in = torch.randn(32, 1024)
with torch.no_grad():
    out_best = gate(test_in)

gate_last = GateNet()
sd_l = {}
for ks in ['0.weight','0.bias','2.weight','2.bias']:
    sd_l[ks] = state_last[f'module.base_model.gate_net.{ks}']
gate_last.net.load_state_dict(sd_l)
gate_last.eval()
with torch.no_grad():
    out_last = gate_last(test_in)

print(f'  输入: randn(32, 1024)')
print(f'  {"":8s} {"Mean":>8s} {"Std":>8s} {"Min":>8s} {"Max":>8s} {"<0.3占比":>10s} {"0.4~0.6占比":>12s} {">0.7占比":>10s}')
for name, out in [('best', out_best), ('last', out_last)]:
    flat = out.flatten()
    print(f'  {name:8s} {out.mean():8.5f} {out.std():8.5f} {out.min():8.5f} {out.max():8.5f} '
          f'{(flat<0.3).sum()*100/flat.numel():9.1f}% {( (flat>=0.4)&(flat<=0.6) ).sum()*100/flat.numel():11.1f}% '
          f'{(flat>0.7).sum()*100/flat.numel():9.1f}%')

print(f'\n  结论: gate值 ≈ {out_best.mean().item():.3f} ± {out_best.std().item():.3f}')
print(f'  → global_feat = max_feat + ({out_best.mean():.3f}) × mean_feat')
if abs(out_best.mean().item() - 0.5) < 0.05:
    print(f'  ≈ 固定50/50混合，自适应能力极弱！')

# ========== Part 2: GFT encoder 内每层的 scale_gate ==========
print('\n\n' + '=' * 90)
print('[Part 2] GFT Encoder 各层 scale_gate (共6层encoder)')
print('=' * 90)

for layer_idx in range(6):
    prefix = f'module.base_model.encoder.blocks.blocks.{layer_idx}.gft_adapter.scale_gate'
    w0 = state_best[f'{prefix}.0.weight']  # [18, 36]
    b0 = state_best[f'{prefix}.0.bias']    # [18]
    w2 = state_best[f'{prefix}.2.weight']  # [2, 18]
    b2 = state_best[f'{prefix}.2.bias']    # [2]
    
    w0_l = state_last[f'{prefix}.0.weight']
    b0_l = state_last[f'{prefix}.0.bias']
    w2_l = state_last[f'{prefix}.2.weight']
    b2_l = state_last[f'{prefix}.2.bias']
    
    print(f'\n--- Encoder Layer {layer_idx} scale_gate ---')
    print(f'  结构: Linear(36->18) -> GELU -> Linear(18->2) -> softmax')
    
    # 第1层
    print(f'  Layer0 W(18,36): best_mean={w0.mean():.5f}±{w0.std():.5f}  last_mean={w0_l.mean():.5f}  ΔL1={(w0-w0_l).abs().mean():.5f}')
    print(f'  Layer0 B(18,):  best_mean={b0.mean():.5f}±{b0.std():.5f}  last_mean={b0_l.mean():.5f}  ΔL1={(b0-b0_l).abs().mean():.5f}')
    
    # 第2层 (scale_gate的最终输出是2维softmax，控制16和32倍缩放的混合权重)
    print(f'  Layer2 W(2,18):  best_mean={w2.mean():.5f}±{w2.std():.5f}  last_mean={w2_l.mean():.5f}  ΔL1={(w2-w2_l).abs().mean():.5f}')
    print(f'  Layer2 B(2,):   best=[{b2[0]:.4f}, {b2[1]:.4f}]  last=[{b2_l[0]:.4f}, {b2_l[1]:.4f}]')
    
    # 计算实际输出的 scale 权重分布
    import torch.nn.functional as F
    with torch.no_grad():
        test = torch.randn(64, 36)
        h = F.gelu(F.linear(test, w0, b0))
        out_s = F.softmax(F.linear(h, w2, b2), dim=-1)
        
        h_l = F.gelu(F.linear(test, w0_l, b0_l))
        out_sl = F.softmax(F.linear(h_l, w2_l, b2_l), dim=-1)
    
    print(f'  输出softmax(best):  scale_16权={out_s[:,0].mean():.4f}±{out_s[:,0].std():.4f}  scale_32权={out_s[:,1].mean():.4f}±{out_s[:,1].std():.4f}')
    print(f'  输出softmax(last):  scale_16权={out_sl[:,0].mean():.4f}±{out_sl[:,0].std():.4f}  scale_32权={out_sl[:,1].mean():.4f}±{out_sl[:,1].std():.4f}')

print('\n\n' + '=' * 90)
print('总结')
print('=' * 90)
