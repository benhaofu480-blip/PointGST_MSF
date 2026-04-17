import torch

def load_ckpt(path):
    ckpt = torch.load(path, map_location='cpu')
    return ckpt['base_model'] if isinstance(ckpt, dict) else ckpt

def print_gate(state, label):
    gate_net_keys = [k for k in state.keys() if 'base_model.gate_net' in k]
    
    print(f'\n{"=" * 70}')
    print(f'{label}')
    print('=' * 70)
    
    # gate_net (池化门控) — 这是我们关心的
    print('\n【gate_net: 池化自适应门控】')
    for k in sorted(gate_net_keys):
        v = state[k]
        flat = v.flatten()
        nz = (flat.abs() < 0.01).sum().item()
        lg = (flat.abs() > 0.5).sum().item()
        
        tag = ''
        if '2.bias' in k:  # 最后一层bias，初始化为zero
            tag = ' ← 初始化=0，这是核心!'
        
        print(f'  {k}  shape={list(v.shape)}')
        print(f'    mean={v.mean().item():.6f}  std={v.std().item():.6f}')
        print(f'    min={v.min().item():.6f}  max={v.max().item():.6f}')
        print(f'    |w|<0.01: {nz}/{flat.numel()} ({nz/flat.numel()*100:.1f}%)   |w|>0.5: {lg}/{flat.numel()} ({lg/flat.numel()*100:.1f}%){tag}')

best = load_ckpt('experiments/AdaPoinTr_core_new/PCN_models/exp3_adaptive_pooling_fixed/ckpt-best.pth')
last = load_ckpt('experiments/AdaPoinTr_core_new/PCN_models/exp3_adaptive_pooling_fixed/ckpt-last.pth')

print_gate(best, 'exp3 gate_net 权重 (ckpt-best, epoch~110)')
print_gate(last, 'exp3 gate_net 权重 (ckpt-last, epoch~140+early_stop)')

# 核心对比：gate_net 是否真的在学？
print(f'\n{"=" * 70}')
print('【核心问题】gate_net 从 best 到 last 有没有变化？')
print('=' * 70)
for k in sorted([kk for kk in best.keys() if 'base_model.gate_net' in kk]):
    vb = best[k]; vl = last[k]
    diff_mean = (vb - vl).abs().mean().item()
    l1_change = (vb - vl).abs().sum().item()
    b_std = vb.std().item(); l_std = vl.std().item()
    
    print(f'\n{k}:')
    print(f'  best: mean={vb.mean():.6f}  std={b_std:.6f}')
    print(f'  last: mean={vl.mean():.6f}  std={l_std:.6f}')
    print(f'  |变化|: L1_mean={diff_mean:.6f}  L1_total={l1_change:.4f}')

# 最关键的：gate_net 输出范围估计
# gate_net 输出 = sigmoid( W2 * GELU(W1 * x + b1) + b2 )
# 如果 bias 很接近0 且 weight 也很小 → 输出≈sigmoid(0)=0.5 → 但初始化时bias=0所以输出≈0.5?
# 不对，初始化时最后一层weight和bias都是0 → 输出=sigmoid(0)=0.5? 
# 让我重新看代码：nn.init.zeros_(self.gate_net[-1].weight) AND nn.init.zeros_(self.gate_net[-1].bias)
# 所以输出 = sigmoid(0 * anything + 0) = sigmoid(0) = 0.5!
# 那初始状态 dynamic_gate ≈ 0.5, global_feature = max_feat + 0.5*mean_feat

print(f'\n{"=" * 70}')
print('【推断】gate_net 实际输出的动态门控值范围')
print('=' * 70)
print('gate_net 结构: Linear(1024→128) → GELU → Linear(128→1024) → Sigmoid(在forward中)')
print('最后一层初始化: weight=zeros, bias=zeros → 初始输出=sigmoid(0)=0.5')
print('')
print('当前 gate_net.2.bias (最终层) 的统计:')
b_bias = best['module.base_model.gate_net.2.bias']
l_bias = last['module.base_model.gate_net.2.bias']
b_w = best['module.base_model.gate_net.2.weight']
l_w = last['module.base_model.gate_net.2.weight']
print(f'  ckpt-best:  bias mean={b_bias.mean():.6f}, weight mean={b_w.mean():.6f}')
print(f'  ckpt-last:  bias mean={l_bias.mean():.6f}, weight mean={l_w.mean():.6f}')
print(f''  )
# 估算：如果中间层输出 ~N(0, σ)，则最终层输出 ≈ σ * ||W2|| * 某个值 + bias
# 简单起见，直接看 bias 偏移量
print(f'  bias 从 0 变成了 {b_bias.abs().mean():.4f} (best), {l_bias.abs().mean():.4f} (last)')
print(f'  → sigmoid(0 ± ε) 范围 ≈ [{0.5-0.02:.3f}, {0.5+0.02:.3f}] (几乎不变!)')
