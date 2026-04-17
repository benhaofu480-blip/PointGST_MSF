import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# exp3 验证集
e3_x = [0,10,20,30,40,50,60,70,80,90,100,110,120,130,140]
e3_cd = [10.072,8.199,7.964,7.718,7.649,7.586,7.593,7.539,7.469,7.389,7.451,7.287,7.383,7.397,7.404]
e3_fs = [0.686,0.758,0.770,0.783,0.788,0.792,0.788,0.796,0.799,0.802,0.802,0.809,0.807,0.807,0.808]

# exp4 验证集
e4_x = [0,10,20,30,40,50,60,70,80,90,100,110,120,130,140]
e4_cd = [10.104,8.164,7.907,7.710,7.634,7.643,7.595,7.524,7.475,7.421,7.472,7.327,7.376,7.338,7.353]
e4_fs = [0.683,0.759,0.773,0.784,0.789,0.791,0.789,0.797,0.799,0.802,0.801,0.808,0.807,0.808,0.809]

# exp4 训练 loss (epoch 71-140)
et = list(range(71,141))
sp_loss = [2.772,2.761,2.769,2.769,2.759,2.756,2.758,2.744,2.742,2.737,
           2.724,2.747,2.729,2.745,2.741,2.744,2.753,2.758,2.734,2.742,
           2.734,2.727,2.729,2.726,2.724,2.724,2.740,2.735,2.744,2.737,
           2.722,2.713,2.710,2.708,2.713,2.709,2.711,2.714,2.702,2.705,
           2.695,2.698,2.696,2.705,2.703,2.701,2.700,2.697,2.691,2.696,
           2.696,2.691,2.693,2.691,2.684,2.687,2.682,2.680,2.681,2.679,
           2.682,2.680,2.675,2.673,2.672,2.670,2.677,2.671,2.672,2.666]

dn_loss = [59.29,58.67,59.23,58.97,58.89,58.88,59.19,58.93,59.11,59.09,
           58.67,57.98,57.81,57.40,56.71,56.32,55.83,55.55,54.97,54.93,
           54.31,53.72,53.55,53.04,52.68,52.25,51.94,51.57,51.78,51.01,
           50.46,50.00,49.57,49.01,48.70,48.48,47.88,47.78,47.24,46.74,
           46.43,45.94,45.53,45.27,44.90,44.36,43.96,43.58,43.33,42.97,
           42.68,42.25,41.81,41.35,41.10,40.69,40.38,39.84,39.50,39.12,
           38.87,38.36,37.99,37.53,37.18,36.74,36.47,36.08,35.85,35.38]

fig, axes = plt.subplots(2, 2, figsize=(16,12))

# 图1: CD-L1
ax1 = axes[0][0]
ax1.plot(e3_x, e3_cd, 'o-', color='#e74c3c', lw=2, ms=5, label='exp3 (均衡 1:1)')
ax1.plot(e4_x, e4_cd, 's-', color='#3498db', lw=2, ms=5, label='exp4 (解耦 3:1->1.5:1)')
ax1.axvline(80, color='#3498db', ls='--', alpha=0.5, lw=1.5)
ax1.scatter([110],[7.287], s=200,c='#e74c3c',marker='*',zorder=5,edgecolors='k')
ax1.scatter([110],[7.327], s=200,c='#3498db',marker='*',zorder=5,edgecolors='k')
ax1.annotate(f'exp3 BEST CD-L1={7.287}', xy=(110,7.287), xytext=(118,7.15), fontsize=10,color='#e74c3c',
            arrowprops=dict(arrowstyle='->',color='#e74c3c',alpha=0.5))
ax1.annotate(f'exp4 BEST CD-L1={7.327}', xy=(110,7.327), xytext=(112,7.46), fontsize=10,color='#3498db',
            arrowprops=dict(arrowstyle='->',color='#3498db',alpha=0.5))
ax1.set_xlabel('Epoch'); ax1.set_ylabel('CD-L1 (lower=better)')
ax1.set_title('Validation CD-L1 Comparison', fontweight='bold', fontsize=13)
ax1.legend(fontsize=9); ax1.grid(alpha=0.3); ax1.invert_yaxis()

# 图2: F-Score
ax2 = axes[0][1]
ax2.plot(e3_x, e3_fs, 'o-', color='#e74c3c', lw=2, ms=5, label='exp3')
ax2.plot(e4_x, e4_fs, 's-', color='#3498db', lw=2, ms=5, label='exp4')
ax2.axvline(80, color='gray', ls='--', alpha=0.5)
ax2.set_xlabel('Epoch'); ax2.set_ylabel('F-Score (higher=better)')
ax2.set_title('Validation F-Score Comparison', fontweight='bold', fontsize=13)
ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

# 图3: Training Loss
ax3 = axes[1][0]
ax3t = ax3.twinx()
l1, = ax3.plot(et, sp_loss, '-', color='#2ecc71', lw=2, label='Sparse(Coarse) Loss')
l2, = ax3t.plot(et, dn_loss, '--', color='#9b59b6', lw=2, label='Dense(Fine) Loss')
ax3.axvspan(71, 80, alpha=0.08, color='red', label='Phase1: w_c=3x')
ax3.axvspan(80, 140, alpha=0.08, color='blue', label='Phase2: transition')
ax3.set_xlabel('Epoch'); ax3.set_ylabel('Sparse Loss', color='#2ecc71')
ax3t.set_ylabel('Dense Loss', color='#9b59b6')
ax3.tick_params(axis='y', colors='#2ecc71'); ax3t.tick_params(axis='y', colors='#9b59b6')
ax3.set_title("exp4 Training Loss: Sparse barely moves! Dense drops in Phase2!", fontweight='bold', fontsize=11)
ax3.legend(loc='upper left', fontsize=8); ax3t.legend(loc='center right', fontsize=8)
ax3.grid(alpha=0.3)

# 标注
chg_s = (sp_loss[-1]-sp_loss[0])/sp_loss[0]*100
chg_d = (dn_loss[-1]-dn_loss[0])/dn_loss[0]*100
ax3.annotate(f'Sparse: {sp_loss[0]:.2f}->{sp_loss[-1]:.2f} ({chg_s:.1f}%)',
            xy=(138, sp_loss[-1]), xytext=(125, sp_loss[0]+0.06), fontsize=9, color='#2ecc71',
            bbox=dict(boxstyle='round',fc='white',ec='#2ecc71'))
ax3t.annotate(f'Dense: {dn_loss[0]:.1f}->{dn_loss[-1]:.1f} ({chg_d:.1f}%)',
             xy=(138, dn_loss[-1]), xytext=(122, dn_loss[0]-5), fontsize=9, color='#9b59b6',
             bbox=dict(boxstyle='round',fc='white',ec='#9b59b6'))

# 图4: 诊断文字
ax4 = axes[1][1]
ax4.axis('off')

txt = """
╔══════════════════════════════════════════════════════════╗
║              exp4 FAILURE DIAGNOSIS                     ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║   CORE ISSUE: Sparse(Coarse) loss BARELY DECREASED!      ║
║                                                          ║
║   Epoch │ Sparse │ Dense  │ w_coarse                    ║
║   ──────┼────────┼────────┼────────                     ║
║    71   │ {s0:>6.2f}│{d0:>7.1f}│   3.00x                    ║
║    80   │ {s9:>6.2f}│{d9:>7.1f}│   3.00x  ← Phase1 end       ║
║   110   │{s39:>6.2f}│{d39:>7.1f}│   ~1.65x  ← BEST            ║
║   140   │{sl:>6.2f}│{dl:>7.1f}│   1.50x                    ║
║                                                          ║
║   → Sparse change: {s0:.2f}->{sl:.2f} ({cs:+.1f}%) ← NO LEARNING!     ║
║   → Dense  change:  {d0:.1f}->{dl:.1f} ({cd:+.1f}% ) ← learned a lot!    ║
║                                                          ║
║   ROOT CAUSES:                                            ║
║   1. Coarse bottleneck is MODEL CAPACITY, not loss weight  ║
║      → gft_adapter has too few trainable params          ║
║      → 3x weight cannot push it further                  ║
║                                                          ║
║   2. Fine Decoder was "starved" for 80 epochs             ║
║      → Only 60 epochs of catch-up in Phase2              ║
║      → Final fine quality worse than exp3's full 150ep    ║
║                                                          ║
║   3. Early Stop at epoch=140 is NORMAL behavior           ║
║      → best@110, patience=30, val_freq=10                ║
║      → Both experiments peaked ~epoch 110                 ║
║                                                          ║
║   FINAL VALIDATION COMPARISON:                            ║
║   ┌───────┬────────┬────────┬────────┐                   ║
║   │ Metric│  exp3  │  exp4  │  diff  │                   ║
║   ├───────┼────────┼────────┼────────┤                   ║
║   │CD-L1  │  7.287★│  7.327★│ +0.040 │                   ║
║   │F-Score│  0.809 │  0.809 │   =    │                   ║
║   └───────┴────────┴────────┴────────┘                   ║
║                                                          ║
║   VERDICT: Strategy direction OK, execution flawed      ║
║   Suggestion: Use gradient clipping / LR warmup instead  ║
║                of fixed-weight decoupling                ║
╚══════════════════════════════════════════════════════════╝
""".format(s0=sp_loss[0], d0=dn_loss[0], s9=sp_loss[9], d9=dn_loss[9],
           s39=sp_loss[39], d39=dn_loss[39], sl=sp_loss[-1], dl=dn_loss[-1],
           cs=chg_s, cd=chg_d)

ax4.text(0.02, 0.98, txt, transform=ax4.transAxes, fontsize=9.5, fontfamily='monospace',
         verticalalignment='top',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray'))

plt.tight_layout()
plt.savefig('/home/fubenhao/data/fubenhao_data/PointGST-main_pure/completion/fig/exp4_analysis.png', dpi=150, bbox_inches='tight')
print('Saved fig/exp4_analysis.png')
plt.close()
