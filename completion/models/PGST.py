##############################################################
# % Author: Castle
# % Date:01/12/2022
###############################################################

import torch
import torch.nn as nn
from functools import partial, reduce
from timm.models.layers import DropPath, trunc_normal_
from extensions.chamfer_dist import ChamferDistanceL1
from .build import MODELS, build_model_from_cfg
from models.Transformer_utils import *
from utils import misc
from .z_order import xyz2key

class PCSA(nn.Module):
    """原版 PCSA（与 Paper_related/PGST.py 同构，forward 含 down/up 谱适配）。"""
    def __init__(self, dim, adapt_dim):
        super().__init__()
        self.adapt_dim = adapt_dim
        self.norm_ly1 = nn.LayerNorm(adapt_dim)
        self.norm_ly2 = nn.LayerNorm(adapt_dim)
        self.act = nn.SiLU()
        self.down = nn.Linear(dim, adapt_dim)
        self.up = nn.Linear(adapt_dim, dim)
        self.adapt = nn.Linear(adapt_dim, adapt_dim)
        nn.init.zeros_(self.adapt.weight)
        nn.init.zeros_(self.adapt.bias)
        self.drop_fourier = DropPath(0.)
        self.drop_adapt1 = DropPath(0.)
        self.drop_adapt2 = DropPath(0.)
        self.drop_out = nn.Dropout(0.)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, input, sub_U, idx):
        h = self.down(input)
        h = self.act(h)
        x = h

        B0, group_num, group_size, _ = sub_U[0].shape
        G0 = group_num * group_size
        sub_x0 = sort(x, idx[0])
        sub_x0 = sub_x0.reshape(B0, group_num, group_size, self.adapt_dim)

        B1, group_num, group_size, _ = sub_U[1].shape
        sub_x1 = sub_x0.reshape(B1, group_num, group_size, self.adapt_dim)

        sub_x_f0 = sub_U[0].transpose(-2, -1) @ sub_x0
        sub_h_f0 = sub_x_f0
        sub_x_f0 = self.norm_ly2(sub_x_f0)
        sub_x_f0 = sub_h_f0 + self.drop_adapt1(self.act(self.drop_out(self.adapt(sub_x_f0))))
        sub_x0 = sub_U[0] @ sub_x_f0

        sub_x0 = sub_x0.reshape(B0, G0, self.adapt_dim)
        sub_x0 = sort(sub_x0, idx[1])

        sub_x_f1 = sub_U[1].transpose(-2, -1) @ sub_x1
        sub_h_f1 = sub_x_f1
        sub_x_f1 = self.norm_ly2(sub_x_f1)
        sub_x_f1 = sub_h_f1 + self.drop_adapt2(self.act(self.drop_out(self.adapt(sub_x_f1))))
        sub_x1 = sub_U[1] @ sub_x_f1

        sub_x1 = sub_x1.reshape(B0, G0, self.adapt_dim)
        sub_x0 = sort(sub_x1, idx[1])

        x = sub_x0 + sub_x1
        h = x + h
        h = self.up(h)
        return h


class MSF(nn.Module):
    """
    Multi-scale Fusion (MSF) Module
    替代 PCSA，修复逆置换 Bug、解耦频率滤波器、引入残差独立动态门控。
    初始时与 baseline 完全等价：(1+0)*sub_x0 + (1+0)*sub_x1 = sub_x0 + sub_x1
    """
    def __init__(self, dim, adapt_dim):
        super().__init__()
        self.adapt_dim = adapt_dim

        self.norm_ly2 = nn.LayerNorm(adapt_dim)
        self.act = nn.SiLU()
        self.down = nn.Linear(dim, adapt_dim)
        self.up = nn.Linear(adapt_dim, dim)

        # 解耦的独立滤波器
        self.adapt16 = nn.Linear(adapt_dim, adapt_dim)
        self.adapt32 = nn.Linear(adapt_dim, adapt_dim)
        nn.init.zeros_(self.adapt16.weight)
        nn.init.zeros_(self.adapt16.bias)
        nn.init.zeros_(self.adapt32.weight)
        nn.init.zeros_(self.adapt32.bias)

        # 残差双独立门控：输出 2 个通道，分别对应两个尺度的权重偏置
        self.scale_gate = nn.Sequential(
            nn.Linear(adapt_dim, adapt_dim // 2),
            nn.GELU(),
            nn.Linear(adapt_dim // 2, 2),
        )
        nn.init.zeros_(self.scale_gate[-1].weight)
        nn.init.zeros_(self.scale_gate[-1].bias)

        self.drop_adapt1 = DropPath(0.)
        self.drop_adapt2 = DropPath(0.)
        self.drop_out = nn.Dropout(0.)

        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, input, sub_U, idx):
        h = self.down(input)
        h = self.act(h)
        x = h

        # ==================== 准备数据 ====================
        B0, group_num0, group_size0, _ = sub_U[0].shape
        G0 = group_num0 * group_size0
        B1, group_num1, group_size1, _ = sub_U[1].shape
        G1 = group_num1 * group_size1

        # 将原始特征按 Z-order 排序，两个尺度各自从排序后的全量数据 reshape
        x_sorted = sort(x, idx[0])
        sub_x0 = x_sorted.reshape(B0, group_num0, group_size0, self.adapt_dim)
        sub_x1 = x_sorted.reshape(B1, group_num1, group_size1, self.adapt_dim)

        # ==================== 尺度 0 (16x16) 处理 ====================
        sub_x_f0 = sub_U[0].transpose(-2, -1) @ sub_x0
        sub_h_f0 = sub_x_f0
        sub_x_f0 = self.norm_ly2(sub_x_f0)
        sub_x_f0 = sub_h_f0 + self.drop_adapt1(self.act(self.drop_out(self.adapt16(sub_x_f0))))
        sub_x0 = sub_U[0] @ sub_x_f0
        sub_x0 = sub_x0.reshape(B0, G0, self.adapt_dim)
        # 逆置换：还原回空间坐标顺序
        sub_x0_restored = sort(sub_x0, idx[1])

        # ==================== 尺度 1 (32x32) 处理 ====================
        sub_x_f1 = sub_U[1].transpose(-2, -1) @ sub_x1
        sub_h_f1 = sub_x_f1
        sub_x_f1 = self.norm_ly2(sub_x_f1)
        sub_x_f1 = sub_h_f1 + self.drop_adapt2(self.act(self.drop_out(self.adapt32(sub_x_f1))))
        sub_x1 = sub_U[1] @ sub_x_f1
        sub_x1 = sub_x1.reshape(B1, G1, self.adapt_dim)
        # 逆置换：各自独立还原
        sub_x1_restored = sort(sub_x1, idx[1])

        # ==================== 残差独立动态融合 ====================
        # 用还原后特征的均值感知局部结构
        fusion_feat = (sub_x0_restored + sub_x1_restored) / 2.0
        # gate_weights: (B, G, 2)
        gate_weights = self.scale_gate(fusion_feat)
        w0_delta = gate_weights[..., 0:1]
        w1_delta = gate_weights[..., 1:2]
        # 初始时 w0_delta=w1_delta=0 → (1+0)*x0+(1+0)*x1 = x0+x1，与 baseline 等价
        x = (1.0 + w0_delta) * sub_x0_restored + (1.0 + w1_delta) * sub_x1_restored

        # ==================== 输出 ====================
        h = x + h
        h = self.up(h)
        return h

class MSF_scalar_group_refined_v2_final(nn.Module):
    """
    Group-dominant scalar gating + point residual correction with bounded soft routing.
    - Group-level logits provide stable coarse routing
    - Point-level residual logits provide lightweight local adjustment
    - Softmax ensures normalized gate competition (sum-to-one, non-negative)
    - Scale factor restores add-style magnitude (initially around 1.0+x0 + 1.0+x1 semantics)
    - Monitor uses class-level buffers, compatible with classmethod flush_gate_stats()
    """
    _g0_vals = []
    _g1_vals = []
    _tau_vals = []

    @classmethod
    def flush_gate_stats(cls, logger=None):
        if not cls._g0_vals:
            return
        g0 = torch.cat(cls._g0_vals).float()
        g1 = torch.cat(cls._g1_vals).float()
        tau = torch.cat(cls._tau_vals).float()
        msg = (
            "[Gate Monitor] g0 mean/min/max/std = "
            f"{g0.mean():.4f} {g0.min():.4f} {g0.max():.4f} {g0.std():.4f} | "
            f"g1 mean/min/max/std = "
            f"{g1.mean():.4f} {g1.min():.4f} {g1.max():.4f} {g1.std():.4f} | "
            f"tau mean/min/max/std = "
            f"{tau.mean():.4f} {tau.min():.4f} {tau.max():.4f} {tau.std():.4f}"
        )
        print_log(msg, logger=logger)
        cls._g0_vals.clear()
        cls._g1_vals.clear()
        cls._tau_vals.clear()

    def __init__(self, dim, adapt_dim):
        super().__init__()
        self.adapt_dim = adapt_dim
        self.eps = 1e-6

        self.norm_ly2 = nn.LayerNorm(adapt_dim)
        self.act = nn.SiLU()
        self.down = nn.Linear(dim, adapt_dim)
        self.up = nn.Linear(adapt_dim, dim)

        # Decoupled spectral filters
        self.adapt16 = nn.Linear(adapt_dim, adapt_dim)
        self.adapt32 = nn.Linear(adapt_dim, adapt_dim)
        nn.init.zeros_(self.adapt16.weight)
        nn.init.zeros_(self.adapt16.bias)
        nn.init.zeros_(self.adapt32.weight)
        nn.init.zeros_(self.adapt32.bias)

        # Group energy encoder
        self.energy_mlp0 = nn.Sequential(
            nn.Linear(adapt_dim, adapt_dim // 2),
            nn.GELU(),
            nn.Linear(adapt_dim // 2, adapt_dim),
        )
        self.energy_mlp1 = nn.Sequential(
            nn.Linear(adapt_dim, adapt_dim // 2),
            nn.GELU(),
            nn.Linear(adapt_dim // 2, adapt_dim),
        )

        # Group-level routing logits (per group)
        self.group_mlp0 = nn.Sequential(
            nn.Linear(adapt_dim, adapt_dim),
            nn.GELU(),
            nn.Linear(adapt_dim, 1),
        )
        self.group_mlp1 = nn.Sequential(
            nn.Linear(adapt_dim, adapt_dim),
            nn.GELU(),
            nn.Linear(adapt_dim, 1),
        )

        # Point-level residual routing logits (lightweight)
        self.refine_mlp0 = nn.Sequential(
            nn.Linear(adapt_dim * 2, adapt_dim),
            nn.GELU(),
            nn.Linear(adapt_dim, 1),
        )
        self.refine_mlp1 = nn.Sequential(
            nn.Linear(adapt_dim * 2, adapt_dim),
            nn.GELU(),
            nn.Linear(adapt_dim, 1),
        )

        # Initialize to near identity/add-style behavior
        nn.init.zeros_(self.group_mlp0[-1].weight)
        nn.init.zeros_(self.group_mlp0[-1].bias)
        nn.init.zeros_(self.group_mlp1[-1].weight)
        nn.init.zeros_(self.group_mlp1[-1].bias)
        nn.init.zeros_(self.refine_mlp0[-1].weight)
        nn.init.zeros_(self.refine_mlp0[-1].bias)
        nn.init.zeros_(self.refine_mlp1[-1].weight)
        nn.init.zeros_(self.refine_mlp1[-1].bias)

        self.drop_adapt1 = DropPath(0.)
        self.drop_adapt2 = DropPath(0.)
        self.drop_out = nn.Dropout(0.)

        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

        # Routing control
        self.log_tau = nn.Parameter(torch.tensor(-1.0))  # tau = 0.4 + 1.6 * sigmoid(log_tau) in [0.4, 2.0]
        self.logit_clip = 2.0
        self.point_delta_scale = 0.10
        self.fuse_scale = 2.0  # compensate softmax 0.5+0.5 init to keep magnitude consistent with prior add-style fusion

    def _assert_shapes(self, B0, B1, G0, G1):
        assert B0 == B1, f'Batch mismatch: B0={B0}, B1={B1}'
        assert G0 == G1, f'Point count mismatch: G0={G0}, G1={G1}'

    def forward(self, input, sub_U, idx):
        h = self.down(input)
        h = self.act(h)
        x = h

        # 1) multi-scale reshape
        B0, group_num0, group_size0, _ = sub_U[0].shape
        G0 = group_num0 * group_size0
        B1, group_num1, group_size1, _ = sub_U[1].shape
        G1 = group_num1 * group_size1
        self._assert_shapes(B0, B1, G0, G1)

        x_sorted = sort(x, idx[0])
        sub_x0 = x_sorted.reshape(B0, group_num0, group_size0, self.adapt_dim)
        sub_x1 = x_sorted.reshape(B1, group_num1, group_size1, self.adapt_dim)

        # 2) spectral branch 16
        sub_x_f0 = sub_U[0].transpose(-2, -1) @ sub_x0
        sub_h_f0 = sub_x_f0
        sub_x_f0 = self.norm_ly2(sub_x_f0)
        sub_x_f0 = sub_h_f0 + self.drop_adapt1(self.act(self.drop_out(self.adapt16(sub_x_f0))))
        sub_x0 = sub_U[0] @ sub_x_f0
        sub_x0 = sub_x0.reshape(B0, G0, self.adapt_dim)
        sub_x0_restored = sort(sub_x0, idx[1])

        # 3) spectral branch 32
        sub_x_f1 = sub_U[1].transpose(-2, -1) @ sub_x1
        sub_h_f1 = sub_x_f1
        sub_x_f1 = self.norm_ly2(sub_x_f1)
        sub_x_f1 = sub_h_f1 + self.drop_adapt2(self.act(self.drop_out(self.adapt32(sub_x_f1))))
        sub_x1 = sub_U[1] @ sub_x_f1
        sub_x1 = sub_x1.reshape(B1, G1, self.adapt_dim)
        sub_x1_restored = sort(sub_x1, idx[1])

        # 4) group energy context
        energy_0 = torch.sqrt((sub_x_f0 ** 2).mean(dim=2) + 1e-6)
        energy_1 = torch.sqrt((sub_x_f1 ** 2).mean(dim=2) + 1e-6)
        ctx_0 = self.energy_mlp0(energy_0)  # (B, group_num0, adapt_dim)
        ctx_1 = self.energy_mlp1(energy_1)  # (B, group_num1, adapt_dim)

        # broadcast context to point-level and restore order
        ctx_0_spatial = sort(
            ctx_0.unsqueeze(2).expand(B0, group_num0, group_size0, self.adapt_dim).reshape(B0, G0, self.adapt_dim),
            idx[1]
        )
        ctx_1_spatial = sort(
            ctx_1.unsqueeze(2).expand(B1, group_num1, group_size1, self.adapt_dim).reshape(B1, G1, self.adapt_dim),
            idx[1]
        )

        # 5) group logits -> point-level by broadcast
        group_logit0 = self.group_mlp0(ctx_0)  # (B, group_num0, 1)
        group_logit1 = self.group_mlp1(ctx_1)  # (B, group_num1, 1)
        group_logit0 = sort(
            group_logit0.unsqueeze(2).expand(B0, group_num0, group_size0, 1).reshape(B0, G0, 1),
            idx[1]
        )
        group_logit1 = sort(
            group_logit1.unsqueeze(2).expand(B1, group_num1, group_size1, 1).reshape(B1, G1, 1),
            idx[1]
        )

        # 6) point residual logits with context
        delta_input0 = torch.cat([sub_x0_restored, ctx_0_spatial], dim=-1)
        delta_input1 = torch.cat([sub_x1_restored, ctx_1_spatial], dim=-1)
        delta_logit0 = torch.tanh(self.refine_mlp0(delta_input0)) * self.point_delta_scale
        delta_logit1 = torch.tanh(self.refine_mlp1(delta_input1)) * self.point_delta_scale

        # 7) bounded + soften routing
        logit0 = torch.clamp(group_logit0 + delta_logit0, -self.logit_clip, self.logit_clip)
        logit1 = torch.clamp(group_logit1 + delta_logit1, -self.logit_clip, self.logit_clip)

        tau = 0.4 + 1.6 * torch.sigmoid(self.log_tau)  # [0.4, 2.0]
        logits = torch.cat([logit0, logit1], dim=-1) / tau
        gate = torch.softmax(logits, dim=-1)
        gate0 = gate[..., 0:1]
        gate1 = gate[..., 1:2]

        # monitor
        if not self.training:
            cls = self.__class__
            cls._g0_vals.append(gate0.detach().cpu().float().reshape(-1))
            cls._g1_vals.append(gate1.detach().cpu().float().reshape(-1))
            cls._tau_vals.append(tau.detach().cpu().float().reshape(-1))

        # 8) fuse (scale restored to add-style magnitude)
        x = (gate0 * sub_x0_restored + gate1 * sub_x1_restored) * self.fuse_scale

        # 9) residual output
        h = x + h
        h = self.up(h)
        return h


class MSF_scalar_group_refined_v2_tanh(nn.Module):
    """
    Group-dominant scalar gating + point residual correction with bounded tanh routing.
    - Group-level logits provide stable coarse routing
    - Point-level residual logits provide lightweight local adjustment
    - gate = 1 + alpha * tanh(logit), no softmax/temperature needed
    - Logit clipping and gate-stat monitoring for stability
    - Same fusion form: x = (gate0 * x0 + gate1 * x1) * fuse_scale
    """
    _g0_vals = []
    _g1_vals = []

    @classmethod
    def flush_gate_stats(cls, logger=None):
        if not cls._g0_vals:
            return
        g0 = torch.cat(cls._g0_vals).float()
        g1 = torch.cat(cls._g1_vals).float()
        msg = (
            "[Gate Monitor] g0 mean/min/max/std = "
            f"{g0.mean():.4f} {g0.min():.4f} {g0.max():.4f} {g0.std():.4f} | "
            f"g1 mean/min/max/std = "
            f"{g1.mean():.4f} {g1.min():.4f} {g1.max():.4f} {g1.std():.4f}"
        )
        print_log(msg, logger=logger)
        cls._g0_vals.clear()
        cls._g1_vals.clear()

    def __init__(self, dim, adapt_dim):
        super().__init__()
        self.adapt_dim = adapt_dim
        self.eps = 1e-6

        self.norm_ly2 = nn.LayerNorm(adapt_dim)
        self.act = nn.SiLU()
        self.down = nn.Linear(dim, adapt_dim)
        self.up = nn.Linear(adapt_dim, dim)

        # Decoupled spectral filters
        self.adapt16 = nn.Linear(adapt_dim, adapt_dim)
        self.adapt32 = nn.Linear(adapt_dim, adapt_dim)
        nn.init.zeros_(self.adapt16.weight)
        nn.init.zeros_(self.adapt16.bias)
        nn.init.zeros_(self.adapt32.weight)
        nn.init.zeros_(self.adapt32.bias)

        # Group energy encoder
        self.energy_mlp0 = nn.Sequential(
            nn.Linear(adapt_dim, adapt_dim // 2),
            nn.GELU(),
            nn.Linear(adapt_dim // 2, adapt_dim),
        )
        self.energy_mlp1 = nn.Sequential(
            nn.Linear(adapt_dim, adapt_dim // 2),
            nn.GELU(),
            nn.Linear(adapt_dim // 2, adapt_dim),
        )

        # Group-level routing logits (per group)
        self.group_mlp0 = nn.Sequential(
            nn.Linear(adapt_dim, adapt_dim),
            nn.GELU(),
            nn.Linear(adapt_dim, 1),
        )
        self.group_mlp1 = nn.Sequential(
            nn.Linear(adapt_dim, adapt_dim),
            nn.GELU(),
            nn.Linear(adapt_dim, 1),
        )

        # Point-level residual routing logits (lightweight)
        self.refine_mlp0 = nn.Sequential(
            nn.Linear(adapt_dim * 2, adapt_dim),
            nn.GELU(),
            nn.Linear(adapt_dim, 1),
        )
        self.refine_mlp1 = nn.Sequential(
            nn.Linear(adapt_dim * 2, adapt_dim),
            nn.GELU(),
            nn.Linear(adapt_dim, 1),
        )

        # Initialize to near identity/add-style behavior
        nn.init.zeros_(self.group_mlp0[-1].weight)
        nn.init.zeros_(self.group_mlp0[-1].bias)
        nn.init.zeros_(self.group_mlp1[-1].weight)
        nn.init.zeros_(self.group_mlp1[-1].bias)
        nn.init.zeros_(self.refine_mlp0[-1].weight)
        nn.init.zeros_(self.refine_mlp0[-1].bias)
        nn.init.zeros_(self.refine_mlp1[-1].weight)
        nn.init.zeros_(self.refine_mlp1[-1].bias)

        self.drop_adapt1 = DropPath(0.)
        self.drop_adapt2 = DropPath(0.)
        self.drop_out = nn.Dropout(0.)

        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

        # Routing control
        self.logit_clip = 2.0
        self.point_delta_scale = 0.10
        self.gate_scale = 0.6
        self.fuse_scale = 1.0  # keep fusion magnitude around add-style after pointwise residual gates

    def _assert_shapes(self, B0, B1, G0, G1):
        assert B0 == B1, f'Batch mismatch: B0={B0}, B1={B1}'
        assert G0 == G1, f'Point count mismatch: G0={G0}, G1={G1}'

    def forward(self, input, sub_U, idx):
        h = self.down(input)
        h = self.act(h)
        x = h

        # 1) multi-scale reshape
        B0, group_num0, group_size0, _ = sub_U[0].shape
        G0 = group_num0 * group_size0
        B1, group_num1, group_size1, _ = sub_U[1].shape
        G1 = group_num1 * group_size1
        self._assert_shapes(B0, B1, G0, G1)

        x_sorted = sort(x, idx[0])
        sub_x0 = x_sorted.reshape(B0, group_num0, group_size0, self.adapt_dim)
        sub_x1 = x_sorted.reshape(B1, group_num1, group_size1, self.adapt_dim)

        # 2) spectral branch 16
        sub_x_f0 = sub_U[0].transpose(-2, -1) @ sub_x0
        sub_h_f0 = sub_x_f0
        sub_x_f0 = self.norm_ly2(sub_x_f0)
        sub_x_f0 = sub_h_f0 + self.drop_adapt1(self.act(self.drop_out(self.adapt16(sub_x_f0))))
        sub_x0 = sub_U[0] @ sub_x_f0
        sub_x0 = sub_x0.reshape(B0, G0, self.adapt_dim)
        sub_x0_restored = sort(sub_x0, idx[1])

        # 3) spectral branch 32
        sub_x_f1 = sub_U[1].transpose(-2, -1) @ sub_x1
        sub_h_f1 = sub_x_f1
        sub_x_f1 = self.norm_ly2(sub_x_f1)
        sub_x_f1 = sub_h_f1 + self.drop_adapt2(self.act(self.drop_out(self.adapt32(sub_x_f1))))
        sub_x1 = sub_U[1] @ sub_x_f1
        sub_x1 = sub_x1.reshape(B1, G1, self.adapt_dim)
        sub_x1_restored = sort(sub_x1, idx[1])

        # 4) group energy context
        energy_0 = torch.sqrt((sub_x_f0 ** 2).mean(dim=2) + 1e-6)
        energy_1 = torch.sqrt((sub_x_f1 ** 2).mean(dim=2) + 1e-6)
        ctx_0 = self.energy_mlp0(energy_0)  # (B, group_num0, adapt_dim)
        ctx_1 = self.energy_mlp1(energy_1)  # (B, group_num1, adapt_dim)

        # broadcast context to point-level and restore order
        ctx_0_spatial = sort(
            ctx_0.unsqueeze(2).expand(B0, group_num0, group_size0, self.adapt_dim).reshape(B0, G0, self.adapt_dim),
            idx[1]
        )
        ctx_1_spatial = sort(
            ctx_1.unsqueeze(2).expand(B1, group_num1, group_size1, self.adapt_dim).reshape(B1, G1, self.adapt_dim),
            idx[1]
        )

        # 5) group logits -> point-level by broadcast
        group_logit0 = self.group_mlp0(ctx_0)  # (B, group_num0, 1)
        group_logit1 = self.group_mlp1(ctx_1)  # (B, group_num1, 1)
        group_logit0 = sort(
            group_logit0.unsqueeze(2).expand(B0, group_num0, group_size0, 1).reshape(B0, G0, 1),
            idx[1]
        )
        group_logit1 = sort(
            group_logit1.unsqueeze(2).expand(B1, group_num1, group_size1, 1).reshape(B1, G1, 1),
            idx[1]
        )

        # 6) point residual logits with context
        delta_input0 = torch.cat([sub_x0_restored, ctx_0_spatial], dim=-1)
        delta_input1 = torch.cat([sub_x1_restored, ctx_1_spatial], dim=-1)
        delta_logit0 = torch.tanh(self.refine_mlp0(delta_input0)) * self.point_delta_scale
        delta_logit1 = torch.tanh(self.refine_mlp1(delta_input1)) * self.point_delta_scale

        # 7) bounded + gated routing
        logit0 = torch.clamp(group_logit0 + delta_logit0, -self.logit_clip, self.logit_clip)
        logit1 = torch.clamp(group_logit1 + delta_logit1, -self.logit_clip, self.logit_clip)

        gate0 = 1.0 + self.gate_scale * torch.tanh(logit0)
        gate1 = 1.0 + self.gate_scale * torch.tanh(logit1)

        # monitor
        if not self.training:
            cls = self.__class__
            cls._g0_vals.append(gate0.detach().cpu().float().reshape(-1))
            cls._g1_vals.append(gate1.detach().cpu().float().reshape(-1))

        # 8) fuse (same alignment form)
        x = (gate0 * sub_x0_restored + gate1 * sub_x1_restored) * self.fuse_scale

        # 9) residual output
        h = x + h
        h = self.up(h)
        return h


class MSF_scalar(nn.Module):
    """
    Multi-scale Fusion (MSF) Module - 标量门控版本
    基于点-组联合门控，但输出标量门控（1维）而非通道级门控
    """
    def __init__(self, dim, adapt_dim):
        super().__init__()
        self.adapt_dim = adapt_dim

        self.norm_ly2 = nn.LayerNorm(adapt_dim)
        self.act = nn.SiLU()
        self.down = nn.Linear(dim, adapt_dim)
        self.up = nn.Linear(adapt_dim, dim)

        # 解耦的独立滤波器
        self.adapt16 = nn.Linear(adapt_dim, adapt_dim)
        self.adapt32 = nn.Linear(adapt_dim, adapt_dim)
        nn.init.zeros_(self.adapt16.weight)
        nn.init.zeros_(self.adapt16.bias)
        nn.init.zeros_(self.adapt32.weight)
        nn.init.zeros_(self.adapt32.bias)

        # 组级能量提取器
        self.energy_mlp0 = nn.Sequential(
            nn.Linear(adapt_dim, adapt_dim // 2),
            nn.GELU(),
            nn.Linear(adapt_dim // 2, adapt_dim),
        )
        self.energy_mlp1 = nn.Sequential(
            nn.Linear(adapt_dim, adapt_dim // 2),
            nn.GELU(),
            nn.Linear(adapt_dim // 2, adapt_dim),
        )

        # 跨尺度差异归一化
        self.norm_diff = nn.LayerNorm(adapt_dim)

        # 点级门控网络（标量输出）
        # 输入：点特征 + 组上下文 + 差异特征 = 3 * adapt_dim
        self.point_mlp0 = nn.Sequential(
            nn.Linear(adapt_dim * 3, adapt_dim),
            nn.GELU(),
            nn.Linear(adapt_dim, 1),  # 标量输出
        )
        self.point_mlp1 = nn.Sequential(
            nn.Linear(adapt_dim * 3, adapt_dim),
            nn.GELU(),
            nn.Linear(adapt_dim, 1),  # 标量输出
        )
        # 0初始化最后一层，保证初始状态与baseline等价
        nn.init.zeros_(self.point_mlp0[-1].weight)
        nn.init.zeros_(self.point_mlp0[-1].bias)
        nn.init.zeros_(self.point_mlp1[-1].weight)
        nn.init.zeros_(self.point_mlp1[-1].bias)

        self.drop_adapt1 = DropPath(0.)
        self.drop_adapt2 = DropPath(0.)
        self.drop_out = nn.Dropout(0.)

        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, input, sub_U, idx):
        h = self.down(input)
        h = self.act(h)
        x = h

        # ==================== 准备数据 ====================
        B0, group_num0, group_size0, _ = sub_U[0].shape
        G0 = group_num0 * group_size0
        B1, group_num1, group_size1, _ = sub_U[1].shape
        G1 = group_num1 * group_size1

        # 将原始特征按 Z-order 排序
        x_sorted = sort(x, idx[0])
        sub_x0 = x_sorted.reshape(B0, group_num0, group_size0, self.adapt_dim)
        sub_x1 = x_sorted.reshape(B1, group_num1, group_size1, self.adapt_dim)

        # ==================== 尺度 0 (16点组) 频谱处理 ====================
        sub_x_f0 = sub_U[0].transpose(-2, -1) @ sub_x0
        sub_h_f0 = sub_x_f0
        sub_x_f0 = self.norm_ly2(sub_x_f0)
        sub_x_f0 = sub_h_f0 + self.drop_adapt1(self.act(self.drop_out(self.adapt16(sub_x_f0))))
        sub_x0 = sub_U[0] @ sub_x_f0
        sub_x0 = sub_x0.reshape(B0, G0, self.adapt_dim)
        sub_x0_restored = sort(sub_x0, idx[1])

        # ==================== 尺度 1 (32点组) 频谱处理 ====================
        sub_x_f1 = sub_U[1].transpose(-2, -1) @ sub_x1
        sub_h_f1 = sub_x_f1
        sub_x_f1 = self.norm_ly2(sub_x_f1)
        sub_x_f1 = sub_h_f1 + self.drop_adapt2(self.act(self.drop_out(self.adapt32(sub_x_f1))))
        sub_x1 = sub_U[1] @ sub_x_f1
        sub_x1 = sub_x1.reshape(B1, G1, self.adapt_dim)
        sub_x1_restored = sort(sub_x1, idx[1])

        # ==================== 组级频谱能量提取 ====================
        energy_0 = torch.sqrt((sub_x_f0 ** 2).mean(dim=2) + 1e-6)
        energy_1 = torch.sqrt((sub_x_f1 ** 2).mean(dim=2) + 1e-6)
        ctx_0 = self.energy_mlp0(energy_0)
        ctx_1 = self.energy_mlp1(energy_1)

        # 广播到点级
        ctx_0_exp = ctx_0.unsqueeze(2).expand(B0, group_num0, group_size0, self.adapt_dim)
        ctx_0_spatial = sort(ctx_0_exp.reshape(B0, G0, self.adapt_dim), idx[1])
        ctx_1_exp = ctx_1.unsqueeze(2).expand(B1, group_num1, group_size1, self.adapt_dim)
        ctx_1_spatial = sort(ctx_1_exp.reshape(B1, G1, self.adapt_dim), idx[1])

        # ==================== 跨尺度冲突感知 ====================
        diff_feat = torch.abs(sub_x0_restored - sub_x1_restored)
        diff_feat = self.norm_diff(diff_feat)

        # ==================== 点-组联合门控（标量输出） ====================
        gate_input_0 = torch.cat([sub_x0_restored, ctx_0_spatial, diff_feat], dim=-1)
        gate_input_1 = torch.cat([sub_x1_restored, ctx_1_spatial, diff_feat], dim=-1)

        # 生成标量门控增量 (B, N, 1)
        delta_0_point = torch.tanh(self.point_mlp0(gate_input_0))
        delta_1_point = torch.tanh(self.point_mlp1(gate_input_1))

        # 映射至中心点 1.0，广播到所有通道
        gate_0_spatial = 1.0 + delta_0_point
        gate_1_spatial = 1.0 + delta_1_point

        # ==================== 融合 ====================
        x = gate_0_spatial * sub_x0_restored + gate_1_spatial * sub_x1_restored

        h = x + h
        h = self.up(h)
        return h


class MSF_scalar_nodiff(nn.Module):
    """
    Multi-scale Fusion (MSF) Module - 标量门控版本（无差异特征）
    与 MSF_scalar 相同，但移除了 diff_feat 输入

    门控监控：
      验证时每个 batch 的 gate_0/gate_1 值被累积到类变量中，
      验证轮结束后调用 flush_gate_stats() 打印整轮的统计量并清空缓冲。
      g0 对应 16点组（较高频），g1 对应 32点组（较低频）。
    """

    # 类级别 accumulator：跨 batch 收集门控值（验证时使用）
    _g0_vals: list = []
    _g1_vals: list = []

    @classmethod
    def flush_gate_stats(cls, logger=None):
        """打印本轮验证中 g0/g1 的统计量，并清空缓冲。"""
        if not cls._g0_vals:
            return
        g0 = torch.cat(cls._g0_vals).float()
        g1 = torch.cat(cls._g1_vals).float()
        msg = (
            f"[Gate Monitor] g0(16点高频): "
            f"mean={g0.mean():.4f}  min={g0.min():.4f}  "
            f"max={g0.max():.4f}  std={g0.std():.4f} | "
            f"g1(32点低频): "
            f"mean={g1.mean():.4f}  min={g1.min():.4f}  "
            f"max={g1.max():.4f}  std={g1.std():.4f}"
        )
        print_log(msg, logger=logger)
        cls._g0_vals.clear()
        cls._g1_vals.clear()

    def __init__(self, dim, adapt_dim):
        super().__init__()
        self.adapt_dim = adapt_dim

        self.norm_ly2 = nn.LayerNorm(adapt_dim)
        self.act = nn.SiLU()
        self.down = nn.Linear(dim, adapt_dim)
        self.up = nn.Linear(adapt_dim, dim)

        # 解耦的独立滤波器
        self.adapt16 = nn.Linear(adapt_dim, adapt_dim)
        self.adapt32 = nn.Linear(adapt_dim, adapt_dim)
        nn.init.zeros_(self.adapt16.weight)
        nn.init.zeros_(self.adapt16.bias)
        nn.init.zeros_(self.adapt32.weight)
        nn.init.zeros_(self.adapt32.bias)

        # 组级能量提取器
        self.energy_mlp0 = nn.Sequential(
            nn.Linear(adapt_dim, adapt_dim // 2),
            nn.GELU(),
            nn.Linear(adapt_dim // 2, adapt_dim),
        )
        self.energy_mlp1 = nn.Sequential(
            nn.Linear(adapt_dim, adapt_dim // 2),
            nn.GELU(),
            nn.Linear(adapt_dim // 2, adapt_dim),
        )

        # 点级门控网络（标量输出，无diff_feat，输入为2x）
        self.point_mlp0 = nn.Sequential(
            nn.Linear(adapt_dim * 2, adapt_dim),
            nn.GELU(),
            nn.Linear(adapt_dim, 1),  # 标量输出
        )
        self.point_mlp1 = nn.Sequential(
            nn.Linear(adapt_dim * 2, adapt_dim),
            nn.GELU(),
            nn.Linear(adapt_dim, 1),  # 标量输出
        )
        # 0初始化最后一层
        nn.init.zeros_(self.point_mlp0[-1].weight)
        nn.init.zeros_(self.point_mlp0[-1].bias)
        nn.init.zeros_(self.point_mlp1[-1].weight)
        nn.init.zeros_(self.point_mlp1[-1].bias)

        self.drop_adapt1 = DropPath(0.)
        self.drop_adapt2 = DropPath(0.)
        self.drop_out = nn.Dropout(0.)

        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, input, sub_U, idx):
        h = self.down(input)
        h = self.act(h)
        x = h

        # ==================== 准备数据 ====================
        B0, group_num0, group_size0, _ = sub_U[0].shape
        G0 = group_num0 * group_size0
        B1, group_num1, group_size1, _ = sub_U[1].shape
        G1 = group_num1 * group_size1

        # 将原始特征按 Z-order 排序
        x_sorted = sort(x, idx[0])
        sub_x0 = x_sorted.reshape(B0, group_num0, group_size0, self.adapt_dim)
        sub_x1 = x_sorted.reshape(B1, group_num1, group_size1, self.adapt_dim)

        # ==================== 尺度 0 (16点组) 频谱处理 ====================
        sub_x_f0 = sub_U[0].transpose(-2, -1) @ sub_x0
        sub_h_f0 = sub_x_f0
        sub_x_f0 = self.norm_ly2(sub_x_f0)
        sub_x_f0 = sub_h_f0 + self.drop_adapt1(self.act(self.drop_out(self.adapt16(sub_x_f0))))
        sub_x0 = sub_U[0] @ sub_x_f0
        sub_x0 = sub_x0.reshape(B0, G0, self.adapt_dim)
        sub_x0_restored = sort(sub_x0, idx[1])

        # ==================== 尺度 1 (32点组) 频谱处理 ====================
        sub_x_f1 = sub_U[1].transpose(-2, -1) @ sub_x1
        sub_h_f1 = sub_x_f1
        sub_x_f1 = self.norm_ly2(sub_x_f1)
        sub_x_f1 = sub_h_f1 + self.drop_adapt2(self.act(self.drop_out(self.adapt32(sub_x_f1))))
        sub_x1 = sub_U[1] @ sub_x_f1
        sub_x1 = sub_x1.reshape(B1, G1, self.adapt_dim)
        sub_x1_restored = sort(sub_x1, idx[1])

        # ==================== 组级频谱能量提取 ====================
        energy_0 = torch.sqrt((sub_x_f0 ** 2).mean(dim=2) + 1e-6)
        energy_1 = torch.sqrt((sub_x_f1 ** 2).mean(dim=2) + 1e-6)
        ctx_0 = self.energy_mlp0(energy_0)
        ctx_1 = self.energy_mlp1(energy_1)

        # 广播到点级
        ctx_0_exp = ctx_0.unsqueeze(2).expand(B0, group_num0, group_size0, self.adapt_dim)
        ctx_0_spatial = sort(ctx_0_exp.reshape(B0, G0, self.adapt_dim), idx[1])
        ctx_1_exp = ctx_1.unsqueeze(2).expand(B1, group_num1, group_size1, self.adapt_dim)
        ctx_1_spatial = sort(ctx_1_exp.reshape(B1, G1, self.adapt_dim), idx[1])

        # ==================== 点-组联合门控（标量输出，无diff_feat） ====================
        gate_input_0 = torch.cat([sub_x0_restored, ctx_0_spatial], dim=-1)
        gate_input_1 = torch.cat([sub_x1_restored, ctx_1_spatial], dim=-1)

        # 生成标量门控增量 (B, N, 1)
        delta_0_point = torch.tanh(self.point_mlp0(gate_input_0))
        delta_1_point = torch.tanh(self.point_mlp1(gate_input_1))

        # 映射至中心点 1.0
        gate_0_spatial = 1.0 + delta_0_point
        gate_1_spatial = 1.0 + delta_1_point

        # ==================== 融合 ====================
        x = gate_0_spatial * sub_x0_restored + gate_1_spatial * sub_x1_restored

        h = x + h
        h = self.up(h)
        return h


class MSF_scalar_convex(nn.Module):
    """
    Multi-scale Fusion (MSF) Module - 归一化凸组合版本 (Convex Combination)

    改进动机：
      MSF_scalar 采用双门控 gate_0, gate_1 ∈ [0,2] 独立浮动，理论上融合特征的
      总尺度（Total Feature Scale）可在 [0, 4] 之间震荡，存在"特征模长失控"隐患。
      为彻底锁定融合后的特征尺度，本版本引入凸组合约束：
        x = w * x0 + (1-w) * x1,  w = sigmoid(MLP([x0, x1, ctx0, ctx1]))
      此时 w + (1-w) = 1 恒成立，融合特征模长与原始特征始终在同一量级内，
      消除了解码器偏移量因尺度震荡而发散的风险。

    与 MSF_scalar 的关键区别：
      - 将独立的两路 tanh 门控替换为单路 sigmoid 凸组合权重 w
      - 门控 MLP 同时观察两个尺度的点特征 + 组能量上下文（对称设计），
        而非各自只看自己一侧（AI 原方案为单侧，此处修正为双侧）
      - 参数量从 2×MLP 降为 1×MLP（输入 4×adapt_dim → 1）

    初始等价性：
      zero-init 最后一层 → 初始 w = sigmoid(0) = 0.5 → x = 0.5*x0 + 0.5*x1
      （相比 MSF_scalar 初始 x = 1.0*x0 + 1.0*x1，初始幅度缩小一半，
       但由 residual h = x + h 和 up 的 zero-init 保障整体初始等价性）
    """
    def __init__(self, dim, adapt_dim):
        super().__init__()
        self.adapt_dim = adapt_dim

        self.norm_ly2 = nn.LayerNorm(adapt_dim)
        self.act = nn.SiLU()
        self.down = nn.Linear(dim, adapt_dim)
        self.up = nn.Linear(adapt_dim, dim)

        # 解耦的独立频谱滤波器（与 MSF_scalar 相同）
        self.adapt16 = nn.Linear(adapt_dim, adapt_dim)
        self.adapt32 = nn.Linear(adapt_dim, adapt_dim)
        nn.init.zeros_(self.adapt16.weight)
        nn.init.zeros_(self.adapt16.bias)
        nn.init.zeros_(self.adapt32.weight)
        nn.init.zeros_(self.adapt32.bias)

        # 组级频谱能量提取器（与 MSF_scalar 相同）
        self.energy_mlp0 = nn.Sequential(
            nn.Linear(adapt_dim, adapt_dim // 2),
            nn.GELU(),
            nn.Linear(adapt_dim // 2, adapt_dim),
        )
        self.energy_mlp1 = nn.Sequential(
            nn.Linear(adapt_dim, adapt_dim // 2),
            nn.GELU(),
            nn.Linear(adapt_dim // 2, adapt_dim),
        )

        # 凸组合权重网络：双侧对称输入 [x0, x1, ctx0, ctx1] = 4 * adapt_dim
        # 输出单标量 w，经 sigmoid 映射到 (0, 1)
        # zero-init 最后一层保证训练初始 w = 0.5（等权平均）
        self.convex_gate = nn.Sequential(
            nn.Linear(adapt_dim * 4, adapt_dim),
            nn.GELU(),
            nn.Linear(adapt_dim, 1),
        )
        nn.init.zeros_(self.convex_gate[-1].weight)
        nn.init.zeros_(self.convex_gate[-1].bias)

        self.drop_adapt1 = DropPath(0.)
        self.drop_adapt2 = DropPath(0.)
        self.drop_out = nn.Dropout(0.)

        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, input, sub_U, idx):
        h = self.down(input)
        h = self.act(h)
        x = h

        B0, group_num0, group_size0, _ = sub_U[0].shape
        G0 = group_num0 * group_size0
        B1, group_num1, group_size1, _ = sub_U[1].shape
        G1 = group_num1 * group_size1

        x_sorted = sort(x, idx[0])
        sub_x0 = x_sorted.reshape(B0, group_num0, group_size0, self.adapt_dim)
        sub_x1 = x_sorted.reshape(B1, group_num1, group_size1, self.adapt_dim)

        # ==================== 尺度 0 (16点组) 频谱处理 ====================
        sub_x_f0 = sub_U[0].transpose(-2, -1) @ sub_x0
        sub_h_f0 = sub_x_f0
        sub_x_f0 = self.norm_ly2(sub_x_f0)
        sub_x_f0 = sub_h_f0 + self.drop_adapt1(self.act(self.drop_out(self.adapt16(sub_x_f0))))
        sub_x0 = sub_U[0] @ sub_x_f0
        sub_x0 = sub_x0.reshape(B0, G0, self.adapt_dim)
        sub_x0_restored = sort(sub_x0, idx[1])

        # ==================== 尺度 1 (32点组) 频谱处理 ====================
        sub_x_f1 = sub_U[1].transpose(-2, -1) @ sub_x1
        sub_h_f1 = sub_x_f1
        sub_x_f1 = self.norm_ly2(sub_x_f1)
        sub_x_f1 = sub_h_f1 + self.drop_adapt2(self.act(self.drop_out(self.adapt32(sub_x_f1))))
        sub_x1 = sub_U[1] @ sub_x_f1
        sub_x1 = sub_x1.reshape(B1, G1, self.adapt_dim)
        sub_x1_restored = sort(sub_x1, idx[1])

        # ==================== 组级频谱能量提取 & 广播到点级 ====================
        energy_0 = torch.sqrt((sub_x_f0 ** 2).mean(dim=2) + 1e-6)
        energy_1 = torch.sqrt((sub_x_f1 ** 2).mean(dim=2) + 1e-6)
        ctx_0 = self.energy_mlp0(energy_0)
        ctx_1 = self.energy_mlp1(energy_1)

        ctx_0_exp = ctx_0.unsqueeze(2).expand(B0, group_num0, group_size0, self.adapt_dim)
        ctx_0_spatial = sort(ctx_0_exp.reshape(B0, G0, self.adapt_dim), idx[1])
        ctx_1_exp = ctx_1.unsqueeze(2).expand(B1, group_num1, group_size1, self.adapt_dim)
        ctx_1_spatial = sort(ctx_1_exp.reshape(B1, G1, self.adapt_dim), idx[1])

        # ==================== 凸组合融合 ====================
        # 双侧对称拼接：同时感知两个尺度的点特征和频谱能量上下文
        # 相比 MSF_scalar 的双路独立 tanh 门控，此处强制 w + (1-w) = 1，
        # 保证融合后特征模长始终在合理范围内，避免解码器偏移量发散
        gate_input = torch.cat([
            sub_x0_restored, sub_x1_restored,
            ctx_0_spatial,   ctx_1_spatial,
        ], dim=-1)  # (B, N, 4 * adapt_dim)

        w = torch.sigmoid(self.convex_gate(gate_input))  # (B, N, 1)，∈ (0, 1)
        x = w * sub_x0_restored + (1.0 - w) * sub_x1_restored

        h = x + h
        h = self.up(h)
        return h


class _MSF_pure_GroupBase(nn.Module):
    """
    MSF 纯组级通道门控共享骨架：组能量 -> group_mlp -> 广播到点级 -> 子类门控 -> 融合。
    验证轮结束后调用 flush_gate_stats() 打印整轮统计量并清空缓冲。
    """
    gate_monitor_label = 'pure_group'
    _g0_vals = []
    _g1_vals = []

    @classmethod
    def flush_gate_stats(cls, logger=None):
        if not cls._g0_vals:
            return
        g0 = torch.cat(cls._g0_vals).float()
        g1 = torch.cat(cls._g1_vals).float()
        msg = (
            f"[Gate Monitor][{cls.gate_monitor_label}] g0 mean/min/max/std = "
            f"{g0.mean():.4f} {g0.min():.4f} {g0.max():.4f} {g0.std():.4f} | "
            f"g1 mean/min/max/std = "
            f"{g1.mean():.4f} {g1.min():.4f} {g1.max():.4f} {g1.std():.4f}"
        )
        print_log(msg, logger=logger)
        cls._g0_vals.clear()
        cls._g1_vals.clear()

    def __init__(self, dim, adapt_dim):
        super().__init__()
        self.dim = dim
        self.adapt_dim = adapt_dim
        self.eps = 1e-6
        self.gate_scale = 0.5

        self.norm_ly2 = nn.LayerNorm(adapt_dim)
        self.act = nn.SiLU()
        self.down = nn.Linear(dim, adapt_dim)
        self.up = nn.Linear(adapt_dim, dim)

        self.adapt16 = nn.Linear(adapt_dim, adapt_dim)
        self.adapt32 = nn.Linear(adapt_dim, adapt_dim)
        nn.init.zeros_(self.adapt16.weight)
        nn.init.zeros_(self.adapt16.bias)
        nn.init.zeros_(self.adapt32.weight)
        nn.init.zeros_(self.adapt32.bias)

        self.energy_mlp0 = nn.Sequential(
            nn.Linear(adapt_dim, adapt_dim // 2),
            nn.GELU(),
            nn.Linear(adapt_dim // 2, adapt_dim),
        )
        self.energy_mlp1 = nn.Sequential(
            nn.Linear(adapt_dim, adapt_dim // 2),
            nn.GELU(),
            nn.Linear(adapt_dim // 2, adapt_dim),
        )

        self.group_mlp0 = nn.Sequential(
            nn.Linear(adapt_dim, adapt_dim),
            nn.GELU(),
            nn.Linear(adapt_dim, adapt_dim),
        )
        self.group_mlp1 = nn.Sequential(
            nn.Linear(adapt_dim, adapt_dim),
            nn.GELU(),
            nn.Linear(adapt_dim, adapt_dim),
        )

        nn.init.zeros_(self.group_mlp0[-1].weight)
        nn.init.zeros_(self.group_mlp0[-1].bias)
        nn.init.zeros_(self.group_mlp1[-1].weight)
        nn.init.zeros_(self.group_mlp1[-1].bias)

        self.drop_adapt1 = DropPath(0.)
        self.drop_adapt2 = DropPath(0.)
        self.drop_out = nn.Dropout(0.)

        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

        self.logit_clip = 2.0
        self.export_route_to_decoder = False
        self._route_feat = None

    def _spatial_gates(self, logit0_spatial, logit1_spatial):
        raise NotImplementedError

    def _build_route_feat(
        self,
        sub_x0_res,
        sub_x1_res,
        logit0_spatial,
        logit1_spatial,
        energy_0,
        energy_1,
        B0,
        B1,
        group_num0,
        group_num1,
        group_size0,
        group_size1,
        G0,
        G1,
        idx,
    ):
        """Per-point MSF routing summary for decoder memory conditioning (dim=4)."""
        conflict = (sub_x0_res - sub_x1_res).abs().mean(dim=-1, keepdim=True)
        w = torch.sigmoid((logit0_spatial - logit1_spatial).mean(dim=-1, keepdim=True))
        # energy_* : (B, group_num, adapt_dim) -> per-group scalar then broadcast to points
        e0_group = energy_0.mean(dim=-1, keepdim=True)
        e1_group = energy_1.mean(dim=-1, keepdim=True)
        e0 = sort(
            e0_group.unsqueeze(2).expand(B0, group_num0, group_size0, 1).reshape(B0, G0, 1),
            idx[1],
        )
        e1 = sort(
            e1_group.unsqueeze(2).expand(B1, group_num1, group_size1, 1).reshape(B1, G1, 1),
            idx[1],
        )
        return torch.cat([w, conflict, e0, e1], dim=-1)

    def _apply_point_refine_logits(
        self,
        logit0_spatial,
        logit1_spatial,
        sub_x0_res,
        sub_x1_res,
        ctx_0,
        ctx_1,
        B0,
        B1,
        group_num0,
        group_num1,
        group_size0,
        group_size1,
        G0,
        G1,
        idx,
    ):
        if not hasattr(self, 'refine_mlp0'):
            return logit0_spatial, logit1_spatial

        ctx_0_spatial = sort(
            ctx_0.unsqueeze(2).expand(B0, group_num0, group_size0, self.adapt_dim).reshape(B0, G0, self.adapt_dim),
            idx[1],
        )
        ctx_1_spatial = sort(
            ctx_1.unsqueeze(2).expand(B1, group_num1, group_size1, self.adapt_dim).reshape(B1, G1, self.adapt_dim),
            idx[1],
        )

        delta0 = torch.tanh(self.refine_mlp0(torch.cat([sub_x0_res, ctx_0_spatial], dim=-1))) * self.point_delta_scale
        delta1 = torch.tanh(self.refine_mlp1(torch.cat([sub_x1_res, ctx_1_spatial], dim=-1))) * self.point_delta_scale

        logit0_spatial = torch.clamp(logit0_spatial + delta0, -self.logit_clip, self.logit_clip)
        logit1_spatial = torch.clamp(logit1_spatial + delta1, -self.logit_clip, self.logit_clip)

        if not self.training:
            cls = self.__class__
            if hasattr(cls, '_delta0_vals'):
                cls._delta0_vals.append(delta0.detach().cpu().float().reshape(-1))
                cls._delta1_vals.append(delta1.detach().cpu().float().reshape(-1))

        return logit0_spatial, logit1_spatial

    def forward(self, input, sub_U, idx):
        h_down = self.act(self.down(input))

        B0, group_num0, group_size0, _ = sub_U[0].shape
        G0 = group_num0 * group_size0
        B1, group_num1, group_size1, _ = sub_U[1].shape
        G1 = group_num1 * group_size1

        x_sorted = sort(h_down, idx[0])
        sub_x0 = x_sorted.reshape(B0, group_num0, group_size0, self.adapt_dim)
        sub_x1 = x_sorted.reshape(B1, group_num1, group_size1, self.adapt_dim)

        # Scale 16
        sub_x_f0 = sub_U[0].transpose(-2, -1) @ sub_x0
        sub_h_f0 = sub_x_f0
        sub_x_f0 = self.norm_ly2(sub_x_f0)
        sub_x_f0 = sub_h_f0 + self.drop_adapt1(self.act(self.drop_out(self.adapt16(sub_x_f0))))
        sub_x0_out = sub_U[0] @ sub_x_f0
        sub_x0_res = sort(sub_x0_out.reshape(B0, G0, self.adapt_dim), idx[1])

        # Scale 32
        sub_x_f1 = sub_U[1].transpose(-2, -1) @ sub_x1
        sub_h_f1 = sub_x_f1
        sub_x_f1 = self.norm_ly2(sub_x_f1)
        sub_x_f1 = sub_h_f1 + self.drop_adapt2(self.act(self.drop_out(self.adapt32(sub_x_f1))))
        sub_x1_out = sub_U[1] @ sub_x_f1
        sub_x1_res = sort(sub_x1_out.reshape(B1, G1, self.adapt_dim), idx[1])

        # 纯组级门控计算
        energy_0 = torch.sqrt((sub_x_f0 ** 2).mean(dim=2) + self.eps)
        energy_1 = torch.sqrt((sub_x_f1 ** 2).mean(dim=2) + self.eps)

        ctx_0 = self.energy_mlp0(energy_0)
        ctx_1 = self.energy_mlp1(energy_1)

        logit0 = torch.clamp(self.group_mlp0(ctx_0), -self.logit_clip, self.logit_clip)
        logit1 = torch.clamp(self.group_mlp1(ctx_1), -self.logit_clip, self.logit_clip)

        logit0_spatial = sort(
            logit0.unsqueeze(2).expand(B0, group_num0, group_size0, self.adapt_dim).reshape(B0, G0, self.adapt_dim),
            idx[1],
        )
        logit1_spatial = sort(
            logit1.unsqueeze(2).expand(B1, group_num1, group_size1, self.adapt_dim).reshape(B1, G1, self.adapt_dim),
            idx[1],
        )

        logit0_spatial, logit1_spatial = self._apply_point_refine_logits(
            logit0_spatial,
            logit1_spatial,
            sub_x0_res,
            sub_x1_res,
            ctx_0,
            ctx_1,
            B0,
            B1,
            group_num0,
            group_num1,
            group_size0,
            group_size1,
            G0,
            G1,
            idx,
        )

        g0_spatial, g1_spatial = self._spatial_gates(logit0_spatial, logit1_spatial)

        if not self.training:
            cls = self.__class__
            cls._g0_vals.append(g0_spatial.detach().cpu().float().reshape(-1))
            cls._g1_vals.append(g1_spatial.detach().cpu().float().reshape(-1))

        if getattr(self, '_export_vis', False):
            self._vis_cache = {
                'g0': g0_spatial.detach().float().cpu(),
                'g1': g1_spatial.detach().float().cpu(),
                'spec_energy_0': (sub_x_f0.pow(2).mean(dim=-1)).detach().float().cpu(),
                'spec_energy_1': (sub_x_f1.pow(2).mean(dim=-1)).detach().float().cpu(),
            }

        x_fused = g0_spatial * sub_x0_res + g1_spatial * sub_x1_res
        h_residual = x_fused + h_down

        if self.export_route_to_decoder:
            self._route_feat = self._build_route_feat(
                sub_x0_res,
                sub_x1_res,
                logit0_spatial,
                logit1_spatial,
                energy_0,
                energy_1,
                B0,
                B1,
                group_num0,
                group_num1,
                group_size0,
                group_size1,
                G0,
                G1,
                idx,
            )

        return self.up(h_residual)


class MSF_pure_Group(_MSF_pure_GroupBase):
    """
    Multi-scale Fusion (MSF) - 纯组级通道门控 + Softmax 竞争路由
    g = 1 + (softmax - 0.5)，每通道 g0 + g1 = 2
    """
    gate_monitor_label = 'softmax'
    _g0_vals = []
    _g1_vals = []

    def _spatial_gates(self, logit0_spatial, logit1_spatial):
        gate = torch.softmax(torch.stack([logit0_spatial, logit1_spatial], dim=-2), dim=-2)
        g0_spatial = 1.0 + (gate[:, :, 0, :] - 0.5)
        g1_spatial = 1.0 + (gate[:, :, 1, :] - 0.5)
        return g0_spatial, g1_spatial


class MSF_pure_Group_tanh(_MSF_pure_GroupBase):
    """纯组级通道门控 + 独立 Tanh 路由：g_i = 1 + alpha * tanh(l_i)"""
    gate_monitor_label = 'tanh'
    _g0_vals = []
    _g1_vals = []

    def _spatial_gates(self, logit0_spatial, logit1_spatial):
        g0_spatial = 1.0 + self.gate_scale * torch.tanh(logit0_spatial)
        g1_spatial = 1.0 + self.gate_scale * torch.tanh(logit1_spatial)
        return g0_spatial, g1_spatial


class MSF_pure_Group_sigmoid(_MSF_pure_GroupBase):
    """纯组级通道门控 + 独立 Sigmoid 对称路由：g_i = 1 + alpha * (2*sigmoid(l_i) - 1)"""
    gate_monitor_label = 'sigmoid'
    _g0_vals = []
    _g1_vals = []

    def _spatial_gates(self, logit0_spatial, logit1_spatial):
        g0_spatial = 1.0 + self.gate_scale * (2.0 * torch.sigmoid(logit0_spatial) - 1.0)
        g1_spatial = 1.0 + self.gate_scale * (2.0 * torch.sigmoid(logit1_spatial) - 1.0)
        return g0_spatial, g1_spatial


class MSF_pure_Group_sigmoid_point(_MSF_pure_GroupBase):
    """
    Sigmoid 组级通道门控 + 轻量点级通道残差。
    logit = logit_group_broadcast + point_delta_scale * tanh(refine_mlp([x_point, ctx_point]))
    refine 末层零初始化；point_delta_scale 较小，避免破坏组内门控一致性。
    """
    gate_monitor_label = 'sigmoid_point'
    _g0_vals = []
    _g1_vals = []
    _delta0_vals = []
    _delta1_vals = []
    point_delta_scale = 0.05

    @classmethod
    def flush_gate_stats(cls, logger=None):
        super().flush_gate_stats(logger=logger)
        if not cls._delta0_vals:
            return
        d0 = torch.cat(cls._delta0_vals).float()
        d1 = torch.cat(cls._delta1_vals).float()
        msg = (
            f"[Gate Monitor][{cls.gate_monitor_label}] delta0 mean/std = "
            f"{d0.mean():.4f} {d0.std():.4f} | delta1 mean/std = "
            f"{d1.mean():.4f} {d1.std():.4f} | point_delta_scale = {cls.point_delta_scale}"
        )
        print_log(msg, logger=logger)
        cls._delta0_vals.clear()
        cls._delta1_vals.clear()

    def __init__(self, dim, adapt_dim):
        super().__init__(dim, adapt_dim)
        self.refine_mlp0 = nn.Sequential(
            nn.Linear(adapt_dim * 2, adapt_dim),
            nn.GELU(),
            nn.Linear(adapt_dim, adapt_dim),
        )
        self.refine_mlp1 = nn.Sequential(
            nn.Linear(adapt_dim * 2, adapt_dim),
            nn.GELU(),
            nn.Linear(adapt_dim, adapt_dim),
        )
        nn.init.zeros_(self.refine_mlp0[-1].weight)
        nn.init.zeros_(self.refine_mlp0[-1].bias)
        nn.init.zeros_(self.refine_mlp1[-1].weight)
        nn.init.zeros_(self.refine_mlp1[-1].bias)

    def _spatial_gates(self, logit0_spatial, logit1_spatial):
        g0_spatial = 1.0 + self.gate_scale * (2.0 * torch.sigmoid(logit0_spatial) - 1.0)
        g1_spatial = 1.0 + self.gate_scale * (2.0 * torch.sigmoid(logit1_spatial) - 1.0)
        return g0_spatial, g1_spatial


class SelfAttnBlockApi(nn.Module):
    r'''
        1. Norm Encoder Block 
            block_style = 'attn'
        2. Concatenation Fused Encoder Block
            block_style = 'attn-deform'  
            combine_style = 'concat'
        3. Three-layer Fused Encoder Block
            block_style = 'attn-deform'  
            combine_style = 'onebyone'        
    '''
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, block_style='attn-deform', combine_style='concat',
            k=10, n_group=2, adapter_mode=None
        ):

        super().__init__()
        self.combine_style = combine_style
        assert combine_style in ['concat', 'onebyone'], f'got unexpect combine_style {combine_style} for local and global attn'
        self.norm1 = norm_layer(dim)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # 根据 adapter_mode 选择对应的适配器
        if adapter_mode == 'pcsa':
            self.gft_adapter = PCSA(dim, 36)
        elif adapter_mode == 'msf_scalar':
            self.gft_adapter = MSF_scalar(dim, 36)
        elif adapter_mode == 'msf_scalar_nodiff':
            self.gft_adapter = MSF_scalar_nodiff(dim, 36)
        elif adapter_mode == 'msf_scalar_convex':
            self.gft_adapter = MSF_scalar_convex(dim, 36)
        elif adapter_mode == 'msf_scalar_group_refined_v2_final':
            self.gft_adapter = MSF_scalar_group_refined_v2_final(dim, 36)
        elif adapter_mode == 'msf_scalar_group_refined_v2_tanh':
            self.gft_adapter = MSF_scalar_group_refined_v2_tanh(dim, 36)
        elif adapter_mode == 'msf_pure_group':
            self.gft_adapter = MSF_pure_Group(dim, 36)
        elif adapter_mode == 'msf_pure_group_tanh':
            self.gft_adapter = MSF_pure_Group_tanh(dim, 36)
        elif adapter_mode == 'msf_pure_group_sigmoid':
            self.gft_adapter = MSF_pure_Group_sigmoid(dim, 36)
        elif adapter_mode == 'msf_pure_group_sigmoid_point':
            self.gft_adapter = MSF_pure_Group_sigmoid_point(dim, 36)
        else:
            # 默认使用原始 MSF（组级门控）
            self.gft_adapter = MSF(dim, 36)

        # Api desigin
        block_tokens = block_style.split('-')
        assert len(block_tokens) > 0 and len(block_tokens) <= 2, f'invalid block_style {block_style}'
        self.block_length = len(block_tokens)
        self.attn = None
        self.local_attn = None
        for block_token in block_tokens:
            assert block_token in ['attn', 'rw_deform', 'deform', 'graph', 'deform_graph'], f'got unexpect block_token {block_token} for Block component'
            if block_token == 'attn':
                self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
            elif block_token == 'rw_deform':
                self.local_attn = DeformableLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group)
            elif block_token == 'deform':
                self.local_attn = DeformableLocalCrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group)
            elif block_token == 'graph':
                self.local_attn = DynamicGraphAttention(dim, k=k)
            elif block_token == 'deform_graph':
                self.local_attn = improvedDeformableLocalGraphAttention(dim, k=k)
        if self.attn is not None and self.local_attn is not None:
            if combine_style == 'concat':
                self.merge_map = nn.Linear(dim*2, dim)
            else:
                self.norm3 = norm_layer(dim)
                self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
                self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, pos, idx=None, sub_U=None, idx_add=None):
        feature_list = []
        if self.block_length == 2:
            if self.combine_style == 'concat':
                norm_x = self.norm1(x)
                if self.attn is not None:
                    global_attn_feat = self.attn(norm_x)
                    feature_list.append(global_attn_feat)
                if self.local_attn is not None:
                    local_attn_feat = self.local_attn(norm_x, pos, idx=idx)
                    feature_list.append(local_attn_feat)
                # combine
                if len(feature_list) == 2:
                    f = torch.cat(feature_list, dim=-1)
                    f = self.merge_map(f)
                    x = x + self.drop_path1(self.ls1(f))
                else:
                    raise RuntimeError()
            else: # onebyone
                x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
                x = x + self.drop_path3(self.ls3(self.local_attn(self.norm3(x), pos, idx=idx)))

        elif self.block_length == 1:
            norm_x = self.norm1(x)
            if self.attn is not None:
                global_attn_feat = self.attn(norm_x)
                feature_list.append(global_attn_feat)
            if self.local_attn is not None:
                local_attn_feat = self.local_attn(norm_x, pos, idx=idx)
                feature_list.append(local_attn_feat)
            # combine
            if len(feature_list) == 1:
                f = feature_list[0]
                x = x + self.drop_path1(self.ls1(f))
            else:
                raise RuntimeError()

        t=self.gft_adapter(x,sub_U,idx_add)
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        x+=t
        return x
   
class CrossAttnBlockApi(nn.Module):
    r'''
        1. Norm Decoder Block 
            self_attn_block_style = 'attn'
            cross_attn_block_style = 'attn'
        2. Concatenation Fused Decoder Block
            self_attn_block_style = 'attn-deform'  
            self_attn_combine_style = 'concat'
            cross_attn_block_style = 'attn-deform'  
            cross_attn_combine_style = 'concat'
        3. Three-layer Fused Decoder Block
            self_attn_block_style = 'attn-deform'  
            self_attn_combine_style = 'onebyone'
            cross_attn_block_style = 'attn-deform'  
            cross_attn_combine_style = 'onebyone'    
        4. Design by yourself
            #  only deform the cross attn
            self_attn_block_style = 'attn'  
            cross_attn_block_style = 'attn-deform'  
            cross_attn_combine_style = 'concat'    
            #  perform graph conv on self attn
            self_attn_block_style = 'attn-graph'  
            self_attn_combine_style = 'concat'    
            cross_attn_block_style = 'attn-deform'  
            cross_attn_combine_style = 'concat'    
    '''
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
            self_attn_block_style='attn-deform', self_attn_combine_style='concat',
            cross_attn_block_style='attn-deform', cross_attn_combine_style='concat',
            k=10, n_group=2
        ):
        super().__init__()        
        self.norm2 = norm_layer(dim)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()      

        # Api desigin
        # first we deal with self-attn
        self.norm1 = norm_layer(dim)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.self_attn_combine_style = self_attn_combine_style
        assert self_attn_combine_style in ['concat', 'onebyone'], f'got unexpect self_attn_combine_style {self_attn_combine_style} for local and global attn'
  
        self_attn_block_tokens = self_attn_block_style.split('-')
        assert len(self_attn_block_tokens) > 0 and len(self_attn_block_tokens) <= 2, f'invalid self_attn_block_style {self_attn_block_style}'
        self.self_attn_block_length = len(self_attn_block_tokens)
        self.self_attn = None
        self.local_self_attn = None
        for self_attn_block_token in self_attn_block_tokens:
            assert self_attn_block_token in ['attn', 'rw_deform', 'deform', 'graph', 'deform_graph'], f'got unexpect self_attn_block_token {self_attn_block_token} for Block component'
            if self_attn_block_token == 'attn':
                self.self_attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
            elif self_attn_block_token == 'rw_deform':
                self.local_self_attn = DeformableLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group)
            elif self_attn_block_token == 'deform':
                self.local_self_attn = DeformableLocalCrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group)
            elif self_attn_block_token == 'graph':
                self.local_self_attn = DynamicGraphAttention(dim, k=k)
            elif self_attn_block_token == 'deform_graph':
                self.local_self_attn = improvedDeformableLocalGraphAttention(dim, k=k)
        if self.self_attn is not None and self.local_self_attn is not None:
            if self_attn_combine_style == 'concat':
                self.self_attn_merge_map = nn.Linear(dim*2, dim)
            else:
                self.norm3 = norm_layer(dim)
                self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
                self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Then we deal with cross-attn
        self.norm_q = norm_layer(dim)
        self.norm_v = norm_layer(dim)
        self.ls4 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path4 = DropPath(drop_path) if drop_path > 0. else nn.Identity()  

        self.cross_attn_combine_style = cross_attn_combine_style
        assert cross_attn_combine_style in ['concat', 'onebyone'], f'got unexpect cross_attn_combine_style {cross_attn_combine_style} for local and global attn'
        
        # Api desigin
        cross_attn_block_tokens = cross_attn_block_style.split('-')
        assert len(cross_attn_block_tokens) > 0 and len(cross_attn_block_tokens) <= 2, f'invalid cross_attn_block_style {cross_attn_block_style}'
        self.cross_attn_block_length = len(cross_attn_block_tokens)
        self.cross_attn = None
        self.local_cross_attn = None
        for cross_attn_block_token in cross_attn_block_tokens:
            assert cross_attn_block_token in ['attn', 'deform', 'graph', 'deform_graph'], f'got unexpect cross_attn_block_token {cross_attn_block_token} for Block component'
            if cross_attn_block_token == 'attn':
                self.cross_attn = CrossAttention(dim, dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
            elif cross_attn_block_token == 'deform':
                self.local_cross_attn = DeformableLocalCrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group)
            elif cross_attn_block_token == 'graph':
                self.local_cross_attn = DynamicGraphAttention(dim, k=k)
            elif cross_attn_block_token == 'deform_graph':
                self.local_cross_attn = improvedDeformableLocalGraphAttention(dim, k=k)
        if self.cross_attn is not None and self.local_cross_attn is not None:
            if cross_attn_combine_style == 'concat':
                self.cross_attn_merge_map = nn.Linear(dim*2, dim)
            else:
                self.norm_q_2 = norm_layer(dim)
                self.norm_v_2 = norm_layer(dim)
                self.ls5 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
                self.drop_path5 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, q, v, q_pos, v_pos, self_attn_idx=None, cross_attn_idx=None, denoise_length=None):
        # q = q + self.drop_path(self.self_attn(self.norm1(q)))

        # calculate mask, shape N,N
        # 1 for mask, 0 for not mask
        # mask shape N, N
        # q: [ true_query; denoise_token ]
        if denoise_length is None:
            mask = None
        else:
            query_len = q.size(1)
            mask = torch.zeros(query_len, query_len).to(q.device)
            mask[:-denoise_length, -denoise_length:] = 1.

        # Self attn
        feature_list = []
        if self.self_attn_block_length == 2:
            if self.self_attn_combine_style == 'concat':
                norm_q = self.norm1(q)
                if self.self_attn is not None:
                    global_attn_feat = self.self_attn(norm_q, mask=mask)
                    feature_list.append(global_attn_feat)
                if self.local_self_attn is not None:
                    local_attn_feat = self.local_self_attn(norm_q, q_pos, idx=self_attn_idx, denoise_length=denoise_length)
                    feature_list.append(local_attn_feat)
                # combine
                if len(feature_list) == 2:
                    f = torch.cat(feature_list, dim=-1)
                    f = self.self_attn_merge_map(f)
                    q = q + self.drop_path1(self.ls1(f))
                else:
                    raise RuntimeError()
            else: # onebyone
                q = q + self.drop_path1(self.ls1(self.self_attn(self.norm1(q), mask=mask)))
                q = q + self.drop_path3(self.ls3(self.local_self_attn(self.norm3(q), q_pos, idx=self_attn_idx, denoise_length=denoise_length)))

        elif self.self_attn_block_length == 1:
            norm_q = self.norm1(q)
            if self.self_attn is not None:
                global_attn_feat = self.self_attn(norm_q, mask=mask)
                feature_list.append(global_attn_feat)
            if self.local_self_attn is not None:
                local_attn_feat = self.local_self_attn(norm_q, q_pos, idx=self_attn_idx, denoise_length=denoise_length)
                feature_list.append(local_attn_feat)
            # combine
            if len(feature_list) == 1:
                f = feature_list[0]
                q = q + self.drop_path1(self.ls1(f))
            else:
                raise RuntimeError()

        # q = q + self.drop_path(self.attn(self.norm_q(q), self.norm_v(v)))
        # Cross attn
        feature_list = []
        if self.cross_attn_block_length == 2:
            if self.cross_attn_combine_style == 'concat':
                norm_q = self.norm_q(q)
                norm_v = self.norm_v(v)
                if self.cross_attn is not None:
                    global_attn_feat = self.cross_attn(norm_q, norm_v)
                    feature_list.append(global_attn_feat)
                if self.local_cross_attn is not None:
                    local_attn_feat = self.local_cross_attn(q=norm_q, v=norm_v, q_pos=q_pos, v_pos=v_pos, idx=cross_attn_idx)
                    feature_list.append(local_attn_feat)
                # combine
                if len(feature_list) == 2:
                    f = torch.cat(feature_list, dim=-1)
                    f = self.cross_attn_merge_map(f)
                    q = q + self.drop_path4(self.ls4(f))
                else:
                    raise RuntimeError()
            else: # onebyone
                q = q + self.drop_path4(self.ls4(self.cross_attn(self.norm_q(q), self.norm_v(v))))
                q = q + self.drop_path5(self.ls5(self.local_cross_attn(q=self.norm_q_2(q), v=self.norm_v_2(v), q_pos=q_pos, v_pos=v_pos, idx=cross_attn_idx)))

        elif self.cross_attn_block_length == 1:
            norm_q = self.norm_q(q)
            norm_v = self.norm_v(v)
            if self.cross_attn is not None:
                global_attn_feat = self.cross_attn(norm_q, norm_v)
                feature_list.append(global_attn_feat)
            if self.local_cross_attn is not None:
                local_attn_feat = self.local_cross_attn(q=norm_q, v=norm_v, q_pos=q_pos, v_pos=v_pos, idx=cross_attn_idx)
                feature_list.append(local_attn_feat)
            # combine
            if len(feature_list) == 1:
                f = feature_list[0]
                q = q + self.drop_path4(self.ls4(f))
            else:
                raise RuntimeError()

        q = q + self.drop_path2(self.ls2(self.mlp(self.norm2(q))))
        return q
######################################## Entry ########################################  

class TransformerEncoder(nn.Module):
    """ Transformer Encoder without hierarchical structure
    """
    def __init__(self, embed_dim=256, depth=4, num_heads=4, mlp_ratio=4., qkv_bias=False, init_values=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
        block_style_list=['attn-deform'], combine_style='concat', k=10, n_group=2, adapter_mode=None):
        super().__init__()
        self.k = k
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(SelfAttnBlockApi(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                act_layer=act_layer, norm_layer=norm_layer,
                block_style=block_style_list[i], combine_style=combine_style, k=k, n_group=n_group,
                adapter_mode=adapter_mode
            ))

    def forward(self, x, pos, sub_U, idx_add):
        idx = idx = knn_point(self.k, pos, pos)
        for _, block in enumerate(self.blocks):
            x = block(x, pos, idx=idx, sub_U=sub_U, idx_add=idx_add) 
        return x

class TransformerDecoder(nn.Module):
    """ Transformer Decoder without hierarchical structure
    """
    def __init__(self, embed_dim=256, depth=4, num_heads=4, mlp_ratio=4., qkv_bias=False, init_values=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
        self_attn_block_style_list=['attn-deform'], self_attn_combine_style='concat',
        cross_attn_block_style_list=['attn-deform'], cross_attn_combine_style='concat',
        k=10, n_group=2):
        super().__init__()
        self.k = k
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(CrossAttnBlockApi(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                act_layer=act_layer, norm_layer=norm_layer,
                self_attn_block_style=self_attn_block_style_list[i], self_attn_combine_style=self_attn_combine_style,
                cross_attn_block_style=cross_attn_block_style_list[i], cross_attn_combine_style=cross_attn_combine_style,
                k=k, n_group=n_group
            ))

    def forward(self, q, v, q_pos, v_pos, denoise_length=None):
        if denoise_length is None:
            self_attn_idx = knn_point(self.k, q_pos, q_pos)
        else:
            self_attn_idx = None
        cross_attn_idx = knn_point(self.k, v_pos, q_pos)
        for _, block in enumerate(self.blocks):
            q = block(q, v, q_pos, v_pos, self_attn_idx=self_attn_idx, cross_attn_idx=cross_attn_idx, denoise_length=denoise_length)
        return q

class PointTransformerEncoder(nn.Module):
    """ Vision Transformer for point cloud encoder/decoder
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Args:
        embed_dim (int): embedding dimension
        depth (int): depth of transformer
        num_heads (int): number of attention heads
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim
        qkv_bias (bool): enable bias for qkv if True
        init_values: (float): layer-scale init values
        drop_rate (float): dropout rate
        attn_drop_rate (float): attention dropout rate
        drop_path_rate (float): stochastic depth rate
        norm_layer: (nn.Module): normalization layer
        act_layer: (nn.Module): MLP activation layer
    """
    def __init__(
            self, embed_dim=256, depth=12, num_heads=4, mlp_ratio=4., qkv_bias=True, init_values=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            norm_layer=None, act_layer=None,
            block_style_list=['attn-deform'], combine_style='concat',
            k=10, n_group=2, adapter_mode=None
        ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        assert len(block_style_list) == depth
        self.blocks = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth = depth,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            init_values=init_values,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate = dpr,
            norm_layer=norm_layer,
            act_layer=act_layer,
            block_style_list=block_style_list,
            combine_style=combine_style,
            k=k,
            n_group=n_group,
            adapter_mode=adapter_mode)
        self.norm = norm_layer(embed_dim) 
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, sub_U, idx):
        x = self.blocks(x, pos, sub_U, idx)
        return x

class PointTransformerDecoder(nn.Module):
    """ Vision Transformer for point cloud encoder/decoder
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """
    def __init__(
            self, embed_dim=256, depth=12, num_heads=4, mlp_ratio=4., qkv_bias=True, init_values=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            norm_layer=None, act_layer=None,
            self_attn_block_style_list=['attn-deform'], self_attn_combine_style='concat',
            cross_attn_block_style_list=['attn-deform'], cross_attn_combine_style='concat',
            k=10, n_group=2
        ):
        """
        Args:
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            init_values: (float): layer-scale init values
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        assert len(self_attn_block_style_list) == len(cross_attn_block_style_list) == depth
        self.blocks = TransformerDecoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth = depth,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            init_values=init_values,
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate = dpr,
            norm_layer=norm_layer, 
            act_layer=act_layer,
            self_attn_block_style_list=self_attn_block_style_list, 
            self_attn_combine_style=self_attn_combine_style,
            cross_attn_block_style_list=cross_attn_block_style_list, 
            cross_attn_combine_style=cross_attn_combine_style,
            k=k, 
            n_group=n_group
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, q, v, q_pos, v_pos, denoise_length=None):
        q = self.blocks(q, v, q_pos, v_pos, denoise_length=denoise_length)
        return q

class PointTransformerEncoderEntry(PointTransformerEncoder):
    def __init__(self, config, **kwargs):
        super().__init__(**dict(config))

class PointTransformerDecoderEntry(PointTransformerDecoder):
    def __init__(self, config, **kwargs):
        super().__init__(**dict(config))

######################################## Grouper ########################################  
class DGCNN_Grouper(nn.Module):
    def __init__(self, k = 16):
        super().__init__()
        '''
        K has to be 16
        '''
        print('using group version 2')
        self.k = k
        # self.knn = KNN(k=k, transpose_mode=False)
        self.input_trans = nn.Conv1d(3, 8, 1)

        self.layer1 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 32),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 64),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 64),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 128),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )
        self.num_features = 128
    @staticmethod
    def fps_downsample(coor, x, num_group):
        xyz = coor.transpose(1, 2).contiguous() # b, n, 3
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)

        combined_x = torch.cat([coor, x], dim=1)

        new_combined_x = (
            pointnet2_utils.gather_operation(
                combined_x, fps_idx
            )
        )

        new_coor = new_combined_x[:, :3]
        new_x = new_combined_x[:, 3:]

        return new_coor, new_x

    def get_graph_feature(self, coor_q, x_q, coor_k, x_k):

        # coor: bs, 3, np, x: bs, c, np

        k = self.k
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            # _, idx = self.knn(coor_k, coor_q)  # bs k np
            idx = knn_point(k, coor_k.transpose(-1, -2).contiguous(), coor_q.transpose(-1, -2).contiguous()) # B G M
            idx = idx.transpose(-1, -2).contiguous()
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, x, num):
        '''
            INPUT:
                x : bs N 3
                num : list e.g.[1024, 512]
            ----------------------
            OUTPUT:

                coor bs N 3
                f    bs N C(128) 
        '''
        x = x.transpose(-1, -2).contiguous()

        coor = x
        f = self.input_trans(x)

        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer1(f)
        f = f.max(dim=-1, keepdim=False)[0]

        coor_q, f_q = self.fps_downsample(coor, f, num[0])
        f = self.get_graph_feature(coor_q, f_q, coor, f)
        f = self.layer2(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor = coor_q

        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer3(f)
        f = f.max(dim=-1, keepdim=False)[0]

        coor_q, f_q = self.fps_downsample(coor, f, num[1])
        f = self.get_graph_feature(coor_q, f_q, coor, f)
        f = self.layer4(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor = coor_q

        coor = coor.transpose(-1, -2).contiguous()
        f = f.transpose(-1, -2).contiguous()

        return coor, f

class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )
    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)

class SimpleEncoder(nn.Module):
    def __init__(self, k = 32, embed_dims=128):
        super().__init__()
        self.embedding = Encoder(embed_dims)
        self.group_size = k

        self.num_features = embed_dims

    def forward(self, xyz, n_group):
        # 2048 divide into 128 * 32, overlap is needed
        if isinstance(n_group, list):
            n_group = n_group[-1] 

        center = misc.fps(xyz, n_group) # B G 3
            
        assert center.size(1) == n_group, f'expect center to be B {n_group} 3, but got shape {center.shape}'
        
        batch_size, num_points, _ = xyz.shape
        # knn to get the neighborhood
        idx = knn_point(self.group_size, xyz, center)
        assert idx.size(1) == n_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, n_group, self.group_size, 3).contiguous()
            
        assert neighborhood.size(1) == n_group
        assert neighborhood.size(2) == self.group_size
            
        features = self.embedding(neighborhood) # B G C
        
        return center, features

######################################## Fold ########################################    
class Fold(nn.Module):
    def __init__(self, in_channel, step , hidden_dim=512):
        super().__init__()

        self.in_channel = in_channel
        self.step = step

        a = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(1, step).expand(step, step).reshape(1, -1)
        b = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(step, 1).expand(step, step).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).cuda()

        self.folding1 = nn.Sequential(
            nn.Conv1d(in_channel + 2, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

        self.folding2 = nn.Sequential(
            nn.Conv1d(in_channel + 3, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

    def forward(self, x):
        num_sample = self.step * self.step
        bs = x.size(0)
        features = x.view(bs, self.in_channel, 1).expand(bs, self.in_channel, num_sample)
        seed = self.folding_seed.view(1, 2, num_sample).expand(bs, 2, num_sample).to(x.device)

        x = torch.cat([seed, features], dim=1)
        fd1 = self.folding1(x)
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(x)

        return fd2

class SimpleRebuildFCLayer(nn.Module):
    def __init__(self, input_dims, step, hidden_dim=512):
        super().__init__()
        self.input_dims = input_dims
        self.step = step
        self.layer = Mlp(self.input_dims, hidden_dim, step * 3)

    def forward(self, rec_feature):
        '''
        Input BNC
        '''
        batch_size = rec_feature.size(0)
        g_feature = rec_feature.max(1)[0]
        token_feature = rec_feature
            
        patch_feature = torch.cat([
                g_feature.unsqueeze(1).expand(-1, token_feature.size(1), -1),
                token_feature
            ], dim = -1)
        rebuild_pc = self.layer(patch_feature).reshape(batch_size, -1, self.step , 3)
        assert rebuild_pc.size(1) == rec_feature.size(1)
        return rebuild_pc

def get_basis(center):
    L = torch.cdist(center, center)
    L = 1 / (L / torch.min(L[L > 0], dim=-1, keepdim=True).values + torch.eye(L.size(-1), device=L.device).unsqueeze(0))
    L = get_laplacian(L)
    _, U = torch.linalg.eigh(L)
    return U

def sort(pts: torch.Tensor, idx: torch.Tensor):
    return torch.gather(pts, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, pts.size(-1)))

def get_laplacian(adj_matrix, normalize=True):
    """
    Compute the graph Laplacian matrix.

    Args:
        adj_matrix (torch.Tensor): The adjacency matrix (batch_size, vertices, vertices).
        normalize (bool): Whether to compute the normalized Laplacian.

    Returns:
        torch.Tensor: The Laplacian matrix (batch_size, vertices, vertices).
    """
    if normalize:
        # Degree matrix: sum of rows
        D = torch.sum(adj_matrix, dim=-1)  # (batch_size, vertices)
        # Avoid division by zero by adding epsilon to D
        D_inv_sqrt = torch.rsqrt(D + 1e-6)  # Inverse square root
        D_inv_sqrt = torch.diag_embed(D_inv_sqrt)  # Batch-wise diagonal matrices
        # Normalized Laplacian
        L = torch.eye(adj_matrix.size(-1), device=adj_matrix.device) - \
            D_inv_sqrt @ adj_matrix @ D_inv_sqrt
    else:
        # Degree matrix
        D = torch.sum(adj_matrix, dim=-1)  # (batch_size, vertices)
        D = torch.diag_embed(D)  # Batch-wise diagonal matrices
        # Unnormalized Laplacian
        L = D - adj_matrix

    return L


def resolve_msf_route_mode(config):
    """MSF route injection mode: none | mem | rebuild | query | query_rebuild."""
    mode = getattr(config, 'msf_route_mode', None)
    if mode is None or str(mode).strip() == '':
        mode = 'mem' if bool(
            getattr(config, 'use_msf_route_to_decoder', getattr(config, 'use_msf_route_decoder', False))
        ) else 'none'
    return str(mode).lower()


def msf_route_should_export(mode):
    return mode in ('mem', 'rebuild', 'query', 'query_rebuild')


def pool_msf_route_by_knn(route_feat, coor, query_pos, k=3):
    """KNN pool per-query MSF route summary from encoder centers (B,M,4)."""
    idx = knn_point(k, coor, query_pos)
    grouped = index_points(route_feat, idx)
    return grouped.mean(dim=2)


######################################## PCTransformer ########################################   
class PCTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        encoder_config = config.encoder_config
        decoder_config = config.decoder_config
        self.center_num  = getattr(config, 'center_num', [512, 128])
        self.encoder_type = config.encoder_type
        assert self.encoder_type in ['graph', 'pn'], f'unexpected encoder_type {self.encoder_type}'

        in_chans = 3
        self.num_query = query_num = config.num_query
        global_feature_dim = config.global_feature_dim

        print_log(f'Transformer with config {config}', logger='MODEL')
        # base encoder
        if self.encoder_type == 'graph':
            self.grouper = DGCNN_Grouper(k = 16)
        else:
            self.grouper = SimpleEncoder(k = 32, embed_dims=512)
        self.pos_embed = nn.Sequential(
            nn.Linear(in_chans, 128),
            nn.GELU(),
            nn.Linear(128, encoder_config.embed_dim)
        )  
        self.input_proj = nn.Sequential(
            nn.Linear(self.grouper.num_features, 512),
            nn.GELU(),
            nn.Linear(512, encoder_config.embed_dim)
        )
        # Coarse Level 1 : Encoder
        self.encoder = PointTransformerEncoderEntry(encoder_config)

        self.increase_dim = nn.Sequential(
            nn.Linear(encoder_config.embed_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, global_feature_dim))

        # query generator
        self.coarse_pred = nn.Sequential(
            nn.Linear(global_feature_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 3 * query_num)
        )
        self.mlp_query = nn.Sequential(
            nn.Linear(global_feature_dim + 3, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Linear(1024, decoder_config.embed_dim)
        )
        # assert decoder_config.embed_dim == encoder_config.embed_dim
        if decoder_config.embed_dim == encoder_config.embed_dim:
            self.mem_link = nn.Identity()
        else:
            self.mem_link = nn.Linear(encoder_config.embed_dim, decoder_config.embed_dim)
        # Coarse Level 2 : Decoder
        self.decoder = PointTransformerDecoderEntry(decoder_config)
 
        self.query_ranking = nn.Sequential(
            nn.Linear(3, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.msf_route_mode = resolve_msf_route_mode(config)
        self.route_dim = int(getattr(config, 'msf_route_dim', 4))
        self.use_msf_route_decoder = self.msf_route_mode == 'mem'
        self.export_msf_route = msf_route_should_export(self.msf_route_mode)

        if self.use_msf_route_decoder:
            dec_dim = decoder_config.embed_dim
            self.route_proj = nn.Linear(self.route_dim, dec_dim)
            nn.init.zeros_(self.route_proj.weight)
            nn.init.zeros_(self.route_proj.bias)
            self.route_scale = nn.Parameter(torch.tensor(0.1))
        else:
            self.route_proj = None
            self.route_scale = None

        if self.export_msf_route:
            last_adapter = self.encoder.blocks.blocks[-1].gft_adapter
            if hasattr(last_adapter, 'export_route_to_decoder'):
                last_adapter.export_route_to_decoder = True
            else:
                print_log(
                    f'msf_route_mode={self.msf_route_mode} but last encoder block has no MSF gft_adapter; route disabled.',
                    logger='MODEL',
                )
                self.export_msf_route = False
                self.use_msf_route_decoder = False
                self.msf_route_mode = 'none'

        print_log(
            f'MSF route: mode={self.msf_route_mode}, export={self.export_msf_route}, mem_inject={self.use_msf_route_decoder}',
            logger='MODEL',
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _collect_msf_route_feat(self):
        if not self.export_msf_route:
            return None
        for block in reversed(self.encoder.blocks.blocks):
            adapter = getattr(block, 'gft_adapter', None)
            if adapter is not None and getattr(adapter, '_route_feat', None) is not None:
                return adapter._route_feat
        return None

    def _inject_msf_route_to_mem(self, mem):
        if not self.use_msf_route_decoder:
            return mem
        route_feat = self._collect_msf_route_feat()
        if route_feat is None:
            return mem
        if route_feat.shape[1] != mem.shape[1]:
            raise RuntimeError(
                f'MSF route_feat length {route_feat.shape[1]} != mem length {mem.shape[1]}'
            )
        delta = self.route_proj(route_feat.to(dtype=mem.dtype, device=mem.device))
        return mem + self.route_scale * delta

    def forward(self, xyz):
        bs = xyz.size(0)
        coor, f = self.grouper(xyz, self.center_num) # b n c
        pe =  self.pos_embed(coor)
        x = self.input_proj(f)

        # U = get_basis(coor)
        B, G, _ = coor.shape
        c = coor * 100
        key = xyz2key(c[:, :, 1], c[:, :, 0], c[:, :, 2])
        _, idx0 = torch.sort(key)
        _, idx1 = torch.sort(idx0)
        sub_center=sort(coor,idx0)
        sub_U0 = get_basis(sub_center.reshape(B * (G // 16), 16, 3)).reshape(B, G // 16, 16, 16)
        sub_U1 = get_basis(sub_center.reshape(B * (G // 32), 32, 3)).reshape(B, G // 32, 32, 32)

        x = self.encoder(x + pe, coor, [sub_U0, sub_U1], [idx0,idx1]) # b n c
        global_feature = self.increase_dim(x) # B N C

        global_feature = torch.max(global_feature, dim=1)[0]  # (B, C)

        coarse = self.coarse_pred(global_feature).reshape(bs, -1, 3)

        coarse_inp = misc.fps(xyz, self.num_query//2) # B 128 3
        coarse = torch.cat([coarse, coarse_inp], dim=1) # B 224+128 3?

        mem = self.mem_link(x)
        mem = self._inject_msf_route_to_mem(mem)

        # query selection
        query_ranking = self.query_ranking(coarse) # b n 1
        idx = torch.argsort(query_ranking, dim=1, descending=True) # b n 1
        coarse = torch.gather(coarse, 1, idx[:,:self.num_query].expand(-1, -1, coarse.size(-1)))

        if self.training:
            # add denoise task
            # first pick some point : 64?
            picked_points = misc.fps(xyz, 64)
            picked_points = misc.jitter_points(picked_points)
            coarse = torch.cat([coarse, picked_points], dim=1) # B 256+64 3?
            denoise_length = 64     

            # produce query
            q = self.mlp_query(
            torch.cat([
                global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1),
                coarse], dim = -1)) # b n c

            # forward decoder
            q = self.decoder(q=q, v=mem, q_pos=coarse, v_pos=coor, denoise_length=denoise_length)

            return q, coarse, denoise_length, self._collect_msf_route_feat(), coor

        else:
            # produce query
            q = self.mlp_query(
            torch.cat([
                global_feature.unsqueeze(1).expand(-1, coarse.size(1), -1),
                coarse], dim = -1)) # b n c
            
            # forward decoder
            q = self.decoder(q=q, v=mem, q_pos=coarse, v_pos=coor)

            return q, coarse, 0, self._collect_msf_route_feat(), coor

######################################## PoinTr ########################################  

@MODELS.register_module()
class AdaPoinTr_PGST(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.trans_dim = config.decoder_config.embed_dim
        self.num_query = config.num_query
        self.num_points = getattr(config, 'num_points', None)

        self.decoder_type = config.decoder_type
        assert self.decoder_type in ['fold', 'fc'], f'unexpected decoder_type {self.decoder_type}'

        self.fold_step = 8
        self.base_model = PCTransformer(config)

        self.msf_route_mode = resolve_msf_route_mode(config)
        self.msf_route_knn_k = int(getattr(config, 'msf_route_knn_k', 3))
        self.route_dim = int(getattr(config, 'msf_route_dim', 4))
        self.use_msf_rebuild_route = self.msf_route_mode in ('rebuild', 'query_rebuild')
        if self.use_msf_rebuild_route:
            self.rebuild_route_proj = nn.Sequential(
                nn.Linear(self.route_dim, self.trans_dim),
                nn.GELU(),
                nn.Linear(self.trans_dim, self.trans_dim),
            )
            nn.init.zeros_(self.rebuild_route_proj[-1].weight)
            nn.init.zeros_(self.rebuild_route_proj[-1].bias)
        else:
            self.rebuild_route_proj = None

        print_log(
            f'AdaPoinTr MSF rebuild route: mode={self.msf_route_mode}, '
            f'rebuild_cond={self.use_msf_rebuild_route}, knn_k={self.msf_route_knn_k}',
            logger='MODEL',
        )
        
        if self.decoder_type == 'fold':
            self.factor = self.fold_step**2
            self.decode_head = Fold(self.trans_dim, step=self.fold_step, hidden_dim=256)  # rebuild a cluster point
        else:
            if self.num_points is not None:
                self.factor = self.num_points // self.num_query
                assert self.num_points % self.num_query == 0
                self.decode_head = SimpleRebuildFCLayer(self.trans_dim * 2, step=self.num_points // self.num_query)  # rebuild a cluster point
            else:
                self.factor = self.fold_step**2
                self.decode_head = SimpleRebuildFCLayer(self.trans_dim * 2, step=self.fold_step**2)
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )
        self.reduce_map = nn.Linear(self.trans_dim + 1027, self.trans_dim)
        loss_cfg = getattr(config, 'loss_config', None)
        self.cover_weight = float(getattr(loss_cfg, 'cover_weight', 0.0) or 0.0) if loss_cfg else 0.0
        self.cover_critical_ratio = float(
            getattr(loss_cfg, 'cover_critical_ratio', 0.0) or 0.0
        ) if loss_cfg else 0.0
        self.build_loss_func()

    def build_loss_func(self):
        self.loss_func = ChamferDistanceL1()

    def _chamfer_loss(self, pred, gt):
        if self.cover_weight > 0:
            from utils.chamfer_loss_utils import chamfer_l1_with_cover
            return chamfer_l1_with_cover(
                pred,
                gt,
                cover_weight=self.cover_weight,
                cover_critical_ratio=self.cover_critical_ratio,
            )
        return self.loss_func(pred, gt)

    def _chamfer_loss_per_sample(self, pred, gt):
        from utils.chamfer_loss_utils import chamfer_l1_per_sample
        return chamfer_l1_per_sample(
            pred,
            gt,
            cover_weight=self.cover_weight,
            cover_critical_ratio=self.cover_critical_ratio,
        )

    def get_loss_per_sample(self, ret, gt, epoch=1):
        """Per-sample total / sparse / recon losses, each shape (B,). For batch OHEM."""
        pred_coarse, denoised_coarse, denoised_fine, pred_fine = ret[:4]
        assert pred_fine.size(1) == gt.size(1)

        idx = knn_point(self.factor, gt, denoised_coarse)
        denoised_target = index_points(gt, idx).reshape(gt.size(0), -1, 3)
        assert denoised_target.size(1) == denoised_fine.size(1)

        per_denoised = self._chamfer_loss_per_sample(denoised_fine, denoised_target) * 0.5
        per_coarse = self._chamfer_loss_per_sample(pred_coarse, gt)
        per_fine = self._chamfer_loss_per_sample(pred_fine, gt)
        per_recon = per_coarse + per_fine
        per_total = per_denoised + per_recon
        return per_total, per_denoised, per_recon

    def get_loss(self, ret, gt, epoch=1):
        pred_coarse, denoised_coarse, denoised_fine, pred_fine = ret[:4]
        
        assert pred_fine.size(1) == gt.size(1)

        # denoise loss
        idx = knn_point(self.factor, gt, denoised_coarse) # B n k 
        denoised_target = index_points(gt, idx) # B n k 3 
        denoised_target = denoised_target.reshape(gt.size(0), -1, 3)
        assert denoised_target.size(1) == denoised_fine.size(1)
        loss_denoised = self._chamfer_loss(denoised_fine, denoised_target)
        loss_denoised = loss_denoised * 0.5

        # recon loss
        loss_coarse = self._chamfer_loss(pred_coarse, gt)
        loss_fine = self._chamfer_loss(pred_fine, gt)

        loss_recon = loss_coarse + loss_fine

        return loss_denoised, loss_recon

    def _apply_msf_rebuild_route(self, q, route_feat, coor, query_pos):
        if not self.use_msf_rebuild_route or route_feat is None or self.rebuild_route_proj is None:
            return q
        s_local = pool_msf_route_by_knn(
            route_feat, coor, query_pos, k=self.msf_route_knn_k,
        )
        delta_q = self.rebuild_route_proj(s_local.to(dtype=q.dtype, device=q.device))
        return q + delta_q

    def forward(self, xyz):
        base_out = self.base_model(xyz)
        q, coarse_point_cloud, denoise_length = base_out[0], base_out[1], base_out[2]
        route_feat = base_out[3] if len(base_out) > 3 else None
        coor = base_out[4] if len(base_out) > 4 else None
        q = self._apply_msf_rebuild_route(q, route_feat, coor, coarse_point_cloud)
    
        B, M ,C = q.shape

        global_feature = self.increase_dim(q.transpose(1,2)).transpose(1,2) # B M 1024
        global_feature = torch.max(global_feature, dim=1)[0] # B 1024

        rebuild_feature = torch.cat([
            global_feature.unsqueeze(-2).expand(-1, M, -1),
            q,
            coarse_point_cloud], dim=-1)  # B M 1027 + C

        
        # NOTE: foldingNet
        if self.decoder_type == 'fold':
            rebuild_feature = self.reduce_map(rebuild_feature.reshape(B*M, -1)) # BM C
            relative_xyz = self.decode_head(rebuild_feature).reshape(B, M, 3, -1)    # B M 3 S
            rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-1)).transpose(2,3)  # B M S 3

        else:
            rebuild_feature = self.reduce_map(rebuild_feature) # B M C
            relative_xyz = self.decode_head(rebuild_feature)   # B M S 3
            rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-2))  # B M S 3

        if self.training:
            # split the reconstruction and denoise task
            pred_fine = rebuild_points[:, :-denoise_length].reshape(B, -1, 3).contiguous()
            pred_coarse = coarse_point_cloud[:, :-denoise_length].contiguous()

            denoised_fine = rebuild_points[:, -denoise_length:].reshape(B, -1, 3).contiguous()
            denoised_coarse = coarse_point_cloud[:, -denoise_length:].contiguous()

            assert pred_fine.size(1) == self.num_query * self.factor
            assert pred_coarse.size(1) == self.num_query

            ret = (pred_coarse, denoised_coarse, denoised_fine, pred_fine)
            return ret

        else:
            assert denoise_length == 0
            rebuild_points = rebuild_points.reshape(B, -1, 3).contiguous()  # B N 3

            assert rebuild_points.size(1) == self.num_query * self.factor
            assert coarse_point_cloud.size(1) == self.num_query

            ret = (coarse_point_cloud, rebuild_points)
            return ret