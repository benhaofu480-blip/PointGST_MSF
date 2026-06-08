"""
Loss Factory — 可扩展、可开闭的损失函数管理器。

设计目标：
1. 通过 config yaml 统一控制所有损失的启用/禁用/权重
2. 支持多种损失函数的对比实验（Laplacian, DCD, APML, Normal-Aware CD 等）
3. 新增损失只需写一个类并注册到 factory，不改 PGST.py 主逻辑
4. 支持 coarse/fine 阶段使用不同的损失函数（如 coarse=APML, fine=CD）

使用方式 (config.yaml):
    loss_config:
        base_loss: cd_l1              # fine阶段默认损失
        coarse_loss: apml              # coarse阶段专用损失 (可选, 不设则与base相同)
        auxiliary_losses:             # 辅助损失字典，按需开闭
            laplacian:
                enabled: true
                weight: 0.1
"""

import torch
import torch.nn as nn
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2


# ============================================================
# 注册表：所有可用损失类
# ============================================================

LOSS_REGISTRY = {}


def register_loss(name):
    """装饰器：将损失类注册到全局注册表"""
    def decorator(cls):
        LOSS_REGISTRY[name] = cls
        return cls
    return decorator


# ============================================================
# 基础损失（Base Losses）— 必选其一
# ============================================================

@register_loss('cd_l1')
class CD_L1(nn.Module):
    """Chamfer Distance L1 (标准基础损失)"""
    def __init__(self, **kwargs):
        super().__init__()
        self.cd = ChamferDistanceL1()

    def forward(self, pred, gt):
        return self.cd(pred, gt)


@register_loss('cd_l2')
class CD_L2(nn.Module):
    """Chamfer Distance L2"""
    def __init__(self, **kwargs):
        super().__init__()
        self.cd = ChamferDistanceL2()

    def forward(self, pred, gt):
        return self.cd(pred, gt)


# ============================================================
# 辅助损失（Auxiliary Losses）— 可选，可叠加
# ============================================================

@register_loss('laplacian')
class StochasticMicroLaplacianLoss(nn.Module):
    """
    随机微批拉普拉斯损失 (v5) — 替代 GroupedLaplacianLoss(v4)。

    核心改进:
    - Fine层: 随机采样破除 512×32 分组收缩 (v4的根本缺陷)
    - 使用真实 KNN 邻域替代固定分组质心
    - Micro-batch 1024 保证 L2 cache 命中率和无 FP16 溢出
    - MSF 门控豁免: 高频边缘区域惩罚减小
    - Coarse层: KNN 平滑骨架拓扑 (极轻量)

    参数:
        sample_size:      每次从 16384 点中随机抽取数 (2048=12.5%)
        block_size:       Micro-batch 大小 (1024)
        k:                Fine KNN 邻居数 (16)
        apply_coarse_lap: 是否开启 coarse 骨架平滑
        coarse_k:         Coarse KNN 邻居数 (8)
        coarse_alpha:     Coarse Laplacian 权重
    """
    def __init__(self, sample_size=2048, block_size=1024, k=16,
                 apply_coarse_lap=True, coarse_k=8, coarse_alpha=0.5, **kwargs):
        super().__init__()
        self.M = sample_size
        self.B_size = block_size
        self.k = k
        self.apply_coarse_lap = apply_coarse_lap
        self.coarse_k = coarse_k
        self.coarse_alpha = coarse_alpha

        assert self.M % self.B_size == 0, f"sample_size({self.M}) 必须是 block_size({self.B_size}) 的整数倍"
        self.num_blocks = self.M // self.B_size

    def forward(self, pred_fine, pred_coarse=None, gate_w1=None):
        B, N, _ = pred_fine.shape
        device = pred_fine.device

        # ==========================================
        # 1. 动态随机采样 + Micro-batch 切分 (Fine)
        # ==========================================
        rand_idx = torch.randint(0, N, (B, self.M), device=device)

        # 提取点云子集 [B, M, 3]
        idx_expanded_3d = rand_idx.unsqueeze(-1).expand(-1, -1, 3)
        sub_x = torch.gather(pred_fine, 1, idx_expanded_3d)

        # 同步提取门控权重 [B, M]
        use_gate = False
        if gate_w1 is not None and gate_w1.size(0) == B:
            gate_w1 = gate_w1.view(B, N, 1) if gate_w1.dim() >= 2 else gate_w1.unsqueeze(-1)
            if gate_w1.size(1) == N:
                idx_expanded_1d = rand_idx.unsqueeze(-1)
                sub_gate = torch.gather(gate_w1, 1, idx_expanded_1d)
                mb_gate = sub_gate.view(B * self.num_blocks, self.B_size)
                use_gate = True

        # 切分为 micro-batches [MB, 1024, 3]
        MB = B * self.num_blocks
        mb_x = sub_x.view(MB, self.B_size, 3)

        # ==========================================
        # 2. KNN + Laplacian 计算 (Fine)
        # ==========================================
        dist_matrix = torch.cdist(mb_x, mb_x)          # [MB, 1024, 1024]

        _, knn_idx = torch.topk(dist_matrix, self.k + 1, dim=-1, largest=False)
        knn_idx = knn_idx[:, :, 1:]                     # 排除自身 [MB, 1024, k]

        # 聚合邻居坐标 [MB, 1024, k, 3] → [MB, 1024, 3]
        idx_for_gather = knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        mb_x_expanded = mb_x.unsqueeze(1).expand(-1, self.B_size, -1, -1)
        neighbors = torch.gather(mb_x_expanded, 2, idx_for_gather)
        neighbors_mean = neighbors.mean(dim=2)

        # Laplacian: ||p_i - mean(KNN(p_i))||_2
        lap_norm_fine = (mb_x - neighbors_mean).norm(dim=-1)   # [MB, 1024]

        # MSF 边缘豁免: 高频(门控大)区域惩罚减小
        if use_gate:
            lap_norm_fine = lap_norm_fine * (1.0 - mb_gate.clamp(max=1.0))

        loss_fine_lap = lap_norm_fine.mean()

        # ==========================================
        # 3. 骨架拓扑约束 (Coarse, 极轻量)
        # ==========================================
        loss_coarse_lap = torch.tensor(0.0, device=device)
        if self.apply_coarse_lap and pred_coarse is not None:
            dist_coarse = torch.cdist(pred_coarse, pred_coarse)           # [B, 512, 512]
            _, coarse_knn_idx = torch.topk(dist_coarse, self.coarse_k + 1, dim=-1, largest=False)
            coarse_knn_idx = coarse_knn_idx[:, :, 1:]                    # [B, 512, coarse_k]

            c_idx_gather = coarse_knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
            p_coarse_expanded = pred_coarse.unsqueeze(1).expand(-1, pred_coarse.shape[1], -1, -1)
            c_neighbors = torch.gather(p_coarse_expanded, 2, c_idx_gather)
            c_neighbors_mean = c_neighbors.mean(dim=2)

            lap_norm_coarse = (pred_coarse - c_neighbors_mean).norm(dim=-1)
            loss_coarse_lap = lap_norm_coarse.mean()

        return loss_fine_lap + self.coarse_alpha * loss_coarse_lap


@register_loss('apml')
class APMLOSS(nn.Module):
    """
    Adaptive Probability Matching Loss (APML) — 稀疏 softmax + Sinkhorn。
    
    使用 CUDA sparse kernel 加速，避免构建 N×M 稠密距离矩阵。
    
    原理：
    - 对每对 (pred_i, gt_j) 计算稀疏 softmax 概率 P(i→j) 和 P(j→i)
    - 双向匹配后合并 → Sinkhorn 迭代归一化 (20轮)
    - 最终 loss = Σ P*(i,j) * ||p_i - q_j||_2
    
    Args:
        p_min: 最小概率集中度 (默认0.8, 即80%概率集中在最近邻附近)
        threshold: 稀疏化阈值 (softmax值<threshold的条目丢弃)
        use_cuda_sparse: 是否使用 CUDA sparse kernel (推荐True, 大矩阵加速显著)
    
    显存估算 (coarse: 512×16384, B=32):
    - 稀疏COO: ~2-4M entries × 12B ≈ 48MB
    - 排序/前缀和临时buffer: ~80MB
    - 总额外开销: ~150-200MB ← 24GB卡完全无压力
    
    参考: U-Pool "Unbiased Point Cloud Completion" (ICCV 2023)
    """

    def __init__(self, p_min=0.8, threshold=1e-8, use_cuda_sparse=True, **kwargs):
        super().__init__()
        self.p_min = float(p_min)
        self.threshold = float(threshold)  # 防御: YAML可能将1e-8解析为str
        self.use_cuda_sparse = use_cuda_sparse

        # 尝试加载 CUDA sparse kernel
        self.apml_sparse_fn = None
        if use_cuda_sparse:
            try:
                from extensions.apml_cuda import load_apml_sparse
                self.apml_sparse_fn = load_apml_sparse()
                print("[APML] CUDA sparse kernel loaded successfully")
            except Exception as e:
                print(f"[APML] CUDA sparse kernel 加载失败 ({e}), 回退到 PyTorch 实现")

    @torch.no_grad()
    def _get_nnz_estimate(self, N, M, B):
        """预估稀疏度，用于决定是否用 CUDA sparse 还是 dense fallback"""
        # 粗估: 每个 pred 点在 softmax 后约有 ~sqrt(M)*2 个非零条目
        # row方向: B*N*est, col方向: B*M*est_small
        est_per_point_row = max(5, int((M ** 0.35)))   # M=16384 → ~25
        est_per_point_col = max(3, int((N ** 0.5)))     # N=512   → ~22
        total_est = B * (N * est_per_point_row + M * est_per_point_col)
        return total_est

    def forward(self, pred, gt):
        """
        计算 APML loss。
        
        Args:
            pred: [B, N, 3] 预测点云 (如 coarse: N=512)
            gt:   [B, M, 3] GT点云 (如 GT: M=16384)
        
        Returns:
            loss: 标量
        """
        B, N, D = pred.shape
        M = gt.shape[1]

        # ---- 路径选择: CUDA Sparse vs PyTorch Dense ----
        if self.apml_sparse_fn is not None and self._should_use_sparse(N, M, B):
            return self._forward_cuda_sparse(pred, gt)
        else:
            if self.apml_sparse_fn is None or N * M > 500000:  # 大矩阵但没有CUDA
                print(f"[APML] Warning: N={N}×M={M}={N*M} 无CUDA sparse, 用dense fallback")
            return self._forward_pytorch_dense(pred, gt)

    def _should_use_sparse(self, N, M, B):
        """判断是否应该用 sparse 路径"""
        # 小矩阵直接用 dense 更快 (避免 kernel launch 开销)
        if N * M <= 100000:  # 如 256×256 或 512×128
            return False
        return True

    def _forward_cuda_sparse(self, pred, gt):
        """CUDA Sparse kernel 路径 (推荐, 用于大矩阵)"""
        B, N, D = pred.shape
        COO_i, COO_j, COO_v, loss = self.apml_sparse_fn.forward(
            pred.contiguous().float(),
            gt.contiguous().float(),
            self.p_min,
            self.threshold
        )
        # 归一化: 除以 N 使量级与 CD_L1 可比 (CUDA kernel 返回的是 sum，未除以 N)
        return loss / N

    def _forward_pytorch_dense(self, pred, gt):
        """
        PyTorch Dense Fallback — 用于小矩阵或 CUDA 不可用时。
        
        构建完整 N×M 距离矩阵 + softmax + Sinkhorn。
        仅用于小规模场景 (如 N,M < 1024)，大矩阵会 OOM!
        """
        B, N, D = pred.shape
        M = gt.shape[1]

        # 距离矩阵 [B, N, M]
        dist = torch.cdist(pred, gt)  # [B, N, M]

        # 行向 softmax: P(i→j)
        row_min = dist.min(dim=-1, keepdim=True)[0]  # [B, N, 1]
        gap = dist.sort(dim=-1)[0][:, :, 1] - row_min.squeeze(-1)  # [B, N] 2nd-min gap
        k = float(M)
        log_term = -torch.log(torch.tensor((1.0 - self.p_min) / (k - 1.0), device=pred.device))
        temperature = log_term / gap.clamp(min=1e-8)  # [B, N]

        rel_dist = (dist - row_min)  # [B, N, M]
        # 用逐样本温度广播
        P_row = torch.exp(-rel_dist * temperature.unsqueeze(-1))  # [B, N, M]
        P_row = torch.where(P_row > self.threshold, P_row, torch.zeros_like(P_row))
        P_row_sum = P_row.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        P_row = P_row / P_row_sum

        # 列向 softmax: P(j→i)
        dist_T = dist.transpose(1, 2)  # [B, M, N]
        col_min = dist_T.min(dim=-1, keepdim=True)[0]
        col_gap = dist_T.sort(dim=-1)[0][:, :, 1] - col_min.squeeze(-1)
        k_T = float(N)
        log_term_T = -torch.log(torch.tensor((1.0 - self.p_min) / (k_T - 1.0), device=pred.device))
        temp_T = log_term_T / col_gap.clamp(min=1e-8)

        rel_dist_T = (dist_T - col_min)
        P_col = torch.exp(-rel_dist_T * temp_T.unsqueeze(-1))
        P_col = torch.where(P_col > self.threshold, P_col, torch.zeros_like(P_col))
        P_col_sum = P_col.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        P_col = P_col / P_col_sum  # [B, M, N]

        # 合并双向匹配 (几何平均)
        P_match = (P_row * P_col.transpose(1, 2)).clamp(min=0)  # [B, N, M]

        # Sinkhorn 迭代归一化 (简化版: 交替行列归一化)
        for _ in range(20):
            P_match = P_match / (P_match.sum(dim=-1, keepdim=True).clamp(min=1e-8))
            P_match = P_match / (P_match.sum(dim=1, keepdim=True).clamp(min=1e-8))

        # 加权距离 — 除以 N 归一化，使量级与 CD_L1 (~3~10) 可比
        # 否则 sum(N*M) 对 512×16384 矩阵会产生 ~2e7 的值，淹没其他 loss
        weighted_dist = (P_match * dist).sum(dim=[1, 2])  # [B]
        return weighted_dist.mean() / N  # ← 关键: 除以 pred 点数


@register_loss('directional_cd')
class DirectionalChamferDistanceLoss(nn.Module):
    """
    原版 Directional Chamfer Distance (DCD) — 预留接口。
    
    参考: Xiang, et al. "DCD: Directional Chamfer Distance" 
    TODO: 实现时需要补充具体的 DCD 计算逻辑
    """
    def __init__(self, **kwargs):
        super().__init__()
        raise NotImplementedError("原版 DCD 尚未实现，等待接入")

    def forward(self, pred_fine, gt, **kwargs):
        raise NotImplementedError


@register_loss('normal_aware_cd')
class NormalAwareChamferDistanceLoss(nn.Module):
    """
    Normal-Aware Chamfer Distance — 预留接口。
    
    公式: CD_normal(p,q) = CD(p,q) + λ * |cos_angle(normal_p, normal_q)|
    需要离线预处理GT点云法向。
    """
    def __init__(self, lambda_na=0.1, **kwargs):
        super().__init__()
        self.lambda_na = lambda_na
        raise NotImplementedError("Normal-Aware CD 尚未实现，需配合GT法向预处理")

    def forward(self, pred_fine, gt, normals_gt=None, **kwargs):
        raise NotImplementedError


@register_loss('info_nce')
class InfoNCEContrastiveLoss(nn.Module):
    """
    InfoNCE 对比损失 (InfoCD) — 预留接口，保留作为失败记录参考。
    """
    def __init__(self, tau=0.05, alpha=0.1, gt_samples=2048, **kwargs):
        super().__init__()
        raise NotImplementedError("InfoCD 已验证无效，保留仅作参考")


# ============================================================
# Loss Manager — 核心调度器
# ============================================================

class LossManager(nn.Module):
    """
    统一损失管理器。
    
    从 config 字典构建，支持：
    - base_loss: fine 阶段的重建损失 (默认 cd_l1)
    - coarse_loss: coarse 阶段的专用损失 (可选, 不设则与 base 相同)
    - auxiliary_losses: 可叠加的辅助损失 (Laplacian 等)
    
    典型用法:
      coarse=APML(512→16384) + fine=CD_L1(16384→16384) + aux=Laplacian
    
    Args:
        loss_cfg: 从 config.yaml 解析出的 loss_config 字典
    """

    def __init__(self, loss_cfg=None):
        super().__init__()

        # ---- 基础损失 (fine阶段默认) ----
        base_name = loss_cfg.get('base_loss', 'cd_l1') if loss_cfg else 'cd_l1'
        assert base_name in LOSS_REGISTRY, f"未知 base_loss: {base_name}，可选: {list(LOSS_REGISTRY.keys())}"
        base_kwargs = loss_cfg.get('base_kwargs', {}) if loss_cfg else {}
        self.base_loss = LOSS_REGISTRY[base_name](**base_kwargs)
        self.base_name = base_name

        # ---- Coarse 阶段专用损失 (可选覆盖) ----
        coarse_name = loss_cfg.get('coarse_loss', None) if loss_cfg else None
        self.coarse_loss_weight = float(loss_cfg.get('coarse_loss_weight', 1.0)) if loss_cfg else 1.0
        if coarse_name is not None:
            assert coarse_name in LOSS_REGISTRY, f"未知 coarse_loss: {coarse_name}"
            coarse_kwargs = loss_cfg.get('coarse_kwargs', {}) if loss_cfg else {}
            self.coarse_loss = LOSS_REGISTRY[coarse_name](**coarse_kwargs)
            self.coarse_name = coarse_name
        else:
            self.coarse_loss = self.base_loss  # 复用 base
            self.coarse_name = f"{base_name}(shared)"

        # ---- 辅助损失（可开闭）----
        self.aux_losses = nn.ModuleDict()
        self.aux_weights = {}
        self.aux_configs = {}

        if loss_cfg and 'auxiliary_losses' in loss_cfg:
            for aux_name, aux_cfg in loss_cfg['auxiliary_losses'].items():
                if not aux_cfg.get('enabled', False):
                    continue
                assert aux_name in LOSS_REGISTRY, f"未知 auxiliary loss: {aux_name}，可选: {list(LOSS_REGISTRY.keys())}"
                aux_kwargs = {k: v for k, v in aux_cfg.items() if k not in ('enabled', 'weight')}
                try:
                    self.aux_losses[aux_name] = LOSS_REGISTRY[aux_name](**aux_kwargs)
                    self.aux_weights[aux_name] = aux_cfg.get('weight', 0.1)
                    self.aux_configs[aux_name] = aux_cfg
                except NotImplementedError as e:
                    print(f"[LossManager] 警告: {aux_name} 未实现，跳过: {e}")

        # 打印汇总信息
        self._print_summary()

    def _print_summary(self):
        print("=" * 60)
        print(f"[LossManager] Base Loss (fine):  {self.base_name}")
        print(f"[LossManager] Coarse Loss:       {self.coarse_name} (weight={self.coarse_loss_weight})")
        print(f"[LossManager] Auxiliary Losses ({len(self.aux_losses)} enabled):")
        for name, module in self.aux_losses.items():
            w = self.aux_weights[name]
            print(f"    + {name}: weight={w} | {module.__class__.__name__}")
        if len(self.aux_losses) == 0:
            print("    (无辅助损失)")
        print("=" * 60)

    def compute_base_cd(self, pred, gt):
        """计算基础 CD（用于 fine 阶段）"""
        return self.base_loss(pred, gt)

    def compute_coarse_loss(self, pred_coarse, gt):
        """
        计算 coarse 阶段损失（可能与 fine 不同）。
        
        当配置了 coarse_loss 时使用专用损失（如 APML），
        否则回退到与 fine 相同的 base_loss。
        
        返回值会乘以 coarse_loss_weight (默认1.0)，
        用于平衡 coarse loss 与 fine CD_L1 的量级。
        """
        return self.coarse_loss_weight * self.coarse_loss(pred_coarse, gt)

    def compute_auxiliary(self, pred_fine, pred_coarse=None, gt=None, gate_w1=None):
        """
        计算所有已启用的辅助损失之和。
        
        Returns:
            total_aux_loss: 标量
            aux_breakdown: dict
        """
        total_aux = torch.tensor(0.0, device=pred_fine.device)
        breakdown = {}

        for name, module in self.aux_losses.items():
            w = self.aux_weights[name]

            if name == 'laplacian':
                loss_val = module(pred_fine, pred_coarse=pred_coarse, gate_w1=gate_w1)
            elif name == 'apml':
                # APML 作为 auxiliary 时作用于 coarse
                loss_val = module(pred_coarse, gt=gt)
            elif name in ('directional_cd', 'normal_aware_cd'):
                loss_val = module(pred_fine, gt=gt)
            elif name == 'info_nce':
                loss_val = module(pred_fine, gt=gt)
            else:
                loss_val = module(pred_fine)

            total_aux = total_aux + w * loss_val
            breakdown[name] = loss_val.item()

        return total_aux, breakdown

    def get_enabled_aux_names(self):
        """返回当前启用的辅助损失名称列表"""
        return list(self.aux_losses.keys())
