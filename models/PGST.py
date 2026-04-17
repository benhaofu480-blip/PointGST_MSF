import torch
import torch.nn as nn
from timm.models.layers import DropPath


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

def get_basis(center, density=None):
    """
    计算图拉普拉斯的特征分解。
    
    Args:
        center: (B, N, 3) 中心点坐标
        density: (B, N) 局部点密度 (可选，DASA使用)
    
    Returns:
        U: (B, N, N) 特征向量矩阵
        eigenvalues: (B, N) 特征值
    """
    B, N, C = center.shape
    device = center.device
    dtype = center.dtype
    
    dist = torch.cdist(center, center)  # (B, N, N)
    
    # 邻接矩阵：KNN 高斯核，避免全连接导致的病态矩阵
    k = min(6, N)  # KNN邻居数，避免过大
    _, knn_idx = dist.topk(k + 1, dim=-1, largest=False)  # (B, N, k+1) 包含自身
    
    # 构建稀疏邻接矩阵
    eye = torch.eye(N, device=device, dtype=dtype).unsqueeze(0)  # (1, N, N)
    
    # 用 KNN 邻居的均值距离做归一化
    knn_dists = torch.gather(dist, -1, knn_idx)  # (B, N, k+1)
    # 排除自身(第0列)的均值
    mean_knn_dist = knn_dists[:, :, 1:].mean(dim=-1, keepdim=True)  # (B, N, 1)
    mean_knn_dist = mean_knn_dist.clamp(min=1e-6)  # 避免除零
    
    # 全局最小非零距离用于归一化
    dist_fill = dist.clone()
    dist_fill[dist_fill == 0] = float('inf')
    min_dist = dist_fill.min(dim=-1).values.min(dim=-1).values  # (B,)
    min_dist = min_dist.clamp(min=1e-6).view(-1, 1, 1)  # (B, 1, 1)
    
    # 高斯核邻接矩阵
    adj = torch.zeros_like(dist)
    sigma = min_dist  # 用全局最邻近距离作为带宽
    adj = torch.exp(-0.5 * (dist / sigma) ** 2)
    
    # 保留 KNN 邻居，其余置零
    knn_mask = torch.zeros_like(dist)
    knn_mask.scatter_(-1, knn_idx, 1.0)
    adj = adj * knn_mask
    # 确保对角线为0（无自环）
    adj = adj * (1.0 - eye)

    # DASA: 用密度加权邻接矩阵
    if density is not None:
        d_i = density.unsqueeze(-1)   # (B, N, 1)
        d_j = density.unsqueeze(-2)   # (B, 1, N)
        density_weight = torch.sqrt(d_i * d_j + 1e-8)
        adj = adj * density_weight

    L = get_laplacian(adj)
    
    try:
        eigenvalues, U = torch.linalg.eigh(L)
    except RuntimeError:
        # fallback: 加正则化重新计算
        L_reg = L + 1e-3 * eye
        eigenvalues, U = torch.linalg.eigh(L_reg)
    
    return U, eigenvalues  # 返回特征向量和特征值

def sort(pts: torch.Tensor, idx: torch.Tensor):
    return torch.gather(pts, dim=1, index=idx.unsqueeze(-1).expand(-1, -1, pts.size(-1)))

class PCSA(nn.Module):
    def __init__(self, dim, cfg, use_ecfr=False, use_dasa=False):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rank = cfg.rank
        self.use_ecfr = use_ecfr
        self.use_dasa = use_dasa
        self.norm_ly1 = nn.LayerNorm(self.rank)
        self.norm_ly2 = nn.LayerNorm(self.rank)

        self.act = nn.SiLU()
        self.down = nn.Linear(dim, self.rank)
        self.up = nn.Linear(self.rank, dim)

        self.scale=1.

        self.adapt1 = nn.Linear(self.rank, self.rank)
        nn.init.zeros_(self.adapt1.weight)
        nn.init.zeros_(self.adapt1.bias)
        
        self.drop_adapt1 = DropPath(0.)
        self.drop_adapt2 = DropPath(0.)
        self.drop_out = nn.Dropout(0.)

        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

        # ECFR: 特征值条件化门控
        if self.use_ecfr:
            self.eigen_gate = nn.Sequential(
                nn.Linear(1, self.rank // 4),
                nn.GELU(),
                nn.Linear(self.rank // 4, self.rank),
            )

    def forward(self, input, U, sub_U, idx, sub_eigenvalues=None, sub_density=None):
        h = self.down(input)

        B0, group_num, group_size, _ = sub_U.shape
        G0 = group_num * group_size

        x = h[:, -G0:, :]
        h = self.act(h)
        
        sub_x0=sort(x,idx[0])
        sub_x0 = sub_x0.reshape(B0, group_num, group_size, self.rank)
        
        # 频域变换
        sub_x_f0 = sub_U @ sub_x0

        # ECFR: 用局部特征值条件化频域特征
        if self.use_ecfr and sub_eigenvalues is not None:
            # sub_eigenvalues: (B*group_num, group_size) → (B0, group_num, group_size)
            local_complexity = sub_eigenvalues[:, :, :self.rank].mean(dim=-1, keepdim=True)  # (B0, group_num, 1)
            gate = torch.sigmoid(self.eigen_gate(local_complexity))  # (B0, group_num, self.rank)
            sub_x_f0 = sub_x_f0 * gate

        sub_h_f0 = sub_x_f0
        sub_x_f0 = self.norm_ly2(sub_x_f0)
        sub_x_f0 = sub_h_f0 + self.drop_adapt2(self.act(self.drop_out(self.adapt1(sub_x_f0))))
        sub_x0 = sub_U.transpose(-2, -1) @ sub_x_f0

        sub_x0 = sub_x0.reshape(B0, G0, self.rank)
        sub_x0=sort(sub_x0,idx[1])

        x = x + sub_x0

        h[:, -G0:, :] = x + h[:, -G0:, :]
        h = self.up(h)
        return h*self.scale
