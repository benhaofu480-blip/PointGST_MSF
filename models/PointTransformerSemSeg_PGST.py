"""
PointGST Semantic Segmentation Model for S3DIS
基于 PointTransformerPartSeg_PGST 改造:
1. 去掉类别条件信息（S3DIS 是纯语义分割，无物体类别）
2. 输出 13 类（S3DIS 标准语义类别）
3. 输入 6 维 (xyz+rgb)，不用法向量
4. 支持 DASA (Density-Aware Spectral Adapter) 开关
5. 支持 ECFR 开关
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from timm.models.layers import DropPath, trunc_normal_
from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN

from .build import MODELS
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
from models.PGST import PCSA, sort, get_basis
from models.z_order import xyz2key


# ============== Helper Functions ==============

def fps(data, number):
    fps_idx = pointnet2_utils.furthest_point_sample(data.contiguous(), number)
    fps_data = pointnet2_utils.gather_operation(
        data.transpose(1, 2).contiguous(), fps_idx
    ).transpose(1, 2).contiguous()
    return fps_data


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


# ============== Backbone Components (same as PartSeg) ==============

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
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        batch_size, num_points, _ = xyz.shape
        center = fps(xyz, self.num_group)
        _, idx = self.knn(xyz, center)
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, cfg, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, adapt=True,
                 use_ecfr=False, use_dasa=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.adapt = adapt
        if adapt:
            self.gft_adapter = PCSA(dim, cfg, use_ecfr=use_ecfr, use_dasa=use_dasa)

    def forward(self, x, U, sub_U, idx, sub_eigenvalues=None, sub_density=None):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        if self.adapt:
            t = self.gft_adapter(x, U, sub_U, idx,
                                 sub_eigenvalues=sub_eigenvalues,
                                 sub_density=sub_density)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.adapt:
            x = x + t
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, cfg, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., use_ecfr=False, use_dasa=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                cfg=cfg,
                use_ecfr=use_ecfr,
                use_dasa=use_dasa,
            )
            for i in range(depth)])

    def forward(self, x, pos, U, sub_U, idx, sub_eigenvalues=None, sub_density=None):
        feature_list = []
        fetch_idx = [3, 7, 11]
        for i, block in enumerate(self.blocks):
            x = block(x + pos, U, sub_U, idx,
                      sub_eigenvalues=sub_eigenvalues,
                      sub_density=sub_density)
            if i in fetch_idx:
                feature_list.append(x)
        return feature_list


# ============== Upsampling Components ==============

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


class DGCNN_Propagation(nn.Module):
    def __init__(self, k=4):
        super().__init__()
        self.k = k
        self.knn = KNN(k=k, transpose_mode=False)

        self.layer1 = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=1, bias=False),
            nn.GroupNorm(4, 512),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(1024, 384, kernel_size=1, bias=False),
            nn.GroupNorm(4, 384),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def get_graph_feature(self, coor_q, x_q, coor_k, x_k):
        k = self.k
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            _, idx = self.knn(coor_k, coor_q)
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

    def forward(self, coor, f, coor_q, f_q):
        f_q = self.get_graph_feature(coor_q, f_q, coor, f)
        f_q = self.layer1(f_q)
        f_q = f_q.max(dim=-1, keepdim=False)[0]

        f_q = self.get_graph_feature(coor_q, f_q, coor_q, f_q)
        f_q = self.layer2(f_q)
        f_q = f_q.max(dim=-1, keepdim=False)[0]

        return f_q


# ============== Main Model ==============

@MODELS.register_module()
class PointTransformerSemSeg_PGST(nn.Module):
    """S3DIS 语义分割模型: PointBERT backbone + PCSA 适配器 + 多尺度上采样"""

    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.num_heads = config.num_heads
        self.num_classes = config.NUM_CLASSES  # 13 for S3DIS

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims
        self.local = config.local
        self.use_ecfr = getattr(config, 'use_ecfr', False)
        self.use_dasa = getattr(config, 'use_dasa', False)

        # Backbone
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            cfg=config,
            embed_dim=self.trans_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            drop_path_rate=dpr,
            use_ecfr=self.use_ecfr,
            use_dasa=self.use_dasa,
        )
        self.norm = nn.LayerNorm(self.trans_dim)

        # Upsampling network
        # 与 PartSeg 的关键区别: propagation_0 的 in_channel 不包含 16 维类别 one-hot
        self.propagation_2 = PointNetFeaturePropagation(
            in_channel=self.trans_dim + 3, mlp=[self.trans_dim * 4, self.trans_dim])
        self.propagation_1 = PointNetFeaturePropagation(
            in_channel=self.trans_dim + 3, mlp=[self.trans_dim * 4, self.trans_dim])
        self.propagation_0 = PointNetFeaturePropagation(
            in_channel=self.trans_dim + 3, mlp=[self.trans_dim * 4, self.trans_dim])
        self.dgcnn_pro_1 = DGCNN_Propagation(k=4)
        self.dgcnn_pro_2 = DGCNN_Propagation(k=4)

        # FC segmentation head (13 classes for S3DIS)
        self.conv1 = nn.Conv1d(self.trans_dim, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, self.num_classes, 1)

        self.build_loss_func()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

    def build_loss_func(self):
        # ignore_index=-1 忽略无标注点
        self.loss_ce = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, pts, cls_label=None):
        """
        pts: (B, N, 6) xyz+rgb
        cls_label: 占位参数 (S3DIS无物体类别)
        返回: (B, N, num_classes) 点级分割logits
        """
        B, N, _ = pts.shape
        xyz = pts[:, :, :3]  # (B, N, 3)
        rgb = pts[:, :, 3:]  # (B, N, 3)

        # xyz 中心化：减去质心，避免大坐标导致的数值问题
        xyz_mean = xyz.mean(dim=1, keepdim=True)  # (B, 1, 3)
        xyz = xyz - xyz_mean

        # Group: FPS + KNN
        neighborhood, center = self.group_divider(xyz)

        # Encode local features
        group_input_tokens = self.encoder(neighborhood)
        group_input_tokens = self.reduce_dim(group_input_tokens)

        # Prepare transformer input
        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_pos = self.cls_pos.expand(B, -1, -1)
        pos = self.pos_embed(center)
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)

        # PCSA setup
        # 全局 U 实际未被 PCSA 使用，仅为保持接口一致
        # 使用 identity 作为全局 U 的安全 fallback（N=128 时 eigendecompose 不稳定）
        _, G, _ = center.shape
        U = torch.eye(G, device=center.device, dtype=center.dtype).unsqueeze(0).expand(B, -1, -1)
        c = center * 100
        key = xyz2key(c[:, :, 1], c[:, :, 0], c[:, :, 2])
        _, idx0 = torch.sort(key)
        _, idx1 = torch.sort(idx0)
        sub_center = sort(center, idx0)
        group_size = self.local
        group_num = G // group_size

        # DASA: 计算局部密度 (必须在 get_basis 之前，因为要传入密度加权)
        sub_density = None
        if self.use_dasa:
            with torch.no_grad():
                nbh = neighborhood  # (B, G, group_size, 3)
                nbh_var = nbh.var(dim=2).sum(dim=-1)  # (B, G) 方差作为密度指标
                # 归一化到 [0, 1]，加 clamp 防止极端值
                nbh_var_min = nbh_var.min(dim=1, keepdim=True)[0]
                nbh_var_max = nbh_var.max(dim=1, keepdim=True)[0]
                range_var = nbh_var_max - nbh_var_min + 1e-8
                nbh_var_norm = (nbh_var - nbh_var_min) / range_var  # (B, G)
                nbh_var_norm = nbh_var_norm.clamp(0.0, 1.0)  # 安全 clamp
                # 按sub_U排序顺序重排
                density_sorted = sort(nbh_var_norm.unsqueeze(-1), idx0).squeeze(-1)  # (B, G)
                sub_density = density_sorted.reshape(B * group_num, group_size)  # (B*group_num, group_size)

        sub_U, sub_eigenvalues = get_basis(
            sub_center.reshape(B * group_num, group_size, 3),
            density=sub_density
        )
        sub_U = sub_U.reshape(B, group_num, group_size, group_size)

        # Transformer
        feature_list = self.blocks(x, pos, U, sub_U, [idx0, idx1],
                                   sub_eigenvalues=sub_eigenvalues,
                                   sub_density=sub_density)
        feature_list = [self.norm(f)[:, 1:].transpose(-1, -2).contiguous() for f in feature_list]

        # Multi-scale upsampling (无类别条件)
        # Level 0: original points (only xyz, no class one-hot)
        center_level_0 = xyz.transpose(-1, -2).contiguous()  # (B, 3, N)
        f_level_0 = center_level_0  # (B, 3, N)

        # Level 1: 512 FPS points
        center_level_1 = fps(xyz, 512).transpose(-1, -2).contiguous()
        f_level_1 = center_level_1

        # Level 2: 256 FPS points
        center_level_2 = fps(xyz, 256).transpose(-1, -2).contiguous()
        f_level_2 = center_level_2

        # Level 3: group centers
        center_level_3 = center.transpose(-1, -2).contiguous()

        # Top-down propagation
        f_level_3 = feature_list[2]
        f_level_2 = self.propagation_2(center_level_2, center_level_3, f_level_2, feature_list[1])
        f_level_1 = self.propagation_1(center_level_1, center_level_3, f_level_1, feature_list[0])

        # Bottom-up DGCNN refinement
        f_level_2 = self.dgcnn_pro_2(center_level_3, f_level_3, center_level_2, f_level_2)
        f_level_1 = self.dgcnn_pro_1(center_level_2, f_level_2, center_level_1, f_level_1)
        f_level_0 = self.propagation_0(center_level_0, center_level_1, f_level_0, f_level_1)

        # FC segmentation head
        feat = F.relu(self.bn1(self.conv1(f_level_0)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)

        return x

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path, map_location='cpu')
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                if k.startswith('ACT_encoder'):
                    base_ckpt[k[len('ACT_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
