"""
PointGST Part Segmentation Model
基于原版PointBERT分割架构 + PCSA适配器
关键改进：
1. 原版PointNetFeaturePropagation + DGCNN上采样网络（~5.16M参数）
2. PCSA几何结构适配器
3. 类别条件信息（one-hot）融入上采样
4. Transformer多尺度中间层特征提取（layer 3,7,11）
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
    """FPS采样，输入(B,N,3)，输出(B,number,3)"""
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


# ============== Backbone Components ==============

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
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, adapt=True, use_ecfr=False):
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
            self.gft_adapter = PCSA(dim, cfg, use_ecfr=use_ecfr)

    def forward(self, x, U, sub_U, idx, sub_eigenvalues=None):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        if self.adapt:
            t = self.gft_adapter(x, U, sub_U, idx, sub_eigenvalues=sub_eigenvalues)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.adapt:
            x = x + t
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, cfg, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., use_ecfr=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                cfg=cfg,
                use_ecfr=use_ecfr,
            )
            for i in range(depth)])

    def forward(self, x, pos, U, sub_U, idx, sub_eigenvalues=None):
        feature_list = []
        fetch_idx = [3, 7, 11]
        for i, block in enumerate(self.blocks):
            x = block(x + pos, U, sub_U, idx, sub_eigenvalues=sub_eigenvalues)
            if i in fetch_idx:
                feature_list.append(x)
        return feature_list


# ============== Upsampling Components (from Point-BERT segmentation) ==============

class PointNetFeaturePropagation(nn.Module):
    """PointNet++ 风格的特征传播层：反距离加权3-NN插值 + MLP"""
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
        """
        xyz1: (B, 3, N) 输入点坐标
        xyz2: (B, 3, S) 采样点坐标（更稀疏）
        points1: (B, D1, N) 输入点特征
        points2: (B, D2, S) 采样点特征（需要上采样）
        返回: (B, D', N) 上采样后的特征
        """
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
    """DGCNN风格的特征传播：KNN图卷积"""
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
        # coor/x: (B, C, N)
        k = self.k
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            _, idx = self.knn(coor_k, coor_q)  # (B, k, N_q)
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
        """
        coor: (B, 3, G) 源坐标
        f: (B, C, G) 源特征
        coor_q: (B, 3, N) 查询坐标
        f_q: (B, C, N) 查询特征
        返回: (B, 384, N) 增强后的查询特征
        """
        f_q = self.get_graph_feature(coor_q, f_q, coor, f)
        f_q = self.layer1(f_q)
        f_q = f_q.max(dim=-1, keepdim=False)[0]

        f_q = self.get_graph_feature(coor_q, f_q, coor_q, f_q)
        f_q = self.layer2(f_q)
        f_q = f_q.max(dim=-1, keepdim=False)[0]

        return f_q


# ============== Main Model ==============

@MODELS.register_module()
class PointTransformerPartSeg_PGST(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.num_heads = config.num_heads
        self.num_parts = config.NUM_PARTS

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims
        self.local = config.local
        self.use_ecfr = getattr(config, 'use_ecfr', False)

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
        )
        self.norm = nn.LayerNorm(self.trans_dim)

        # Upsampling network (from Point-BERT segmentation, ~5.16M params)
        self.propagation_2 = PointNetFeaturePropagation(
            in_channel=self.trans_dim + 3, mlp=[self.trans_dim * 4, self.trans_dim])
        self.propagation_1 = PointNetFeaturePropagation(
            in_channel=self.trans_dim + 3, mlp=[self.trans_dim * 4, self.trans_dim])
        self.propagation_0 = PointNetFeaturePropagation(
            in_channel=self.trans_dim + 3 + 16, mlp=[self.trans_dim * 4, self.trans_dim])
        self.dgcnn_pro_1 = DGCNN_Propagation(k=4)
        self.dgcnn_pro_2 = DGCNN_Propagation(k=4)

        # FC segmentation head
        self.conv1 = nn.Conv1d(self.trans_dim, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, self.num_parts, 1)

        # Class-to-parts mapping (for IoU evaluation)
        self.class2parts = {
            0: [0, 1, 2, 3], 1: [4, 5], 2: [6, 7], 3: [8, 9, 10, 11],
            4: [12, 13, 14, 15], 5: [16, 17, 18], 6: [19, 20, 21], 7: [22, 23],
            8: [24, 25, 26, 27], 9: [28, 29, 30, 31], 10: [32, 33, 34],
            11: [35, 36, 37], 12: [38, 39, 40], 13: [41, 42, 43],
            14: [44, 45, 46], 15: [47, 48, 49]
        }

        self.build_loss_func()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def forward(self, pts, cls_label):
        """
        pts: (B, N, 6) 点云坐标(+法向量)
        cls_label: (B,) 物体类别标签 0-15
        返回: (B, N, num_parts) 点级分割logits
        """
        B, N, _ = pts.shape
        xyz = pts[:, :, :3]  # (B, N, 3)

        # Group: FPS + KNN
        neighborhood, center = self.group_divider(xyz)
        # neighborhood: (B, G, M, 3), center: (B, G, 3)

        # Encode local features
        group_input_tokens = self.encoder(neighborhood)  # (B, G, encoder_dims)
        group_input_tokens = self.reduce_dim(group_input_tokens)  # (B, G, trans_dim)

        # Prepare transformer input
        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_pos = self.cls_pos.expand(B, -1, -1)
        pos = self.pos_embed(center)
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)

        # PCSA setup
        U, _ = get_basis(center)
        _, G, _ = center.shape
        c = center * 100
        key = xyz2key(c[:, :, 1], c[:, :, 0], c[:, :, 2])
        _, idx0 = torch.sort(key)
        _, idx1 = torch.sort(idx0)
        sub_center = sort(center, idx0)
        group_size = self.local
        group_num = G // group_size
        sub_U, sub_eigenvalues = get_basis(
            sub_center.reshape(B * group_num, group_size, 3)
        )
        sub_U = sub_U.reshape(B, group_num, group_size, group_size)

        # Transformer: extract intermediate features at layers [3, 7, 11]
        feature_list = self.blocks(x, pos, U, sub_U, [idx0, idx1], sub_eigenvalues=sub_eigenvalues)
        feature_list = [self.norm(f)[:, 1:].transpose(-1, -2).contiguous() for f in feature_list]
        # feature_list[0]: layer 3 -> (B, 384, G)
        # feature_list[1]: layer 7 -> (B, 384, G)
        # feature_list[2]: layer 11 -> (B, 384, G)

        # Multi-scale upsampling with class conditioning
        cls_one_hot = F.one_hot(cls_label, 16).float().unsqueeze(-1)  # (B, 16, 1)
        cls_one_hot = cls_one_hot.expand(-1, -1, N)  # (B, 16, N)

        # Level 0: original points with class info
        center_level_0 = xyz.transpose(-1, -2).contiguous()  # (B, 3, N)
        f_level_0 = torch.cat([cls_one_hot, center_level_0], 1)  # (B, 19, N)

        # Level 1: 512 FPS points
        center_level_1 = fps(xyz, 512).transpose(-1, -2).contiguous()  # (B, 3, 512)
        f_level_1 = center_level_1  # (B, 3, 512)

        # Level 2: 256 FPS points
        center_level_2 = fps(xyz, 256).transpose(-1, -2).contiguous()  # (B, 3, 256)
        f_level_2 = center_level_2  # (B, 3, 256)

        # Level 3: group centers (G points)
        center_level_3 = center.transpose(-1, -2).contiguous()  # (B, 3, G)

        # Top-down: propagate from group-level features to intermediate levels
        f_level_3 = feature_list[2]  # (B, 384, G)
        f_level_2 = self.propagation_2(center_level_2, center_level_3, f_level_2, feature_list[1])
        f_level_1 = self.propagation_1(center_level_1, center_level_3, f_level_1, feature_list[0])

        # Bottom-up: refine with DGCNN graph convolution
        f_level_2 = self.dgcnn_pro_2(center_level_3, f_level_3, center_level_2, f_level_2)
        f_level_1 = self.dgcnn_pro_1(center_level_2, f_level_2, center_level_1, f_level_1)
        f_level_0 = self.propagation_0(center_level_0, center_level_1, f_level_0, f_level_1)

        # FC segmentation head
        feat = F.relu(self.bn1(self.conv1(f_level_0)))  # (B, 128, N)
        x = self.drop1(feat)
        x = self.conv2(x)  # (B, num_parts, N)
        x = x.permute(0, 2, 1)  # (B, N, num_parts)

        return x

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
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
