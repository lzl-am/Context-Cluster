"""
ContextCluster implementation
# --------------------------------------------------------
# Context Cluster -- Image as Set of Points, ICLR'23 Oral
# Licensed under The MIT License [see LICENSE for details]
# Written by Xu Ma (ma.xu1@northeastern.com)
# --------------------------------------------------------
"""
import os
import copy
import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple
from einops import rearrange
import torch.nn.functional as F

try:
    from mmseg.models.builder import BACKBONES as seg_BACKBONES
    from mmseg.utils import get_root_logger
    from mmcv.runner import _load_checkpoint

    has_mmseg = True
except ImportError:
    print("If for semantic segmentation, please install mmsegmentation first")
    has_mmseg = False

try:
    from mmdet.models.builder import BACKBONES as det_BACKBONES
    from mmdet.utils import get_root_logger
    from mmcv.runner import _load_checkpoint

    has_mmdet = True
except ImportError:
    print("If for detection, please install mmdetection first")
    has_mmdet = False


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224),
        'crop_pct': .95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'model_small': _cfg(crop_pct=0.9),
    'model_medium': _cfg(crop_pct=0.95),
}


class PointRecuder(nn.Module):
    """
    点降采样器 - 本质上等价于卷积操作的Patch Embedding
    
    作用:
    - 将输入特征图进行降采样，减少点的数量
    - 同时改变通道维度
    
    数学等价性:
    - 对于规则网格上的点，通过步长卷积进行降采样
    - 数学上等价于对点集进行区域聚合
    
    Input:  [B, in_chans, H, W]
    Output: [B, embed_dim, H/stride, W/stride]
    """

    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class GroupNorm(nn.GroupNorm):
    """
    使用1个组的GroupNorm，等价于LayerNorm在空间维度上的应用
    
    为什么用GroupNorm而不是BatchNorm?
    - BatchNorm对batch size敏感，小batch效果差
    - GroupNorm与batch size无关，更稳定
    - 1个组的GroupNorm = LayerNorm (对每个样本的所有通道归一化)
    
    Input: [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):
    """
    计算两个张量之间的成对余弦相似度矩阵
    
    余弦相似度公式: cos(a,b) = (a·b) / (||a|| × ||b||)
    
    Args:
        x1: [B, ..., M, D]  M个D维向量
        x2: [B, ..., N, D]  N个D维向量
    
    Returns:
        sim: [B, ..., M, N]  相似度矩阵，sim[i,j]表示x1[i]和x2[j]的相似度
    
    用途: 计算聚类中心与各点之间的相似度
    """
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)

    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim

# region Cluster模块
class Cluster(nn.Module):
    def __init__(self, dim, out_dim, proposal_w=2, proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24,
                 return_center=False):
        """

        :param dim:  channel nubmer
        :param out_dim: channel nubmer
        :param proposal_w: the sqrt(proposals) value, we can also set a different value
        :param proposal_h: the sqrt(proposals) value, we can also set a different value
        :param fold_w: the sqrt(number of regions) value, we can also set a different value
        :param fold_h: the sqrt(number of regions) value, we can also set a different value
        :param heads:  heads number in context cluster
        :param head_dim: dimension of each head in context cluster
        :param return_center: if just return centers instead of dispatching back (deprecated).
            
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        Cluster 模块工作流程                              │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  1. 特征提取:                                                           │
    │     x ──┬──► f(x) ──► 用于计算相似度的特征                              │
    │         └──► v(x) ──► 用于聚合的value特征                               │
    │                                                                         │
    │  2. 生成聚类中心 (Context Cluster Centers):                             │
    │     使用 AdaptiveAvgPool2d 从特征图中生成 M 个候选中心                   │
    │     centers = AvgPool(f(x))  # [B, C, proposal_h, proposal_w]           │
    │                                                                         │
    │  3. 计算相似度:                                                         │
    │     sim = sigmoid(α * cos_sim(centers, points) + β)                     │
    │     - α, β 是可学习参数                                                 │
    │     - 每个点找到最相似的中心 (hard assignment)                          │
    │                                                                         │
    │  4. 聚合 (Aggregate):                                                   │
    │     center_features = Σ(sim * value) / Σ(sim)                           │
    │     将属于同一中心的点的value加权求和                                    │
    │                                                                         │
    │  5. 分发 (Dispatch):                                                    │
    │     output = Σ(sim * center_features)                                   │
    │     将聚合后的中心特征分发回各个点                                       │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
        """
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim

        # f: 用于计算相似度的投影
        self.f = nn.Conv2d(dim, heads * head_dim, kernel_size=1)  # for similarity
        # v: 用于聚合的value投影
        self.v = nn.Conv2d(dim, heads * head_dim, kernel_size=1)  # for value
        # proj: 输出投影，将多头合并后投影到out_dim
        self.proj = nn.Conv2d(heads * head_dim, out_dim, kernel_size=1)  # for projecting channel number

        # 可学习的相似度缩放参数
        # sim = sigmoid(α * cos_sim + β)
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))

        # 通过自适应平均池化生成中心
        self.centers_proposal = nn.AdaptiveAvgPool2d((proposal_w, proposal_h))
        
        # 区域划分参数
        self.fold_w = fold_w
        self.fold_h = fold_h
        self.return_center = return_center

    def forward(self, x):  # [b,c,w,h]
        """
        Args:
            x: [B, C, W, H] 输入特征图
        
        Returns:
            out: [B, out_dim, W, H] 输出特征图
        """
        # ========== Step 1: 特征投影 ==========
        value = self.v(x)
        x = self.f(x)

        # 拆分多头: [B, heads*head_dim, W, H] -> [B*heads, head_dim, W, H]
        x = rearrange(x, "b (e c) w h -> (b e) c w h", e=self.heads)
        value = rearrange(value, "b (e c) w h -> (b e) c w h", e=self.heads)

        # ========== Step 2: 区域划分 (可选) ==========
        # 将大特征图分成多个小区域，在每个区域内独立聚类
        # 目的: 减少计算量，同时保持局部性
        if self.fold_w > 1 and self.fold_h > 1:
            # split the big feature maps to small local regions to reduce computations.
            b0, c0, w0, h0 = x.shape
            assert w0 % self.fold_w == 0 and h0 % self.fold_h == 0, \
                f"Ensure the feature map size ({w0}*{h0}) can be divided by fold {self.fold_w}*{self.fold_h}"
            x = rearrange(x, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w,
                          f2=self.fold_h)  # [bs*blocks,c,ks[0],ks[1]]
            value = rearrange(value, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w, f2=self.fold_h)
        b, c, w, h = x.shape

        # ========== Step 3: 生成聚类中心 ==========
        # [B, C, proposal_w, proposal_h] → proposal_w * proposal_h 个中心
        centers = self.centers_proposal(x)  # [b,c,C_W,C_H], we set M = C_W*C_H and N = w*h
        value_centers = rearrange(self.centers_proposal(value), 'b c w h -> b (w h) c')  # [b,C_W,C_H,c]
        
        b, c, ww, hh = centers.shape

        # ========== Step 4: 计算相似度 ==========
        # 计算第 M 个中心与第 N 个点的相似度得分
        # pairwise_cos_sim: 余弦相似度，取值范围 [-1, 1]
        # self.sim_alpha * sim + self.sim_beta: 可学习的仿射变换(偏移因子 + 缩放因子)
        # torch.sigmoid 将相似度归一化到 [0, 1]
        sim = torch.sigmoid(
            self.sim_beta +
            self.sim_alpha * pairwise_cos_sim(
                centers.reshape(b, c, -1).permute(0, 2, 1),
                x.reshape(b, c, -1).permute(0, 2, 1)
            )
        )  # [B,M,N]

        # ========== Step 5: Hard Assignment (硬分配) ==========
        # 找到每个点最相似的中心
        # 每个点 n 沿着 dim=1（聚类中心维度 M）求最大值
        sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)

        # 创建 One-Hot 掩码
        mask = torch.zeros_like(sim)  # binary #[B,M,N]
        # 最大位置填 1
        # dim=1：沿着「聚类中心维度 M」进行填充 → 和之前 max 的 dim=1 严格一致，必须对齐
        # index=sim_max_idx：填充的「位置索引」，形状 [B,1,N] → 每个特征点 n 对应的最优中心的索引 m
        # value=1.0：填充的「数值」→ 在最优中心的位置填 1，其余位置保持 0
        mask.scatter_(1, sim_max_idx, 1.)

        # 利用广播乘法实现最终的筛选，只保留最大相似度
        sim = sim * mask

        # ========== Step 6: 聚合 (Aggregate) ==========
        # 将每个中心管辖的点的value加权求和
        # value: [B, C, w, h] -> [B, N, C]
        value2 = rearrange(value, 'b c w h -> b (w h) c')  # [B,N,D]

        # aggregate step, out shape [B,M,D]
        # 聚合公式: center_out = (Σ sim * value + value_centers) / (Σ sim + 1)
        # value2.unsqueeze(1): [B, 1, N, D]
        # sim.unsqueeze(-1): [B, M, N, 1]
        # 相乘: [B, M, N, D]
        # sum(dim=2): [B, M, D]
        out = ((value2.unsqueeze(dim=1) * sim.unsqueeze(dim=-1)).sum(dim=2) + value_centers) / (
                    sim.sum(dim=-1, keepdim=True) + 1.0)  # [B,M,D]
        # 加1.0防止除零，同时起到残差连接的作用

        # ========== Step 7: 分发 (Dispatch) ==========
        if self.return_center:
            # 只返回中心特征 (deprecated)
            out = rearrange(out, "b (w h) c -> b c w h", w=ww)
        else:
            # 将中心特征分发回每个点
            # out: [B, M, D], sim: [B, M, N]
            # out.unsqueeze(2): [B, M, 1, D]
            # sim.unsqueeze(-1): [B, M, N, 1]
            # 相乘求和: [B, N, D]
            out = (out.unsqueeze(dim=2) * sim.unsqueeze(dim=-1)).sum(dim=1)  # [B,N,D]
            out = rearrange(out, "b (w h) c -> b c w h", w=w)

        # ========== Step 8: 恢复区域划分 ==========
        if self.fold_w > 1 and self.fold_h > 1:
            # recover the splited regions back to big feature maps if use the region partition.
            out = rearrange(out, "(b f1 f2) c w h -> b c (f1 w) (f2 h)", f1=self.fold_w, f2=self.fold_h)
        
        # ========== Step 9: 合并多头 ==========
        out = rearrange(out, "(b e) c w h -> b (e c) w h", e=self.heads)
        out = self.proj(out)
        return out
# endregion


class Mlp(nn.Module):
    """
    MLP模块 - 使用nn.Linear实现
    
    结构: Linear -> GELU -> Dropout -> Linear -> Dropout
    
    在ViT/MetaFormer中的作用:
    - Token Mixer (Cluster) 负责空间信息交互
    - MLP 负责通道维度的特征变换
    
    Input: [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x.permute(0, 2, 3, 1))
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x).permute(0, 3, 1, 2)
        x = self.drop(x)
        return x


# region ClusterBlock
class ClusterBlock(nn.Module):
    """
    Implementation of one block.
    --dim: embedding dim
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth,
        refer to https://arxiv.org/abs/1603.09382
    --use_layer_scale, --layer_scale_init_value: LayerScale,
        refer to https://arxiv.org/abs/2103.17239

    结构:
    ┌─────────────────────────────────────────────┐
    │  x ─────────────────────┐                   │
    │  │                      │                   │
    │  ↓                      │                   │
    │  Norm1                  │                   │
    │  ↓                      │                   │
    │  Cluster (Token Mixer)  │                   │
    │  ↓                      │                   │
    │  LayerScale             │                   │
    │  ↓                      │                   │
    │  DropPath               │                   │
    │  ↓                      │                   │
    │  + ←────────────────────┘  (残差连接)       │
    │  │                                          │
    │  ↓ ─────────────────────┐                   │
    │  Norm2                  │                   │
    │  ↓                      │                   │
    │  MLP (Channel Mixer)    │                   │
    │  ↓                      │                   │
    │  LayerScale             │                   │
    │  ↓                      │                   │
    │  DropPath               │                   │
    │  ↓                      │                   │
    │  + ←────────────────────┘  (残差连接)       │
    │  │                                          │
    │  ↓                                          │
    │  output                                     │
    └─────────────────────────────────────────────┘
    
    技术要点:
    - DropPath (Stochastic Depth): 训练时随机丢弃整个块，增强泛化
    - LayerScale: 对每个块的输出进行可学习缩放，稳定深层网络训练
    """

    def __init__(self, dim, mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 # for context-cluster
                 proposal_w=2, proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24, return_center=False):

        super().__init__()

        self.norm1 = norm_layer(dim)
        # dim, out_dim, proposal_w=2,proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24, return_center=False
        self.token_mixer = Cluster(dim=dim, out_dim=dim, proposal_w=proposal_w, proposal_h=proposal_h,
                                   fold_w=fold_w, fold_h=fold_h, heads=heads, head_dim=head_dim, return_center=False)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        # The following two techniques are useful to train deep ContextClusters.
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            # 初始化为极小值，让深层网络初期接近恒等映射，稳定训练
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def basic_blocks(dim, index, layers,
                 mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop_rate=.0, drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 # for context-cluster
                 proposal_w=2, proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24, return_center=False):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * ( block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(ClusterBlock(
            dim, mlp_ratio=mlp_ratio,
            act_layer=act_layer, norm_layer=norm_layer,
            drop=drop_rate, drop_path=block_dpr,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
            heads=heads, head_dim=head_dim, return_center=False
        ))
    blocks = nn.Sequential(*blocks)

    return blocks
# endregion


# region ContextCluster
class ContextCluster(nn.Module):
    """
    ContextCluster, the main class of our model
    --layers: [x,x,x,x], number of blocks for the 4 stages
    --embed_dims, --mlp_ratios, the embedding dims, mlp ratios
    --downsamples: flags to apply downsampling or not
    --norm_layer, --act_layer: define the types of normalization and activation
    --num_classes: number of classes for the image classification
    --in_patch_size, --in_stride, --in_pad: specify the patch embedding
        for the input image
    --down_patch_size --down_stride --down_pad:
        specify the downsample (patch embed.)
    --fork_feat: whether output features of the 4 stages, for dense prediction
    --init_cfg, --pretrained:
        for mmdetection and mmsegmentation to load pretrained weights

    整体结构:
    ┌────────────────────────────────────────────────────────────────────────┐
    │                                                                        │
    │  Input: [B, 3, 224, 224]                                              │
    │         ↓                                                              │
    │  Position Encoding: 添加2D位置坐标 → [B, 5, 224, 224]                  │
    │         ↓                                                              │
    │  Patch Embedding: stride=4 → [B, embed_dims[0], 56, 56]               │
    │         ↓                                                              │
    │  ┌──────────────────────────────────────────────────────────────────┐ │
    │  │ Stage 1: ClusterBlocks × layers[0]                               │ │
    │  │ 分辨率: 56×56, 通道: embed_dims[0]                                │ │
    │  │ fold: 8×8 (分成64个区域)                                          │ │
    │  └──────────────────────────────────────────────────────────────────┘ │
    │         ↓ PointReducer (downsample)                                    │
    │  ┌──────────────────────────────────────────────────────────────────┐ │
    │  │ Stage 2: ClusterBlocks × layers[1]                               │ │
    │  │ 分辨率: 28×28, 通道: embed_dims[1]                                │ │
    │  │ fold: 4×4 (分成16个区域)                                          │ │
    │  └──────────────────────────────────────────────────────────────────┘ │
    │         ↓ PointReducer (downsample)                                    │
    │  ┌──────────────────────────────────────────────────────────────────┐ │
    │  │ Stage 3: ClusterBlocks × layers[2]                               │ │
    │  │ 分辨率: 14×14, 通道: embed_dims[2]                                │ │
    │  │ fold: 2×2 (分成4个区域)                                           │ │
    │  └──────────────────────────────────────────────────────────────────┘ │
    │         ↓ PointReducer (downsample)                                    │
    │  ┌──────────────────────────────────────────────────────────────────┐ │
    │  │ Stage 4: ClusterBlocks × layers[3]                               │ │
    │  │ 分辨率: 7×7, 通道: embed_dims[3]                                  │ │
    │  │ fold: 1×1 (全局聚类)                                              │ │
    │  └──────────────────────────────────────────────────────────────────┘ │
    │         ↓                                                              │
    │  Global Average Pooling → [B, embed_dims[3]]                          │
    │         ↓                                                              │
    │  Classification Head → [B, num_classes]                               │
    │                                                                        │
    └────────────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, layers, embed_dims=None,
                 mlp_ratios=None, downsamples=None,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.GELU,
                 num_classes=1000,
                 in_patch_size=4, in_stride=4, in_pad=0,
                 down_patch_size=2, down_stride=2, down_pad=0,
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 fork_feat=False,
                 init_cfg=None,
                 pretrained=None,
                 # the parameters for context-cluster
                 proposal_w=[2, 2, 2, 2], proposal_h=[2, 2, 2, 2], fold_w=[8, 4, 2, 1], fold_h=[8, 4, 2, 1],
                 heads=[2, 4, 6, 8], head_dim=[16, 16, 32, 32],
                 **kwargs):

        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat

        self.patch_embed = PointRecuder(
            patch_size=in_patch_size, stride=in_stride, padding=in_pad,
            in_chans=5, embed_dim=embed_dims[0])

        # set the main block in network
        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers,
                                 mlp_ratio=mlp_ratios[i],
                                 act_layer=act_layer, norm_layer=norm_layer,
                                 drop_rate=drop_rate,
                                 drop_path_rate=drop_path_rate,
                                 use_layer_scale=use_layer_scale,
                                 layer_scale_init_value=layer_scale_init_value,
                                 proposal_w=proposal_w[i], proposal_h=proposal_h[i],
                                 fold_w=fold_w[i], fold_h=fold_h[i], heads=heads[i], head_dim=head_dim[i],
                                 return_center=False
                                 )
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                # downsampling between two stages
                network.append(
                    PointRecuder(
                        patch_size=down_patch_size, stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i], embed_dim=embed_dims[i + 1]
                    )
                )

        self.network = nn.ModuleList(network)

        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    # TODO: more elegant way
                    """For RetinaNet, `start_level=1`. The first norm layer will not used.
                    cmd: `FORK_LAST3=1 python -m torch.distributed.launch ...`
                    """
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            # Classifier head
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(
                embed_dims[-1], num_classes) if num_classes > 0 \
                else nn.Identity()

        self.apply(self.cls_init_weights)

        self.init_cfg = copy.deepcopy(init_cfg)
        # load pre-trained model
        if self.fork_feat and (
                self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    # init for classification
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # init for mmdetection or mmsegmentation by loading
    # imagenet pre-trained weights
    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(
                ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = _state_dict
            missing_keys, unexpected_keys = \
                self.load_state_dict(state_dict, False)

            # show for debug
            # print('missing_keys: ', missing_keys)
            # print('unexpected_keys: ', unexpected_keys)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        """
        位置编码

        聚类操作是置换不变的（点的顺序不影响结果）
        需要显式高数模型每个点的空间位置
        """
        _, c, img_w, img_h = x.shape
        # print(f"det img size is {img_w} * {img_h}")
        # register positional information buffer.

        # 生成归一化坐标 [0, 1]
        range_w = torch.arange(0, img_w, step=1) / (img_w - 1.0)
        range_h = torch.arange(0, img_h, step=1) / (img_h - 1.0)

        # 创建 2D 网格并中心化 [-0.5, 0.5]
        fea_pos = torch.stack(torch.meshgrid(range_w, range_h, indexing='ij'), dim=-1).float()
        fea_pos = fea_pos.to(x.device)
        fea_pos = fea_pos - 0.5
        pos = fea_pos.permute(2, 0, 1).unsqueeze(dim=0).expand(x.shape[0], -1, -1, -1)
        
        # 与RGB拼接: [B,3,H,W] + [B,2,H,W] → [B,5,H,W]
        # 通过 patch embedding 转换为 [B, embed_dims[0], H / 4, W / 4]
        x = self.patch_embed(torch.cat([x, pos], dim=1))
        return x

    def forward_tokens(self, x):
        """
        前向传播: 通过所有stages
        
        Args:
            x: [B, C, H, W] 嵌入特征
        
        Returns:
            fork_feat=True: 4个stage的特征列表 (用于检测/分割)
            fork_feat=False: 最终特征 [B, C, H, W]
        """
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                # 收集各stage输出
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            # output the features of four stages for dense prediction
            return outs
        # output only the features of last layer for image classification
        return x

    def forward(self, x):
        """
        完整前向传播
        
        Args:
            x: [B, 3, H, W] 输入图像
        
        Returns:
            fork_feat=True: [feat1, feat2, feat3, feat4] 多尺度特征
            fork_feat=False: [B, num_classes] 分类logits
        """
        # Step 1: 位置编码 + Patch Embedding
        x = self.forward_embeddings(x)
        # Step 2: 通过backbone
        x = self.forward_tokens(x)
        if self.fork_feat:
            # 密集预测: 返回多尺度特征
            return x
        
        # Step 3: 分类
        x = self.norm(x)
        # 全局平均池化: [B, C, H, W] -> [B, C]
        cls_out = self.head(x.mean([-2, -1]))
        # for image classification
        return cls_out
# endregion


# region 模型配置函数
@register_model
def coc_tiny(pretrained=False, **kwargs):
    layers = [3, 4, 5, 2]
    norm_layer = GroupNorm
    embed_dims = [32, 64, 196, 320]
    mlp_ratios = [8, 8, 4, 4]
    downsamples = [True, True, True, True]
    proposal_w = [2, 2, 2, 2]
    proposal_h = [2, 2, 2, 2]
    fold_w = [8, 4, 2, 1]
    fold_h = [8, 4, 2, 1]
    heads = [4, 4, 8, 8]
    head_dim = [24, 24, 24, 24]
    down_patch_size = 3
    down_pad = 1
    model = ContextCluster(
        layers, embed_dims=embed_dims, norm_layer=norm_layer,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        down_patch_size=down_patch_size, down_pad=down_pad,
        proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
        heads=heads, head_dim=head_dim,
        **kwargs)
    model.default_cfg = default_cfgs['model_small']
    return model


@register_model
def coc_tiny2(pretrained=False, **kwargs):
    layers = [3, 4, 5, 2]
    norm_layer = GroupNorm
    embed_dims = [32, 64, 196, 320]
    mlp_ratios = [8, 8, 4, 4]
    downsamples = [True, True, True, True]
    proposal_w = [4, 2, 7, 4]
    proposal_h = [4, 2, 7, 4]
    fold_w = [7, 7, 1, 1]
    fold_h = [7, 7, 1, 1]
    heads = [4, 4, 8, 8]
    head_dim = [24, 24, 24, 24]
    down_patch_size = 3
    down_pad = 1
    model = ContextCluster(
        layers, embed_dims=embed_dims, norm_layer=norm_layer,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        down_patch_size=down_patch_size, down_pad=down_pad,
        proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
        heads=heads, head_dim=head_dim,
        **kwargs)
    model.default_cfg = default_cfgs['model_small']
    return model


@register_model
def coc_small(pretrained=False, **kwargs):
    layers = [2, 2, 6, 2]
    norm_layer = GroupNorm
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [8, 8, 4, 4]
    downsamples = [True, True, True, True]
    proposal_w = [2, 2, 2, 2]
    proposal_h = [2, 2, 2, 2]
    fold_w = [8, 4, 2, 1]
    fold_h = [8, 4, 2, 1]
    heads = [4, 4, 8, 8]
    head_dim = [32, 32, 32, 32]
    down_patch_size = 3
    down_pad = 1
    model = ContextCluster(
        layers, embed_dims=embed_dims, norm_layer=norm_layer,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        down_patch_size=down_patch_size, down_pad=down_pad,
        proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
        heads=heads, head_dim=head_dim,
        **kwargs)
    model.default_cfg = default_cfgs['model_small']
    return model


@register_model
def coc_medium(pretrained=False, **kwargs):
    layers = [4, 4, 12, 4]
    norm_layer = GroupNorm
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [8, 8, 4, 4]
    downsamples = [True, True, True, True]
    proposal_w = [2, 2, 2, 2]
    proposal_h = [2, 2, 2, 2]
    fold_w = [8, 4, 2, 1]
    fold_h = [8, 4, 2, 1]
    heads = [6, 6, 12, 12]
    head_dim = [32, 32, 32, 32]
    down_patch_size = 3
    down_pad = 1
    model = ContextCluster(
        layers, embed_dims=embed_dims, norm_layer=norm_layer,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        down_patch_size=down_patch_size, down_pad=down_pad,
        proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
        heads=heads, head_dim=head_dim,
        **kwargs)
    model.default_cfg = default_cfgs['model_small']
    return model


@register_model
def coc_base_dim64(pretrained=False, **kwargs):
    layers = [6, 6, 24, 6]
    norm_layer = GroupNorm
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [8, 8, 4, 4]
    downsamples = [True, True, True, True]
    proposal_w = [2, 2, 2, 2]
    proposal_h = [2, 2, 2, 2]
    fold_w = [8, 4, 2, 1]
    fold_h = [8, 4, 2, 1]
    heads = [8, 8, 16, 16]
    head_dim = [32, 32, 32, 32]
    down_patch_size = 3
    down_pad = 1
    model = ContextCluster(
        layers, embed_dims=embed_dims, norm_layer=norm_layer,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        down_patch_size=down_patch_size, down_pad=down_pad,
        proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
        heads=heads, head_dim=head_dim,
        **kwargs)
    model.default_cfg = default_cfgs['model_small']
    return model


@register_model
def coc_base_dim96(pretrained=False, **kwargs):
    layers = [4, 4, 12, 4]
    norm_layer = GroupNorm
    embed_dims = [96, 192, 384, 768]
    mlp_ratios = [8, 8, 4, 4]
    downsamples = [True, True, True, True]
    proposal_w = [2, 2, 2, 2]
    proposal_h = [2, 2, 2, 2]
    fold_w = [8, 4, 2, 1]
    fold_h = [8, 4, 2, 1]
    heads = [8, 8, 16, 16]
    head_dim = [32, 32, 32, 32]
    down_patch_size = 3
    down_pad = 1
    model = ContextCluster(
        layers, embed_dims=embed_dims, norm_layer=norm_layer,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        down_patch_size=down_patch_size, down_pad=down_pad,
        proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
        heads=heads, head_dim=head_dim,
        **kwargs)
    model.default_cfg = default_cfgs['model_small']
    return model


"""
Updated: add plain models (without region partition) for tiny, small, and base , etc.
Re-trained with new implementation (PWconv->MLP for faster training and inference), achieve slightly better performance.
"""
@register_model
def coc_tiny_plain(pretrained=False, **kwargs):
    # sharing same parameters as coc_tiny, without region partition.
    layers = [3, 4, 5, 2]
    norm_layer = GroupNorm
    embed_dims = [32, 64, 196, 320]
    mlp_ratios = [8, 8, 4, 4]
    downsamples = [True, True, True, True]
    proposal_w = [4, 4, 2, 2]
    proposal_h = [4, 4, 2, 2]
    fold_w = [1, 1, 1, 1]
    fold_h = [1, 1, 1, 1]
    heads = [4, 4, 8, 8]
    head_dim = [24, 24, 24, 24]
    down_patch_size = 3
    down_pad = 1
    model = ContextCluster(
        layers, embed_dims=embed_dims, norm_layer=norm_layer,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        down_patch_size=down_patch_size, down_pad=down_pad,
        proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
        heads=heads, head_dim=head_dim,
        **kwargs)
    model.default_cfg = default_cfgs['model_small']
    return model


if has_mmdet:
    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class context_cluster_small_feat2(ContextCluster):
        def __init__(self, **kwargs):
                layers = [2, 2, 6, 2]
                norm_layer=GroupNorm
                embed_dims = [64, 128, 320, 512]
                mlp_ratios = [8, 8, 4, 4]
                downsamples = [True, True, True, True]
                proposal_w=[2,2,2,2]
                proposal_h=[2,2,2,2]
                fold_w=[8,4,2,1]
                fold_h=[8,4,2,1]
                heads=[4,4,8,8]
                head_dim=[32,32,32,32]
                down_patch_size=3
                down_pad = 1
                super().__init__(
                    layers, embed_dims=embed_dims, norm_layer=norm_layer,
                    mlp_ratios=mlp_ratios, downsamples=downsamples,
                    down_patch_size = down_patch_size, down_pad=down_pad,
                    proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
                    heads=heads, head_dim=head_dim,
                    fork_feat=True,
                    **kwargs)


    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class context_cluster_small_feat5(ContextCluster):
        def __init__(self, **kwargs):
                layers = [2, 2, 6, 2]
                norm_layer=GroupNorm
                embed_dims = [64, 128, 320, 512]
                mlp_ratios = [8, 8, 4, 4]
                downsamples = [True, True, True, True]
                proposal_w=[5,5,5,5]
                proposal_h=[5,5,5,5]
                fold_w=[8,4,2,1]
                fold_h=[8,4,2,1]
                heads=[4,4,8,8]
                head_dim=[32,32,32,32]
                down_patch_size=3
                down_pad = 1
                super().__init__(
                    layers, embed_dims=embed_dims, norm_layer=norm_layer,
                    mlp_ratios=mlp_ratios, downsamples=downsamples,
                    down_patch_size = down_patch_size, down_pad=down_pad,
                    proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
                    heads=heads, head_dim=head_dim,
                    fork_feat=True,
                    **kwargs)


    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class context_cluster_small_feat7(ContextCluster):
        def __init__(self, **kwargs):
                layers = [2, 2, 6, 2]
                norm_layer=GroupNorm
                embed_dims = [64, 128, 320, 512]
                mlp_ratios = [8, 8, 4, 4]
                downsamples = [True, True, True, True]
                proposal_w=[7,7,7,7]
                proposal_h=[7,7,7,7]
                fold_w=[8,4,2,1]
                fold_h=[8,4,2,1]
                heads=[4,4,8,8]
                head_dim=[32,32,32,32]
                down_patch_size=3
                down_pad = 1
                super().__init__(
                    layers, embed_dims=embed_dims, norm_layer=norm_layer,
                    mlp_ratios=mlp_ratios, downsamples=downsamples,
                    down_patch_size = down_patch_size, down_pad=down_pad,
                    proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
                    heads=heads, head_dim=head_dim,
                    fork_feat=True,
                    **kwargs)


    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class context_cluster_medium_feat2(ContextCluster):
        def __init__(self, **kwargs):
                layers = [4, 4, 12, 4]
                norm_layer=GroupNorm
                embed_dims = [64, 128, 320, 512]
                mlp_ratios = [8, 8, 4, 4]
                downsamples = [True, True, True, True]
                proposal_w=[2,2,2,2]
                proposal_h=[2,2,2,2]
                fold_w=[8,4,2,1]
                fold_h=[8,4,2,1]
                heads=[6,6,12,12]
                head_dim=[32,32,32,32]
                down_patch_size=3
                down_pad = 1
                super().__init__(
                    layers, embed_dims=embed_dims, norm_layer=norm_layer,
                    mlp_ratios=mlp_ratios, downsamples=downsamples,
                    down_patch_size = down_patch_size, down_pad=down_pad,
                    proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
                    heads=heads, head_dim=head_dim,
                    fork_feat=True,
                    **kwargs)


    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class context_cluster_medium_feat5(ContextCluster):
        def __init__(self, **kwargs):
                layers = [4, 4, 12, 4]
                norm_layer=GroupNorm
                embed_dims = [64, 128, 320, 512]
                mlp_ratios = [8, 8, 4, 4]
                downsamples = [True, True, True, True]
                proposal_w=[5, 5, 5, 5]
                proposal_h=[5, 5, 5, 5]
                fold_w=[8,4,2,1]
                fold_h=[8,4,2,1]
                heads=[6,6,12,12]
                head_dim=[32,32,32,32]
                down_patch_size=3
                down_pad = 1
                super().__init__(
                    layers, embed_dims=embed_dims, norm_layer=norm_layer,
                    mlp_ratios=mlp_ratios, downsamples=downsamples,
                    down_patch_size = down_patch_size, down_pad=down_pad,
                    proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
                    heads=heads, head_dim=head_dim,
                    fork_feat=True,
                    **kwargs)


    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class context_cluster_medium_feat7(ContextCluster):
        def __init__(self, **kwargs):
                layers = [4, 4, 12, 4]
                norm_layer=GroupNorm
                embed_dims = [64, 128, 320, 512]
                mlp_ratios = [8, 8, 4, 4]
                downsamples = [True, True, True, True]
                proposal_w=[7,7,7,7]
                proposal_h=[7,7,7,7]
                fold_w=[8,4,2,1]
                fold_h=[8,4,2,1]
                heads=[6,6,12,12]
                head_dim=[32,32,32,32]
                down_patch_size=3
                down_pad = 1
                super().__init__(
                    layers, embed_dims=embed_dims, norm_layer=norm_layer,
                    mlp_ratios=mlp_ratios, downsamples=downsamples,
                    down_patch_size = down_patch_size, down_pad=down_pad,
                    proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
                    heads=heads, head_dim=head_dim,
                    fork_feat=True,
                    **kwargs)
# endregion

if __name__ == '__main__':
    input = torch.rand(2, 3, 224, 224)
    model = coc_base_dim64()
    out = model(input)
    print(model)
    print(out.shape)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params: {:.2f}M".format(n_parameters/1024**2))
