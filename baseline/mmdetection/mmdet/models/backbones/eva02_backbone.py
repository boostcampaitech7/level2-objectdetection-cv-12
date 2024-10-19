# mmdet/models/backbones/eva02_backbone.py

import math
from collections import OrderedDict
from copy import deepcopy
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmengine.logging import MMLogger
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import constant_init, trunc_normal_
from mmengine.runner.checkpoint import CheckpointLoader
from mmengine.utils import to_2tuple

from mmdet.registry import MODELS
from timm.models.layers import PatchEmbed, LayerNorm, DropPath

@MODELS.register_module()
class EVA02Backbone(BaseModule):
    """EVA02 Backbone for use in mmdetection.

    Args:
        img_size (int | tuple): Input image size. Defaults to 448.
        patch_size (int | tuple): Patch size. Defaults to 14.
        in_chans (int): Number of input channels. Defaults to 3.
        embed_dim (int): Embedding dimension. Defaults to 1024.
        depth (int): Number of transformer blocks. Defaults to 24.
        num_heads (int): Number of attention heads. Defaults to 16.
        mlp_ratio (float): MLP expansion ratio. Defaults to 8/3.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Defaults to True.
        drop_rate (float): Dropout rate. Defaults to 0.
        attn_drop_rate (float): Attention dropout rate. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.1.
        norm_cfg (dict): Normalization layer config. Defaults to dict(type='LN').
        out_indices (tuple[int]): Output from which stages. Defaults to (7, 11, 15, 23).
        with_cp (bool): Use checkpointing or not. Defaults to False.
        init_cfg (dict | None): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 img_size=448,
                 patch_size=14,
                 in_chans=3,
                 embed_dim=1024,
                 depth=24,
                 num_heads=16,
                 mlp_ratio=8 / 3,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_cfg=dict(type='LN'),
                 out_indices=(7, 11, 15, 23),
                 with_cp=False,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.embed_dim = embed_dim
        self.out_indices = out_indices
        self.with_cp = with_cp

        # 패치 임베딩
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.num_prefix_tokens = 1  # cls_token

        # 클래스 토큰 및 위치 임베딩
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_prefix_tokens, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer 블록
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            EvaBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_cfg=norm_cfg,
            )
            for i in range(depth)
        ])

        # 최종 LayerNorm
        self.norm = build_norm_layer(norm_cfg, embed_dim)[1]

        # 출력에 대한 LayerNorm 추가
        for i in out_indices:
            layer = build_norm_layer(norm_cfg, embed_dim)[1]
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

        # 가중치 초기화
        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m.bias, nn.Parameter):
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def init_weights(self):
        logger = MMLogger.get_current_instance()
        if self.init_cfg is None:
            logger.warning('No pre-trained weights for EVA02Backbone, training from scratch')
            return
        else:
            checkpoint = self.init_cfg.get('checkpoint', None)
            if checkpoint is None:
                logger.warning('No checkpoint specified in init_cfg')
                return
            else:
                load_checkpoint(self, checkpoint, strict=False, logger=logger)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # B, C, H', W'
        Wh, Ww = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # B, N, C

        cls_tokens = self.cls_token.expand(B, -1, -1)  # B, 1, C
        x = torch.cat((cls_tokens, x), dim=1)  # B, 1+N, C
        x = x + self.pos_embed  # 위치 임베딩 추가
        x = self.pos_drop(x)  # 드롭아웃

        outs = []
        for i, blk in enumerate(self.blocks):
            if self.with_cp and x.requires_grad:
                x = cp.checkpoint(blk, x)
            else:
                x = blk(x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(x)
                out = out[:, 1:]  # cls_token 제외
                B, N, C = out.shape
                out = out.transpose(1, 2).reshape(B, C, Wh, Ww).contiguous()
                outs.append(out)

        return tuple(outs)

class EvaAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x)  # B, N, 3*C
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)  # B, N, 3, num_heads, head_dim
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B, num_heads, N, head_dim

        q, k, v = qkv[0], qkv[1], qkv[2]  # 각각의 텐서 크기: B, num_heads, N, head_dim

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, num_heads, N, N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v)  # B, num_heads, N, head_dim
        x = x.transpose(1, 2).reshape(B, N, C)  # B, N, C

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class EvaBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=8 / 3, qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0., norm_cfg=dict(type='LN')):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.attn = EvaAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FFN(
            embed_dims=dim,
            feedforward_channels=mlp_hidden_dim,
            num_fcs=2,
            ffn_drop=drop,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path),
            act_cfg=dict(type='GELU'),
            add_identity=True,
            init_cfg=None
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))  # 첫 번째 잔차 연결
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # 두 번째 잔차 연결
        return x
