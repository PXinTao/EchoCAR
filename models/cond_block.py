# -*- coding: utf-8 -*-
"""
条件化 Transformer Block：在自注意与前馈之间插入 Cross-Attn
- 仅在每个 stage 的前 open_cross_layers 层启用 Cross-Attn
- 每个 stage 一枚可学习门控（负值初始化），Sigmoid 后乘到 Cross-Attn 残差
- 支持 AdaLN（用 cond_BD 做缩放/平移）
- 独立可跑，下一步将接到 VAR 的 Block 外壳中
"""

import math
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaLN(nn.Module):
    """最小可用 AdaLN：用 cond 产生 (scale, shift)，作用于 LayerNorm 输出"""
    def __init__(self, d_model: int, cond_dim: Optional[int] = None):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.cond_dim = cond_dim or d_model
        self.to_affine = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.cond_dim, 2 * d_model)
        )

    def forward(self, x_BLD: torch.Tensor, cond_BD: torch.Tensor) -> torch.Tensor:
        B, L, D = x_BLD.shape
        aff = self.to_affine(cond_BD)  # [B, 2D]
        scale, shift = aff.chunk(2, dim=-1)  # [B, D], [B, D]
        x = self.ln(x_BLD)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x


class MLP(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))


class CrossAttention(nn.Module):
    """batch_first MultiheadAttention 的简封装 + 可选 attn_mask"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        # PyTorch MHA 自带输出投影和 dropout

    def forward(
        self,
        q_BLD: torch.Tensor,
        k_BGD: torch.Tensor,
        v_BGD: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # attn_mask 形状支持 [Lq, Lk] 或 [B * num_heads, Lq, Lk]；这里使用简型 [Lq, Lk]
        out, _ = self.attn(q_BLD, k_BGD, v_BGD, attn_mask=attn_mask, need_weights=False)
        return out


class CondTransformerBlock(nn.Module):
    """
    结构：x -> (AdaLN+SA) -> +res -> [可选 Cross] -> +res -> (AdaLN+FFN) -> +res
    - open_cross_layers: 仅 stage 内前 N 层启用 Cross
    - stage_gates: 每个 stage 一枚门控（Sigmoid 后乘到 Cross 残差）
    - attn_bias: 若给定 (如 VAR 的 [1,1,L,L])，会转换为 MHA 的 attn_mask（下三角）
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_stages: int = 5,
        open_cross_layers: int = 3,
        gate_init: float = -2.0,
        cond_dim: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.open_cross_layers = open_cross_layers

        # 自注意
        self.ada1 = AdaLN(d_model, cond_dim=cond_dim)
        self.sa = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        # Cross-Attn + stage 门控
        self.cross = CrossAttention(d_model, n_heads, dropout=dropout)
        self.stage_gates = nn.ParameterList([nn.Parameter(torch.tensor(float(gate_init))) for _ in range(num_stages)])

        # 前馈
        self.ada2 = AdaLN(d_model, cond_dim=cond_dim)
        self.mlp = MLP(d_model, mlp_ratio=mlp_ratio, dropout=dropout)
        self.drop2 = nn.Dropout(dropout)

    @staticmethod
    def _to_attn_mask(attn_bias: Optional[torch.Tensor], Lq: int, Lk: int, device) -> Optional[torch.Tensor]:
        """把 VAR 风格的 attn_bias(形如 [1,1,L,L], 下三角=0 上三角=-inf) 转为 MHA attn_mask [Lq, Lk]"""
        if attn_bias is None:
            return None
        # 允许简化：若给定为 [L, L] 或 [1,1,L,L]
        if attn_bias.dim() == 4:
            mask = attn_bias[0, 0, :Lq, :Lk]
        elif attn_bias.dim() == 2:
            mask = attn_bias[:Lq, :Lk]
        else:
            return None
        # MHA 期望的是“禁止位置 = -inf”，允许位置 = 0
        return mask.to(device=device)

    @staticmethod
    def _concat_kv(kv_pairs: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not kv_pairs:
            return None, None
        Ks, Vs = [], []
        for _, (K, V) in kv_pairs.items():
            Ks.append(K)  # [B,G,D]
            Vs.append(V)
        K_cat = torch.cat(Ks, dim=1) if len(Ks) else None  # [B, sumG, D]
        V_cat = torch.cat(Vs, dim=1) if len(Vs) else None
        return K_cat, V_cat

    def forward(
        self,
        x_BLD: torch.Tensor,
        cond_BD: torch.Tensor,
        *,
        attn_bias: Optional[torch.Tensor] = None,
        kv_pairs: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
        stage_id: int = 0,
        layer_offset_in_stage: int = 0,
    ) -> torch.Tensor:
        B, L, D = x_BLD.shape
        device = x_BLD.device

        # ---- 自注意 ----
        h1 = self.ada1(x_BLD, cond_BD)                 # AdaLN 规范化+调制
        sa_mask = self._to_attn_mask(attn_bias, L, L, device)
        sa_out, _ = self.sa(h1, h1, h1, attn_mask=sa_mask, need_weights=False)
        x = x_BLD + self.drop1(sa_out)

        # ---- 条件 Cross-Attn（仅前 N 层启用）----
        use_cross = (kv_pairs is not None) and (layer_offset_in_stage < self.open_cross_layers)
        if use_cross:
            K_cat, V_cat = self._concat_kv(kv_pairs)
            if (K_cat is not None) and (V_cat is not None) and (K_cat.size(0) == B):
                cross_out = self.cross(x, K_cat, V_cat, attn_mask=None)  # Cross 不做自回归 mask
                gate = torch.sigmoid(self.stage_gates[stage_id]) if (0 <= stage_id < len(self.stage_gates)) else torch.tensor(1.0, device=device)
                x = x + gate * cross_out

        # ---- FFN ----
        h2 = self.ada2(x, cond_BD)
        x = x + self.drop2(self.mlp(h2))
        return x
