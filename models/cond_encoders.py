# -*- coding: utf-8 -*-
"""
条件编码器：把 (edge/mask/sketch/fan/coarse) 编码为 Cross-Attn 的 (K, V) 令牌
用法：
  pack = CondEncPack(d_model=768, num_mask_classes=4, latent_ch=32)
  kv = pack({'edge': e, 'mask': m, 'sketch': s, 'fan': f, 'coarse_img': p_img, 'coarse_lat': p_lat}, grid=8)
  -> kv['edge'] = (K_edge, V_edge)   形状 [B, G, D],  G=grid*grid
注意：
  - 所有编码器都有一个可学习门控（标量），初始化为 -2，前向时会乘以 sigmoid(gate)
  - coarse 同时支持图像前缀（coarse_img，范围建议 [0,1]）和潜空间前缀（coarse_lat，[B, latent_ch, H, W]）
  - mask 输入：整数标签，若是 float 会自动 round/clip 到 [0, num_classes-1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


def _gn(ch: int, max_groups: int = 32) -> nn.GroupNorm:
    # 自动选择 group 数，避免不能整除
    g = 1
    for cand in [32, 16, 8, 4, 2]:
        if ch % cand == 0 and cand <= max_groups:
            g = cand
            break
    return nn.GroupNorm(g, ch)


class _Gate(nn.Module):
    def __init__(self, init_value: float = -2.0):
        super().__init__()
        self.g = nn.Parameter(torch.tensor(float(init_value)))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.g) * x
    @property
    def value(self) -> float:
        return torch.sigmoid(self.g).item()


class _ToTokens(nn.Module):
    """把 [B,C,H,W] 池化到 (grid,grid) 并展平为 [B, G, C]"""
    def forward(self, x: torch.Tensor, grid: int) -> torch.Tensor:
        x = F.adaptive_avg_pool2d(x, (grid, grid))  # [B,C,g,g]
        B, C, g, _ = x.shape
        return x.view(B, C, g*g).permute(0, 2, 1).contiguous()  # [B,G,C]


class _KVProj(nn.Module):
    """把通道 C -> D，并产生 K,V： [B,G,C] -> [B,G,D] x2"""
    def __init__(self, in_ch: int, d_model: int):
        super().__init__()
        self.proj_in = nn.Linear(in_ch, d_model) if in_ch != d_model else nn.Identity()
        self.proj_k  = nn.Linear(d_model, d_model)
        self.proj_v  = nn.Linear(d_model, d_model)
        self.norm    = nn.LayerNorm(d_model)
        self.gate    = _Gate(-2.0)
    def forward(self, x_bgC: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.proj_in(x_bgC)
        x = self.norm(x)
        K = self.proj_k(x); V = self.proj_v(x)
        s = torch.sigmoid(self.gate.g)
        return s * K, s * V  # 门控在输出侧统一生效


class EncEdge(nn.Module):
    """边缘图编码器：输入 [B,1,H,W] -> (K,V) [B,G,D]"""
    def __init__(self, d_model: int, in_ch: int = 1, mid: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, 2, 1), _gn(64), nn.SiLU(),
            nn.Conv2d(64, mid, 3, 2, 1), _gn(mid), nn.SiLU(),
            nn.Conv2d(mid, d_model, 3, 1, 1), _gn(d_model), nn.SiLU(),
        )
        self.to_tokens = _ToTokens()
        self.kv = _KVProj(d_model, d_model)

    def forward(self, x_B1HW: torch.Tensor, grid: int) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.backbone(x_B1HW)               # [B,D,H',W']
        tok = self.to_tokens(f, grid)           # [B,G,D]
        return self.kv(tok)                     # (K,V)


class EncSketch(EncEdge):
    """草图编码器：结构与 Edge 相同"""
    pass


class EncFan(nn.Module):
    """扇形边界编码器：输入 [B,1,H,W] -> (K,V)"""
    def __init__(self, d_model: int, in_ch: int = 1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_ch, 32, 7, 1, 3), _gn(32), nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2), _gn(64), nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, d_model, 3, 1, 1), _gn(d_model), nn.SiLU(),
        )
        self.to_tokens = _ToTokens()
        self.kv = _KVProj(d_model, d_model)

    def forward(self, fan_B1HW: torch.Tensor, grid: int) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.backbone(fan_B1HW)
        tok = self.to_tokens(f, grid)
        return self.kv(tok)


class EncMask(nn.Module):
    """分割掩码编码器：输入整数标签 [B,1,H,W] -> (K,V)"""
    def __init__(self, num_classes: int, d_model: int, emb_ch: int = None):
        super().__init__()
        if emb_ch is None:
            emb_ch = max(16, d_model // 4)
        self.num_classes = num_classes
        self.emb = nn.Embedding(num_classes, emb_ch)
        self.pix_proj = nn.Sequential(
            nn.Conv2d(emb_ch, d_model // 2, 3, 2, 1), _gn(d_model // 2), nn.SiLU(),
            nn.Conv2d(d_model // 2, d_model, 3, 2, 1), _gn(d_model), nn.SiLU(),
        )
        self.to_tokens = _ToTokens()
        self.kv = _KVProj(d_model, d_model)

    def forward(self, mask_B1HW: torch.Tensor, grid: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if mask_B1HW.dtype.is_floating_point:
            mask = mask_B1HW.round().clamp_(0, self.num_classes - 1).long().squeeze(1)  # [B,H,W]
        else:
            mask = mask_B1HW.squeeze(1).long()
            mask = mask.clamp_(0, self.num_classes - 1)
        m = self.emb(mask)                       # [B,H,W,emb]
        m = m.permute(0, 3, 1, 2).contiguous()   # [B,emb,H,W]
        f = self.pix_proj(m)                     # [B,D,H',W']
        tok = self.to_tokens(f, grid)            # [B,G,D]
        return self.kv(tok)


class EncCoarse(nn.Module):
    """
    粗粒度前缀编码器：
      - 支持图像前缀 coarse_img: [B,1,H,W]（建议范围 [0,1]）
      - 支持潜空间前缀 coarse_lat: [B,latent_ch,H,W]（来自 RVQ 的累计 latent）
    优先使用 coarse_lat；两者都给时会 concat 后再投影。
    """
    def __init__(self, d_model: int, latent_ch: int = 32):
        super().__init__()
        self.img_enc = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), _gn(64), nn.SiLU(),
            nn.Conv2d(64, d_model, 3, 1, 1), _gn(d_model), nn.SiLU(),
        )
        self.lat_enc = nn.Sequential(
            nn.Conv2d(latent_ch, d_model, 1, 1, 0), _gn(d_model), nn.SiLU(),
        )
        self.fuse = nn.Sequential(              # 若两路并存，做轻量融合
            nn.Conv2d(2 * d_model, d_model, 1, 1, 0), _gn(d_model), nn.SiLU(),
        )
        self.to_tokens = _ToTokens()
        self.kv = _KVProj(d_model, d_model)

    def forward(
        self,
        grid: int,
        coarse_img: Optional[torch.Tensor] = None,
        coarse_lat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = []
        if coarse_lat is not None:
            feats.append(self.lat_enc(coarse_lat))
        if coarse_img is not None:
            feats.append(self.img_enc(coarse_img))
        if not feats:
            # 都没给，返回全零（训练时通常会有至少一项）
            raise ValueError("EncCoarse requires at least one of (coarse_img, coarse_lat).")
        f = feats[0] if len(feats) == 1 else self.fuse(torch.cat(feats, dim=1))
        tok = self.to_tokens(f, grid)           # [B,G,D]
        return self.kv(tok)


class CondEncPack(nn.Module):
    """
    打包器：把输入字典 -> 各路 (K,V)
    支持键：
      - 'edge' : [B,1,H,W]
      - 'sketch': [B,1,H,W]
      - 'fan' : [B,1,H,W]
      - 'mask': [B,1,H,W] (int 或 float 标签)
      - 'coarse_img': [B,1,H,W]
      - 'coarse_lat': [B,Lc,H,W]  (Lc = latent_ch)
    """
    def __init__(self, d_model: int, num_mask_classes: int = 4, latent_ch: int = 32):
        super().__init__()
        self.enc_edge   = EncEdge(d_model)
        self.enc_sketch = EncSketch(d_model)
        self.enc_fan    = EncFan(d_model)
        self.enc_mask   = EncMask(num_classes=num_mask_classes, d_model=d_model)
        self.enc_coarse = EncCoarse(d_model, latent_ch=latent_ch)

    @torch.no_grad()
    def _drop(self, x: Optional[torch.Tensor], p: float) -> Optional[torch.Tensor]:
        if x is None: return None
        if p <= 0:    return x
        return None if torch.rand(1).item() < p else x

    def forward(
        self,
        conds: Dict[str, torch.Tensor],
        grid: int,
        dropout: Dict[str, float] = None
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        dropout = dropout or {}
        out: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

        edge   = self._drop(conds.get('edge'),   dropout.get('edge',   0.0))
        sketch = self._drop(conds.get('sketch'), dropout.get('sketch', 0.0))
        fan    = self._drop(conds.get('fan'),    dropout.get('fan',    0.0))
        mask   = self._drop(conds.get('mask'),   dropout.get('mask',   0.0))

        if edge   is not None: out['edge']   = self.enc_edge(edge, grid)
        if sketch is not None: out['sketch'] = self.enc_sketch(sketch, grid)
        if fan    is not None: out['fan']    = self.enc_fan(fan, grid)
        if mask   is not None: out['mask']   = self.enc_mask(mask, grid)

        # coarse：允许 img / lat 单独或同时存在；dropout 分别控制
        coarse_img = self._drop(conds.get('coarse_img'), dropout.get('coarse_img', 0.0))
        coarse_lat = self._drop(conds.get('coarse_lat'), dropout.get('coarse_lat', 0.0))
        if (coarse_img is not None) or (coarse_lat is not None):
            out['coarse'] = self.enc_coarse(grid, coarse_img=coarse_img, coarse_lat=coarse_lat)

        return out
