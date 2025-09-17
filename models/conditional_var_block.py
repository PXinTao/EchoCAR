import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

# 依赖原始 VAR
try:
    from models.var import VAR
except Exception:
    # 兼容直接运行
    from var import VAR


def _concat_kv_pairs(kv_pairs: dict, d_model: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    统一的 K/V 拼接函数（稳健版）
    输入: kv_pairs 形如 {'edge': (K,V), 'mask': (K,V), 'coarse': (K,V) ...}
         也兼容 (K,V, gate_or_mask)
         - 若第三项是 bool/uint8 张量 => 作为 padding mask（True=padding）
         - 若第三项是浮点张量或标量 => 作为 gate（经 sigmoid）
         - 若 value 是单个张量 => 视为 K，且 V=K
    输出: (K_all, V_all)；若没有任何条件，返回 (None, None)
    """
    if not kv_pairs:
        return None, None

    Ks, Vs = [], []

    for name, pack in kv_pairs.items():
        if pack is None:
            continue

        # 解析不同输入格式
        if isinstance(pack, (tuple, list)):
            if len(pack) >= 2:
                K, V = pack[0], pack[1]
                # 处理第三项：mask 或 gate
                if len(pack) >= 3:
                    third = pack[2]
                    if third is not None:
                        # 区分 mask vs gate
                        if torch.is_tensor(third) and third.dtype in (torch.bool, torch.uint8):
                            # padding mask: True 表示 padding
                            m = third.bool()
                            if m.dim() == 2:  # [B, G]
                                m = m.unsqueeze(-1)  # -> [B, G, 1]
                            K = K.masked_fill(m, 0)
                            V = V.masked_fill(m, 0)
                        else:
                            # gate：标量或浮点张量（做 dtype/device 对齐）
                            if torch.is_tensor(third):
                                gate_val = torch.sigmoid(third.to(device=K.device, dtype=K.dtype))
                            else:
                                gate_val = torch.tensor(float(third), device=K.device, dtype=K.dtype)
                            K = K * gate_val
                            V = V * gate_val
            else:
                continue
        else:
            # 直接给张量 => K=V=pack
            K = V = pack

        # 数值稳定性处理
        K = torch.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)
        V = torch.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)

        # 形状检查
        if K.dim() != 3 or V.dim() != 3:
            print(f"Warning: {name} K/V shape not [B,G,D]: K={tuple(K.shape)}, V={tuple(V.shape)}")
            continue

        Ks.append(K)
        Vs.append(V)

    if not Ks:
        return None, None

    K_all = torch.cat(Ks, dim=1)  # [B, G_total, D]
    V_all = torch.cat(Vs, dim=1)  # [B, G_total, D]

    # （可选）无参数层归一化 —— 建议在 Cross-Attn 内做 LN，这里默认不做
    if d_model is not None:
        K_all = F.layer_norm(K_all, (d_model,), eps=1e-6)
        V_all = F.layer_norm(V_all, (d_model,), eps=1e-6)
        # 维度断言（早期抓错）
        assert K_all.size(-1) == d_model and V_all.size(-1) == d_model, \
            f"KV dim mismatch: expected {d_model}, got K:{K_all.size(-1)}, V:{V_all.size(-1)}"

    return K_all, V_all


class GatedCrossAttention(nn.Module):
    """
    稳定版 Cross-Attn：
    - Q 来自主序列，K/V 来自条件
    - 输入/条件都先过 LN
    - Q/K/V Linear 后，注意力在 float32 中用 SDPA 计算
    - 输出 nan_to_num，最后门控残差
    """
    def __init__(self, d_model: int, n_heads: int, gate_init: float = -6.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # LayerNorm for stability
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_k = nn.LayerNorm(d_model)
        self.ln_v = nn.LayerNorm(d_model)

        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        # 门控：初始化为负值，训练早期尽量少干扰
        self.gate = nn.Parameter(torch.tensor(gate_init, dtype=torch.float32))

    def forward(self, x: torch.Tensor, kv_pairs: Optional[dict]) -> torch.Tensor:
        """
        x: [B, L, D] 主序列（来自 VAR blocks）
        kv_pairs: dict 条件字典，形如 {'edge': (K,V), 'mask': (K,V), ...}
        """
        if kv_pairs is None:
            return x

        B, L, D = x.shape

        # 拼接所有条件的 K/V —— 为避免“双重 LN”，这里不做 LN
        K_all, V_all = _concat_kv_pairs(kv_pairs, d_model=None)
        if K_all is None or V_all is None:
            return x

        # 设备和类型对齐
        K_all = K_all.to(device=x.device, dtype=x.dtype)
        V_all = V_all.to(device=x.device, dtype=x.dtype)

        # 预归一化 + 线性
        q = self.q_proj(self.ln_q(x))          # [B, L, D]
        k = self.k_proj(self.ln_k(K_all))      # [B, G, D]
        v = self.v_proj(self.ln_v(V_all))      # [B, G, D]

        # 重塑为多头格式
        def reshape_for_attn(t, S):
            return t.view(B, S, self.n_heads, self.head_dim).transpose(1, 2).contiguous()  # [B, H, S, Hd]

        q = reshape_for_attn(q, L)                    # [B, H, L, Hd]
        k = reshape_for_attn(k, K_all.size(1))        # [B, H, G, Hd]
        v = reshape_for_attn(v, V_all.size(1))        # [B, H, G, Hd]

        # 在 float32 中计算注意力（数值稳定）
        q_f = q.float()
        k_f = k.float()
        v_f = v.float()

        # SDPA (no causal mask for cross-attention)
        try:
            y = F.scaled_dot_product_attention(
                q_f, k_f, v_f,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False
            )  # [B, H, L, Hd]
        except Exception as e:
            print(f"SDPA failed: {e}, fallback to manual attention")
            # 手动注意力作为 fallback
            scale = 1.0 / (self.head_dim ** 0.5)
            attn = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale  # [B, H, L, G]
            attn = F.softmax(attn, dim=-1)
            y = torch.matmul(attn, v_f)  # [B, H, L, Hd]

        # 还原形状
        y = y.transpose(1, 2).contiguous().view(B, L, D)

        # out_proj：先转到权重dtype再线性，最后转回 x.dtype
        y = self.out_proj(y.to(self.out_proj.weight.dtype)).to(x.dtype)

        # 数值稳定性保护
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        # 门控残差连接
        gate_val = torch.sigmoid(self.gate)
        return x + gate_val * y


class ConditionalVARWrapper(nn.Module):
    """
    包装原始 VAR，在每个 block 之后插入 Gated Cross-Attn（前 open_cross_n 层开启），
    并在 AdaLN 条件向量 cond_BD 里融合文本向量 text_BD（门控相加）。
    注意：不改动原 Block，只在外壳 forward 里做插入。
    """
    def __init__(
        self,
        base_var: VAR,
        n_heads_for_cross: Optional[int] = None,
        open_cross_n: int = 3,                    # 仅前K层开启 Cross-Attn
        fuse_text_init: float = -2.0,             # 文本门控初始值
    ):
        super().__init__()
        self.var = base_var
        self.open_cross_n = int(open_cross_n)

        # 更稳的 d_model 推断
        if hasattr(self.var, "pos_1LC") and self.var.pos_1LC is not None:
            d_model = int(self.var.pos_1LC.shape[-1])
        elif hasattr(self.var, "embed_dim"):
            d_model = int(self.var.embed_dim)
        elif hasattr(self.var, "C"):
            d_model = int(self.var.C)
        else:
            # 最后的保底：通过一个 dummy token 推断
            dummy_token = torch.zeros(1, 1, dtype=torch.long, device=next(self.var.parameters()).device)
            with torch.no_grad():
                d_model = int(self.var.word_embed(dummy_token.float()).shape[-1])

        if n_heads_for_cross is None:
            # 复用原 VAR 的 heads 数量
            if hasattr(self.var, "num_heads"):
                n_heads_for_cross = int(self.var.num_heads)
            else:
                n_heads_for_cross = 8

        print(f"[ConditionalVARWrapper] d_model={d_model}, n_heads={n_heads_for_cross}, open_cross_n={self.open_cross_n}")

        # 为每个 Transformer block 准备一个 Cross-Attn
        self.cross_layers = nn.ModuleList([
            GatedCrossAttention(d_model=d_model, n_heads=n_heads_for_cross, gate_init=-6.0)
            for _ in self.var.blocks
        ])

        # 文本门控（把 text_BD 融进 cond_BD）
        self.text_gate = nn.Parameter(torch.tensor(float(fuse_text_init)))

        # 存储 d_model
        self.d_model = d_model

    @property
    def device(self):
        return next(self.parameters()).device

    def _fuse_text_cond(self, cond_BD_base: torch.Tensor, text_BD: Optional[torch.Tensor]) -> torch.Tensor:
        """
        cond_BD_base: 原始 AdaLN 条件（通常来源于 class_emb）
        text_BD:      文本适配器输出 [B, D]
        融合：cond = cond_base + σ(g_text) * text_BD
        """
        if text_BD is None:
            return cond_BD_base

        # 维度适配
        if text_BD.shape[-1] != cond_BD_base.shape[-1]:
            if not hasattr(self, "_text_proj"):
                self._text_proj = nn.Linear(
                    text_BD.shape[-1], cond_BD_base.shape[-1], bias=False
                ).to(device=self.device, dtype=cond_BD_base.dtype)
            text_BD = self._text_proj(text_BD)

        return cond_BD_base + torch.sigmoid(self.text_gate) * text_BD

    def forward(
        self,
        label_B: torch.Tensor,                              # [B]
        x_BLCv_wo_first_l: torch.Tensor,                   # VAE 特征（原 VAR 输入）
        kv_pairs: Optional[Dict[str, Tuple[torch.Tensor, ...]]] = None,  # 视觉条件 K/V 字典
        text_BD: Optional[torch.Tensor] = None,            # 文本向量 [B, D]
        open_cross_n: Optional[int] = None,                # 可临时覆盖
    ) -> torch.Tensor:
        """
        返回：logits_BLV（与原 VAR 一致）
        """
        if open_cross_n is None:
            open_cross_n = self.open_cross_n

        B = x_BLCv_wo_first_l.shape[0]

        # ==== 1) 组装 SOS + 位置/层级嵌入（完全复用 VAR） ====
        if self.var.prog_si >= 0:
            bg, ed = self.var.begin_ends[self.var.prog_si]
        else:
            bg, ed = (0, self.var.L)

        # 用更通用的 autocast（CPU/GPU 都安全）
        with torch.autocast(device_type=x_BLCv_wo_first_l.device.type, enabled=False):
            # class_emb → base cond（原始路径）
            cond_BD_base = self.var.class_emb(label_B)  # [B, D]

            # 融合文本条件
            cond_BD_fused = self._fuse_text_cond(cond_BD_base, text_BD)

            # 起始 token
            sos = cond_BD_fused.unsqueeze(1).expand(B, self.var.first_l, -1) + \
                  self.var.pos_start.expand(B, self.var.first_l, -1)

            if self.var.prog_si == 0:
                x_BLC = sos
            else:
                x_BLC = torch.cat((sos, self.var.word_embed(x_BLCv_wo_first_l.float())), dim=1)

            x_BLC += self.var.lvl_embed(self.var.lvl_1L[:, :ed].expand(B, -1)) + \
                     self.var.pos_1LC[:, :ed]

        attn_bias = self.var.attn_bias_for_masking[:, :, :ed, :ed]

        # AdaLN 参数生成（shared_ada_lin 缺失时回退）
        need6 = self.d_model * 6
        if hasattr(self.var, 'shared_ada_lin') and self.var.shared_ada_lin is not None:
            cond_ada_6D = self.var.shared_ada_lin(cond_BD_fused)
        else:
            if cond_BD_fused.shape[-1] != need6:
                if not hasattr(self, "_ada_fallback"):
                    self._ada_fallback = nn.Linear(
                        cond_BD_fused.shape[-1], need6, bias=False
                    ).to(device=self.device, dtype=cond_BD_fused.dtype)
                cond_ada_6D = self._ada_fallback(cond_BD_fused)
            else:
                cond_ada_6D = cond_BD_fused

        # dtype 对齐
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        x_BLC = x_BLC.to(dtype=main_type)
        cond_ada_6D = cond_ada_6D.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)

        # ==== 2) 迭代每个 Block；在前 K 层后插入 Cross-Attn ====
        for li, blk in enumerate(self.var.blocks):
            # 原始 VAR Block
            x_BLC = blk(x=x_BLC, cond_BD=cond_ada_6D, attn_bias=attn_bias)

            # 仅前 K 层插 Cross-Attn
            if li < open_cross_n and kv_pairs is not None:
                x_BLC = self.cross_layers[li](x_BLC, kv_pairs)

        # ==== 3) 输出 logits（与原 VAR 一致） ====
        logits_BLV = self.var.get_logits(x_BLC.float(), cond_BD_fused)
        return logits_BLV


def wrap_var_with_condition(
    base_var: VAR,
    n_heads_for_cross: Optional[int] = None,
    open_cross_n: int = 3,
    fuse_text_init: float = -2.0,
) -> ConditionalVARWrapper:
    """
    方便的包装器构造函数。
    用法：
        var = build_your_var(...)
        cond_var = wrap_var_with_condition(var, open_cross_n=3)
        logits = cond_var(label_B, x_BLCv_wo_first_l, kv_pairs=..., text_BD=...)
    """
    return ConditionalVARWrapper(
        base_var=base_var,
        n_heads_for_cross=n_heads_for_cross,
        open_cross_n=open_cross_n,
        fuse_text_init=fuse_text_init,
    )
