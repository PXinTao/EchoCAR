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


class GatedCrossAttention(nn.Module):
# class CrossAttention(nn.Module):
    """
    稳定版 Cross-Attn：
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

        self.ln_q = nn.LayerNorm(d_model)
        self.ln_k = nn.LayerNorm(d_model)
        self.ln_v = nn.LayerNorm(d_model)

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        # 门控更负：训练早期尽量少干扰
        self.gate = nn.Parameter(torch.tensor(gate_init, dtype=torch.float32))

    def forward(self, x: torch.Tensor, kv_pairs: dict | None):
        """
        x: [B, L, D]
        kv_pairs: dict -> (K_all, V_all)  [B, G, D]
        """
        if kv_pairs is None:
            return x

        B, L, D = x.shape
        K_all, V_all = _concat_kv_tokens(kv_pairs, d_model=D)

        if K_all is None or V_all is None:
            return x

        # 预归一化
        q = self.q_proj(self.ln_q(x))          # [B, L, D]
        k = self.k_proj(self.ln_k(K_all))      # [B, G, D]
        v = self.v_proj(self.ln_v(V_all))      # [B, G, D]

        # 到多头形状，并在 float32 做 SDPA
        def _reshape(t, S):
            return t.view(B, S, self.n_heads, self.head_dim).transpose(1, 2).contiguous()  # [B, H, S, Hd]

        q_f = _reshape(q, L).float()
        k_f = _reshape(k, K_all.size(1)).float()
        v_f = _reshape(v, V_all.size(1)).float()

        # 数值稳定的注意力（no mask, no dropout）
        y = F.scaled_dot_product_attention(
            q_f, k_f, v_f,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False
        )  # [B, H, L, Hd]

        # 还原形状 + cast 回原 dtype
        y = y.transpose(1, 2).contiguous().view(B, L, D).to(x.dtype)
        y = self.out_proj(y)

        # 保底卫生
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        g = torch.sigmoid(self.gate)
        return x + g * y


def _concat_kv_tokens(kv_pairs: dict, d_model: int = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    输入: kv_pairs 形如 {'edge': (K,V), 'mask': (K,V), 'coarse_img': (K,V) ...}
         也兼容 (K,V,gate) 第三项会被忽略（门控在外层做）
    处理: 对每一路 K/V 先做 nan_to_num + LayerNorm，再 concat 到 [B, G_all, D]
    输出: (K_all, V_all)；若没有任何条件，返回 (None, None)
    """
    Ks, Vs = [], []
    ln = None

    for name, pack in kv_pairs.items():
        if pack is None:
            continue
        if isinstance(pack, (tuple, list)):
            if len(pack) >= 2:
                K, V = pack[0], pack[1]
            else:
                continue
        else:
            # 非法条目，跳过
            continue

        # 数值卫生
        K = torch.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)
        V = torch.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)

        # LayerNorm 到 D 维（若没给 d_model，则从张量推）
        if d_model is None:
            d_model = K.shape[-1]
        if ln is None:
            ln = nn.LayerNorm(d_model).to(K.device)

        K = ln(K)
        V = ln(V)
        Ks.append(K)
        Vs.append(V)

    if len(Ks) == 0:
        return None, None

    K_all = torch.cat(Ks, dim=1)  # [B, G_all, D]
    V_all = torch.cat(Vs, dim=1)  # [B, G_all, D]
    return K_all, V_all


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

        # 猜测 D（嵌入维度）与 heads
        # VAR 里通常有 self.C 作为 embed_dim；若没有，就从位置/词嵌入推断
        if hasattr(self.var, "C"):
            d_model = int(self.var.C)
        else:
            # fallback: 通过 pos_1LC 或 word_embed 推断
            if hasattr(self.var, "pos_1LC"):
                d_model = int(self.var.pos_1LC.shape[-1])
            else:
                # 最后的保底：尝试用一个 token 过 embed 看维度
                tmp = torch.randn(1, 1, device=next(self.var.parameters()).device, dtype=torch.long)
                d_model = int(self.var.word_embed(tmp).shape[-1])

        if n_heads_for_cross is None:
            # 若原 VAR 有多头数量可用就复用；否则默认 8
            if hasattr(self.var, "num_heads"):
                n_heads_for_cross = int(self.var.num_heads)
            else:
                n_heads_for_cross = 8

        # 为每个 Transformer block 准备一个 Cross-Attn（是否启用由 open_cross_n 控制）
        self.cross_layers = nn.ModuleList(
            [GatedCrossAttention(d_model=d_model, n_heads=n_heads_for_cross) for _ in self.var.blocks]
        )

        # 文本门控（把 text_BD 融进 cond_BD）
        self.text_gate = nn.Parameter(torch.tensor(float(fuse_text_init)))

    @property
    def device(self):
        return next(self.parameters()).device

    def _fuse_text_cond(self, cond_BD_base: torch.Tensor, text_BD: Optional[torch.Tensor]) -> torch.Tensor:
        """
        cond_BD_base: 原始 AdaLN 条件（通常来源于 class_emb 或你的虚拟类）
        text_BD:      文本适配器输出 [B, D]（已投到与 cond_BD 同维度）
        融合：cond = cond_base + σ(g_text) * text_BD
        """
        if text_BD is None:
            return cond_BD_base
        # 校验/适配维度
        if text_BD.shape[-1] != cond_BD_base.shape[-1]:
            # 若你 TextAdapter 输出不是 D 维，建议在外部先投到 D 再传进来；
            # 这里做一次线性适配，避免出错。
            if not hasattr(self, "_text_proj"):
                self._text_proj = nn.Linear(text_BD.shape[-1], cond_BD_base.shape[-1], bias=False).to(self.device)
            text_BD = self._text_proj(text_BD)
        return cond_BD_base + torch.sigmoid(self.text_gate) * text_BD

    def forward(
        self,
        label_B: torch.Tensor,                              # [B] 仍然用 label 决定 base cond（你可传“虚拟类”ID）
        x_BLCv_wo_first_l: torch.Tensor,                   # VAE 特征（原 VAR 输入）
        kv_pairs: Optional[Dict[str, Tuple[torch.Tensor, ...]]] = None,  # 视觉条件 K/V 字典（每路 [B, G, D]）
        text_BD: Optional[torch.Tensor] = None,            # 文本向量 [B, D]（来自 TextCondAdapter）
        open_cross_n: Optional[int] = None,                # 可临时覆盖
    ) -> torch.Tensor:
        """
        返回：logits_BLV（与原 VAR 一致）
        说明：
          - 视觉条件：先用你的 CondEncoder 得到 {name: (K,V)}，再传给 kv_pairs。
          - 文本：先用 TextCondAdapter 得到 text_BD，再传入。
          - open_cross_n：仅前 K 层 block 之后执行一次 Cross-Attn。
        """
        if open_cross_n is None:
            open_cross_n = self.open_cross_n

        B = x_BLCv_wo_first_l.shape[0]
        device = x_BLCv_wo_first_l.device

        # ==== 1) 组装 SOS + 位置/层级嵌入（完全复用 VAR） ====
        if self.var.prog_si >= 0:
            bg, ed = self.var.begin_ends[self.var.prog_si]
        else:
            bg, ed = (0, self.var.L)

        with torch.cuda.amp.autocast(enabled=False):
            # class_emb → base cond（原始路径）
            cond_BD_base = self.var.class_emb(label_B)  # [B, D]

            # 这里融合文本（方案A）
            cond_BD_fused = self._fuse_text_cond(cond_BD_base, text_BD)

            # 起始 token（unconditional/virtual类 + pos_start）
            sos = cond_BD_fused.unsqueeze(1).expand(B, self.var.first_l, -1) + self.var.pos_start.expand(B, self.var.first_l, -1)

            if self.var.prog_si == 0:
                x_BLC = sos
            else:
                x_BLC = torch.cat((sos, self.var.word_embed(x_BLCv_wo_first_l.float())), dim=1)

            x_BLC += self.var.lvl_embed(self.var.lvl_1L[:, :ed].expand(B, -1)) + self.var.pos_1LC[:, :ed]

        attn_bias = self.var.attn_bias_for_masking[:, :, :ed, :ed]

        # cond → AdaLN 参数的共享线性（与原始一致）
        cond_ada_6D = self.var.shared_ada_lin(cond_BD_fused)

        # dtype 对齐（保持与原代码一致）
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        x_BLC = x_BLC.to(dtype=main_type)
        cond_ada_6D = cond_ada_6D.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)

        # ==== 2) 迭代每个 Block；在前 K 层后插入一次 Cross-Attn ====
        # 视觉条件 token 拼接
        kv_tokens = _concat_kv_tokens(kv_pairs) if kv_pairs is not None else None
        if kv_tokens is not None:
            kv_tokens = kv_tokens.to(dtype=main_type, device=x_BLC.device)

        for li, blk in enumerate(self.var.blocks):
            # 原始 VAR Block
            x_BLC = blk(x=x_BLC, cond_BD=cond_ada_6D, attn_bias=attn_bias)

            # 仅前 K 层插 Cross-Attn
            if li < open_cross_n and kv_tokens is not None:
                x_BLC = self.cross_layers[li](x_BLC, kv_tokens)

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
