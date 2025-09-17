# ultra_var/models/condition/text_adapter.py
# -*- coding: utf-8 -*-
"""
TextCondAdapter
---------------
将 BiomedCLIP / open-clip 文本编码器适配为 VAR 先验的条件向量（AdaLN 输入）。
- 默认使用 BiomedCLIP（open-clip），支持线上(hf-hub)和离线(本地目录)两种加载方式
- 冻结大模型参数，仅训练我们的小投影头
- 输出维度对齐到 d_model（即先验的 embed_dim）
- 训练期支持 cond-dropout，方便后续做 CFG
- 提供简易 smoke test（__main__）

依赖:
    pip install open_clip_torch huggingface_hub
"""

from typing import List, Optional, Union, Tuple
import os
import torch
import torch.nn as nn

try:
    import open_clip  # pip package name: open_clip_torch
except Exception as e:
    open_clip = None
    _IMPORT_ERR = e


class _Gate(nn.Module):
    """可学习门控：y = sigmoid(g) * x"""
    def __init__(self, init: float = -2.0):
        super().__init__()
        self.g = nn.Parameter(torch.tensor(float(init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.g) * x

    @property
    def value(self) -> float:
        with torch.no_grad():
            return torch.sigmoid(self.g).item()


class TextCondAdapter(nn.Module):
    """
    文本条件适配器（AdaLN 全局向量）

    Args:
        d_model:      输出到先验的维度（与 VAR 的 embed_dim 对齐）
        model_name:   open-clip 的 model 名称或 hf-hub 路径/本地目录
                      例：'hf-hub:microsoft/BiomedCLIP-PubMedBERT_vision-vit_base_patch16_224'
                          或 '/abs/path/checkpoints/biomedclip'
        pretrained:   open-clip 的 'pretrained' 标志（多数情况下留空即可）
        normalize:    是否对 CLIP 文本特征做 L2 归一化
        proj_hidden:  投影 MLP 隐层维度（0 表示单层 Linear）
        dropout_p:    训练时 cond-dropout 概率（整条文本条件置零）
        gate_init:    输出门控初值（负值=弱注入）
        freeze_clip:  是否冻结底座（True 强烈推荐）
        dtype:        模型主 dtype（'float32' 或 'float16'）
    """
    def __init__(
        self,
        d_model: int,
        model_name: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_vision-vit_base_patch16_224",
        pretrained: Optional[str] = None,
        normalize: bool = True,
        proj_hidden: int = 0,
        dropout_p: float = 0.2,
        gate_init: float = -2.0,
        freeze_clip: bool = True,
        dtype: str = "float32",
    ):
        super().__init__()
        if open_clip is None:
            raise ImportError(
                "open_clip 未安装，先运行：pip install open_clip_torch huggingface_hub\n"
                f"原始错误：{_IMPORT_ERR}"
            )

        self.d_model       = int(d_model)
        self.model_name    = model_name
        self.pretrained    = pretrained
        self.normalize     = bool(normalize)
        self.proj_hidden   = int(proj_hidden)
        self.dropout_p     = float(dropout_p)
        self.freeze_clip   = bool(freeze_clip)
        self._gate         = _Gate(gate_init)

        # 构建 CLIP 文本模型
        self.clip, self.preprocess, self._tokenizer = self._create_clip(model_name, pretrained)

        # 推断文本特征维度（不前向真实文本也可从 text_projection 读到）
        d_clip = self._infer_text_width(self.clip)

        # 小投影头：d_clip -> d_model
        if self.proj_hidden and self.proj_hidden > 0:
            self.proj = nn.Sequential(
                nn.Linear(d_clip, self.proj_hidden, bias=True),
                nn.GELU(),
                nn.LayerNorm(self.proj_hidden),
                nn.Linear(self.proj_hidden, self.d_model, bias=True),
                nn.LayerNorm(self.d_model),
            )
        else:
            self.proj = nn.Sequential(
                nn.Linear(d_clip, self.d_model, bias=True),
                nn.LayerNorm(self.d_model),
            )

        # 冻结底座
        if self.freeze_clip:
            for p in self.clip.parameters():
                p.requires_grad = False

        # 统一 dtype
        if str(dtype).lower() in ("fp16", "float16", "half"):
            self.to(dtype=torch.float16)
        else:
            self.to(dtype=torch.float32)

    # ---------- 构建与推断辅助 ----------
    def _create_clip(self, model_name: str, pretrained: Optional[str]):
        """
        统一处理在线(hf-hub)/离线(本地目录)两种创建方式。
        返回: (model, preprocess, tokenizer_fn)
        """
        # open-clip 的创建接口很灵活：
        # - 如果 model_name 传入 hf-hub 名称或本地目录，也能正确解析
        # - pretrained 通常可以为 None
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )

        # 部分版本的 get_tokenizer 需要传标准 model 名，不认识 hf-hub 路径。
        # 我们做多重兜底，保证总能拿到一个 tokenizer 函数。
        tokenizer = None
        try:
            tokenizer = open_clip.get_tokenizer(model_name)
        except Exception:
            # 常见兜底：ViT-B-16 的 tokenizer
            try:
                tokenizer = open_clip.get_tokenizer("ViT-B-16")
            except Exception:
                # 兜底再兜底：直接用 open_clip.tokenize
                def _tok(x: List[str]):
                    return open_clip.tokenize(x)
                tokenizer = _tok

        return model, preprocess, tokenizer

    @staticmethod
    def _infer_text_width(clip_model) -> int:
        """
        估算 open-clip 文本输出维度。优先读 text_projection，其次试跑一次。
        """
        d_clip = None
        # 绝大多数 open-clip 模型有 text_projection 参数
        tp = getattr(clip_model, "text_projection", None)
        if tp is not None and isinstance(tp, torch.nn.Parameter):
            d_clip = tp.shape[1]

        if d_clip is None:
            # 作为兜底，跑一条假输入
            try:
                toks = open_clip.tokenize(["placeholder"])
                with torch.no_grad():
                    out = clip_model.encode_text(toks)
                d_clip = int(out.shape[-1])
            except Exception:
                # 最后的兜底：猜一个常见宽度（BiomedCLIP 是 512 或 768 常见）
                d_clip = 512
        return int(d_clip)

    # ---------- 前向 ----------
    def forward(
        self,
        texts: Optional[List[str]] = None,
        tokens: Optional[torch.Tensor] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """
        文本 → CLIP → 投影 →（可选归一）→ 门控 →（训练期 cond-dropout）→ [B, d_model]

        Args:
            texts:  List[str]，如果给出，将自动 tokenizer
            tokens: 预先 tokenize 好的 LongTensor（shape [B, T]）
            device: 将 tokens 放到何处；默认用模块所在 device

        Returns:
            cond_BD: [B, d_model]，用于 AdaLN（或直接作为 cond_BD 喂给 VAR）
        """
        if (texts is None) and (tokens is None):
            raise ValueError("必须提供 texts 或 tokens 之一。")

        dev = device or next(self.parameters()).device

        if tokens is None:
            # tokenizer 会自动添加特殊符号/截断
            tokens = self._tokenizer(texts)
        tokens = tokens.to(dev)

        with torch.no_grad():
            # open-clip 的 encode_text 会输出全局文本特征（已过投影）
            clip_feat = self.clip.encode_text(tokens)  # [B, d_clip]
            # 有些实现会自动 L2 normalize，这里再显式处理一次可控
            if self.normalize:
                clip_feat = torch.nn.functional.normalize(clip_feat, dim=-1, eps=1e-6)

        # 学习到 d_model
        cond = self.proj(clip_feat)  # [B, d_model]

        # 门控
        cond = self._gate(cond)

        # 训练期 cond-dropout（整条文本条件置零）
        if self.training and self.dropout_p > 0:
            if torch.rand(1, device=cond.device).item() < self.dropout_p:
                cond = torch.zeros_like(cond)

        return cond  # [B, d_model]

    # ---------- 实用函数 ----------
    @torch.no_grad()
    def tokenize(self, texts: List[str]) -> torch.Tensor:
        """显式暴露一个 tokenizer，便于外部先行处理。"""
        return self._tokenizer(texts)

    @property
    def gate_value(self) -> float:
        """返回当前门控 σ(g) 的标量值，便于日志监控。"""
        return self._gate.value

    def freeze_backbone(self, freeze: bool = True):
        """动态冻结/解冻 open-clip 主干。"""
        for p in self.clip.parameters():
            p.requires_grad = not freeze


# ========== 便捷构造器 ==========
def build_text_adapter_from_cfg(cfg: dict, d_model: int) -> TextCondAdapter:
    """
    从配置 dict 构建 TextCondAdapter，示例 cfg 结构：
    cfg = {
        "model_name": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_vision-vit_base_patch16_224",
        "pretrained": null,
        "normalize": true,
        "proj_hidden": 0,
        "dropout_p": 0.2,
        "gate_init": -2.0,
        "freeze_clip": true,
        "dtype": "float32"
    }
    """
    return TextCondAdapter(
        d_model=d_model,
        model_name=cfg.get("model_name", "hf-hub:microsoft/BiomedCLIP-PubMedBERT_vision-vit_base_patch16_224"),
        pretrained=cfg.get("pretrained", None),
        normalize=cfg.get("normalize", True),
        proj_hidden=cfg.get("proj_hidden", 0),
        dropout_p=cfg.get("dropout_p", 0.2),
        gate_init=cfg.get("gate_init", -2.0),
        freeze_clip=cfg.get("freeze_clip", True),
        dtype=cfg.get("dtype", "float32"),
    )


# ========== smoke test ==========
if __name__ == "__main__":
    # 仅作加载与前向形状检查
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[TextCondAdapter] device={device}")

    adapter = TextCondAdapter(
        d_model=768,
        model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_vision-vit_base_patch16_224",
        pretrained=None,
        normalize=True,
        proj_hidden=0,
        dropout_p=0.2,
        gate_init=-2.0,
        freeze_clip=True,
        dtype="float32",
    ).to(device)

    prompts = [
        "apical 4-chamber view, end-diastole",
        "parasternal long-axis view, end-systole",
    ]
    with torch.no_grad():
        out = adapter(prompts, device=device)  # [B, d_model]
    print("out.shape =", tuple(out.shape), " gate=", f"{adapter.gate_value:.3f}")
    assert out.shape == (len(prompts), 768)
    print("✅ TextCondAdapter smoke test passed.")
