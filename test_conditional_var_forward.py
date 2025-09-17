import os
import torch
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.manual_seed(0)

from models.rvq_ultrasound import create_rvq_adapter_minimal
from models.cond_encoders import CondEncPack
from models.conditional_var_block import wrap_var_with_condition
from models import build_vae_var


def _make_simple_visual_conds(img_01):
    B, _, H, W = img_01.shape
    img = img_01
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=img.dtype, device=img.device).view(1,1,3,3)
    ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=img.dtype, device=img.device).view(1,1,3,3)
    gx = torch.conv2d(img, kx, padding=1); gy = torch.conv2d(img, ky, padding=1)
    edge = (gx.abs() + gy.abs()); edge = (edge - edge.min())/(edge.max()-edge.min()+1e-6)
    mask = (img > img.mean(dim=[2,3], keepdim=True)).float()
    k = torch.ones(1,1,5,5, device=img.device, dtype=img.dtype) / 25.0
    sketch = torch.conv2d(img, k, padding=2)
    yy, xx = torch.meshgrid(torch.linspace(-1,1,H,device=img.device),
                            torch.linspace(-1,1,W,device=img.device), indexing='ij')
    rr = torch.sqrt(xx**2 + yy**2); th = torch.atan2(yy, xx)
    fan = ((rr<1.0) & (th>-0.4) & (th<1.2)).float()[None,None].expand(B,1,H,W)
    return {"edge": edge.clamp(0,1), "mask": mask, "sketch": sketch.clamp(0,1), "fan": fan}


def _sanitize_kv_dict(kv: dict, d_model: int) -> dict:
    """数值卫生：LayerNorm + 替换 NaN/Inf，避免 Cross-Attn 爆数。"""
    ln = torch.nn.LayerNorm(d_model).to(DEVICE)
    out = {}
    for name, pair in kv.items():
        if isinstance(pair, (tuple, list)) and len(pair) >= 2:
            K, V = pair[0], pair[1]
            K = torch.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)
            V = torch.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)
            # [B,G,D] 上最后一维 LN
            K = ln(K)
            V = ln(V)
            out[name] = (K, V)
    return out


def main():
    print(f"[Device] {DEVICE}")

    # 1) RVQ
    rvq = create_rvq_adapter_minimal(device=DEVICE)
    rvq.eval()
    grids = list(rvq.quantizer.v_patch_nums)
    print("[RVQ] stages:", grids)

    B, H, W = 2, 256, 256
    # 输入在 [-1,1]，更贴近 VAE 训练分布
    img = torch.randn(B,1,H,W, device=DEVICE).clamp_(-1,1)
    img_01 = (img + 1) * 0.5

    with torch.no_grad():
        z_list, fhat_img_prefix, fhat_lat_prefix, _, _ = rvq.quantize_residual(img)
        gt_BL = torch.cat(z_list, dim=1)                     # [B, L]
        x_BLCv_wo_first_l = rvq.vqvae.quantize.idxBl_to_var_input(z_list)

    # 目标检查：应在 [0, V-1]
    V = rvq.quantizer.embedding.num_embeddings
    assert gt_BL.dtype == torch.long
    assert int(gt_BL.min()) >= 0 and int(gt_BL.max()) < V, f"targets out of range: [{int(gt_BL.min())}, {int(gt_BL.max())}] vs V={V}"

    # 2) 原始 VAR & 包装
    vae_tmp, base_var = build_vae_var(
        V=V,
        Cvae=rvq.quantizer.embedding.embedding_dim,
        ch=160, share_quant_resi=4,
        device=DEVICE, patch_nums=tuple(grids),
        num_classes=1000, depth=16
    )
    var_ckpt = os.environ.get("VAR_CKPT", "")
    if var_ckpt and os.path.exists(var_ckpt):
        base_var.load_state_dict(torch.load(var_ckpt, map_location="cpu"), strict=False)
        print(f"[VAR] loaded ckpt: {var_ckpt}")

    D_MODEL = getattr(base_var, "C", None) or getattr(base_var, "embed_dim", None) or 768
    cond_var = wrap_var_with_condition(base_var, open_cross_n=3, fuse_text_init=-6.0).to(DEVICE).eval()
    # ^ 初始更强抑制 Cross 门控（-6.0），避免一上来干扰过大

    # 3) 条件：构造每个 stage 的 KV；本测试先用最细一层，并做数值卫生
    cond_pack = CondEncPack(d_model=D_MODEL).to(DEVICE).eval()
    static_conds = _make_simple_visual_conds(img_01)

    kv_per_stage = []
    for si, g in enumerate(grids):
        conds = {
            "coarse_img": fhat_img_prefix[si].to(DEVICE),   # 0~1
            # "coarse_lat": fhat_lat_prefix[si].to(DEVICE), # 如需 latent coarse 可打开
            "edge":   static_conds["edge"].to(DEVICE),
            "mask":   static_conds["mask"].to(DEVICE),
            "sketch": static_conds["sketch"].to(DEVICE),
            "fan":    static_conds["fan"].to(DEVICE),
        }
        kv = cond_pack(conds, grid=g)
        kv = _sanitize_kv_dict(kv, D_MODEL)  # ★ 关键：稳定化
        kv_per_stage.append(kv)

    kv_pairs = kv_per_stage[-1]  # 先用最后一层（16x16）的 KV 字典

    label_B = torch.zeros(B, dtype=torch.long, device=DEVICE)

    # 4) 先跑“无 Cross-Attn”基线，验证不是主干模型的问题
    with torch.no_grad():
        logits_no_x = cond_var(
            label_B=label_B,
            x_BLCv_wo_first_l=x_BLCv_wo_first_l,
            kv_pairs=None,          # ← 不给条件，Cross 关闭
            text_BD=None,
            open_cross_n=0          # ← 保底完全关闭
        )
    assert torch.isfinite(logits_no_x).all(), "baseline logits contain NaN/Inf"
    ce_baseline = F.cross_entropy(logits_no_x.reshape(-1, V), gt_BL.reshape(-1)).item()
    print(f"[Baseline] CE(no cross) = {ce_baseline:.4f}  (参考 ~ ln(V) ≈ {torch.log(torch.tensor(float(V))).item():.4f})")

    # 5) 再打开 Cross-Attn（已做 KV 卫生 & 门控偏置更负）
    with torch.no_grad():
        logits_BLV = cond_var(
            label_B=label_B,
            x_BLCv_wo_first_l=x_BLCv_wo_first_l,
            kv_pairs=kv_pairs,
            text_BD=None,
            open_cross_n=3
        )

    # 数值检查
    assert torch.isfinite(logits_BLV).all(), "logits with cross contain NaN/Inf"
    ce = F.cross_entropy(logits_BLV.reshape(-1, V), gt_BL.reshape(-1)).item()
    print("logits_BLV:", tuple(logits_BLV.shape))
    print("gt_BL     :", tuple(gt_BL.shape))
    print(f"[Cross-Attn ON] CE = {ce:.4f}")

    print("✅ conditional VAR forward (stable) passed.")


if __name__ == "__main__":
    main()
