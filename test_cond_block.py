# -*- coding: utf-8 -*-
import torch
from models.cond_block import CondTransformerBlock
from models.cond_encoders import CondEncPack

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, L, H, W = 2, 64, 256, 256
    D = 768
    latent_ch = 32
    grids = [1, 2, 4, 8, 16]

    # 伪造条件（含 coarse_img / coarse_lat）
    conds = {
        'edge':   torch.rand(B,1,H,W, device=device),
        'sketch': torch.rand(B,1,H,W, device=device),
        'fan':    torch.rand(B,1,H,W, device=device),
        'mask':   torch.randint(0, 4, (B,1,H,W), device=device),
        'coarse_img': torch.rand(B,1,H,W, device=device),
        'coarse_lat': torch.randn(B,latent_ch,H,W, device=device)*0.01,
    }

    pack = CondEncPack(d_model=D, num_mask_classes=4, latent_ch=latent_ch).to(device)

    # Block（设为 eval，便于可重复性）
    block = CondTransformerBlock(
        d_model=D, n_heads=12, mlp_ratio=4.0,
        dropout=0.0, num_stages=len(grids),
        open_cross_layers=3, gate_init=-2.0, cond_dim=D
    ).to(device).eval()

    # token 序列 + 条件向量
    x = torch.randn(B, L, D, device=device)
    cond_vec = torch.randn(B, D, device=device)

    # 取某个 stage 的 kv（例如 grid=8）
    si = grids.index(8)
    kv = pack(conds, grid=grids[si])  # {'edge':(K,V), ..., 'coarse':(K,V)}

    # 1) 开启 Cross（layer_offset_in_stage=0，前3层默认开启）
    y_cross = block(
        x, cond_vec, attn_bias=None, kv_pairs=kv,
        stage_id=si, layer_offset_in_stage=0
    )
    assert y_cross.shape == x.shape and torch.isfinite(y_cross).all()
    print(f"with cross: y.mean={y_cross.mean().item():.6f}")

    # 2) 关闭 Cross（超过 open_cross_layers；或 kv=None）
    y_nocross = block(
        x, cond_vec, attn_bias=None, kv_pairs=kv,
        stage_id=si, layer_offset_in_stage=99
    )
    assert y_nocross.shape == x.shape and torch.isfinite(y_nocross).all()
    diff = (y_cross - y_nocross).abs().mean().item()
    print(f"diff(open vs closed)={diff:.6f}")

    # 3) kv 为空（等价于关）
    y_empty = block(
        x, cond_vec, attn_bias=None, kv_pairs={},
        stage_id=si, layer_offset_in_stage=0
    )
    diff2 = (y_empty - y_nocross).abs().mean().item()
    print(f"diff(nocross vs empty)={diff2:.6f}")

    # 4) 查看门控值
    with torch.no_grad():
        gate_val = torch.sigmoid(block.stage_gates[si]).item()
    print(f"stage {si} gate(sigmoid)={gate_val:.4f}  (应接近 0.1~0.2)")

    print("✅ CondTransformerBlock smoke test passed.")

if __name__ == "__main__":
    main()
