# -*- coding: utf-8 -*-
import torch
from models.cond_encoders import CondEncPack

def _fake_prefix(B=2, H=256, W=256, latent_ch=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # 图像前缀：[0,1]
    coarse_img = torch.rand(B, 1, H, W, device=device)
    # 潜空间前缀
    coarse_lat = torch.randn(B, latent_ch, H, W, device=device) * 0.01
    return coarse_img, coarse_lat

def _fake_inputs(B=2, H=256, W=256, device='cuda' if torch.cuda.is_available() else 'cpu'):
    edge   = torch.rand(B, 1, H, W, device=device)
    sketch = torch.rand(B, 1, H, W, device=device)
    fan    = torch.rand(B, 1, H, W, device=device)
    # 整数 mask（0..3）
    mask   = torch.randint(0, 4, (B, 1, H, W), device=device)
    return dict(edge=edge, sketch=sketch, fan=fan, mask=mask)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    D = 768
    latent_ch = 32
    pack = CondEncPack(d_model=D, num_mask_classes=4, latent_ch=latent_ch).to(device)

    coarse_img, coarse_lat = _fake_prefix(device=device, latent_ch=latent_ch)
    conds = _fake_inputs(device=device)
    conds['coarse_img'] = coarse_img
    conds['coarse_lat'] = coarse_lat

    for g in [1, 2, 4, 8, 16]:
        kv = pack(conds, grid=g)
        G = g * g
        # 必须至少包含 coarse 分支
        assert 'coarse' in kv, "missing 'coarse' kv"
        for name, (K, V) in kv.items():
            assert K.shape == (coarse_img.size(0), G, D), f"{name} K shape wrong: {K.shape}"
            assert V.shape == (coarse_img.size(0), G, D), f"{name} V shape wrong: {V.shape}"
            assert torch.isfinite(K).all() and torch.isfinite(V).all(), f"{name} contains NaN/Inf"
        print(f"[grid={g}] OK with keys: {list(kv.keys())}")

    # Dropout 冒烟（不报错即可）
    kv2 = pack(conds, grid=8, dropout={'edge': 1.0, 'sketch': 1.0, 'mask': 1.0, 'fan': 1.0})
    assert list(kv2.keys()) == ['coarse'], "dropout should keep only 'coarse'"
    print("Dropout path OK")

    print("✅ CondEncoders smoke test passed.")

if __name__ == "__main__":
    main()
