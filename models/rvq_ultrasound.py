# --- rvq_ultrasound.py (cleaned) ---
"""
RVQ适配器：轻量包装VAR的VQVAE，专注于接口适配
直接继承现有组件，最小化修改
放置位置：VAR/models/rvq_ultrasound.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

# 智能导入：兼容所有运行方式
import sys
import os

# 添加必要的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from models.vqvae import VQVAE
    from models.quant import VectorQuantizer2
except ImportError:
    try:
        from vqvae import VQVAE
        from quant import VectorQuantizer2
    except ImportError as e:
        print(f"❌ Failed to import VAR modules: {e}")
        print("💡 Please ensure you have vqvae.py and quant.py in the models/ directory")
        raise


class RVQUltrasoundAdapter(nn.Module):
    """
    RVQ适配器：包装VAR的VQVAE以支持超声图像
    原则：能继承就继承，只做必要的通道适配
    """
    def __init__(self, vqvae: VQVAE, target_channels: int = 1):
        super().__init__()
        self.vqvae = vqvae
        self.quantizer = vqvae.quantize
        self.target_channels = target_channels
        if target_channels != 3:
            self._adapt_channels(target_channels)

    def _adapt_channels(self, target_channels: int):
        """适配输入输出通道数，确保设备和dtype一致性"""
        device = next(self.vqvae.parameters()).device

        # encoder conv_in
        old_conv_in = self.vqvae.encoder.conv_in
        if old_conv_in.in_channels != target_channels:
            new_conv_in = nn.Conv2d(
                target_channels, old_conv_in.out_channels,
                kernel_size=old_conv_in.kernel_size,
                stride=old_conv_in.stride,
                padding=old_conv_in.padding
            ).to(device=device, dtype=old_conv_in.weight.dtype)
            if old_conv_in.in_channels == 3 and target_channels == 1:
                with torch.no_grad():
                    new_conv_in.weight.data = old_conv_in.weight.data.mean(dim=1, keepdim=True)
                    if old_conv_in.bias is not None:
                        new_conv_in.bias.data = old_conv_in.bias.data.clone()
            self.vqvae.encoder.conv_in = new_conv_in

        # decoder conv_out
        old_conv_out = self.vqvae.decoder.conv_out
        if old_conv_out.out_channels != target_channels:
            new_conv_out = nn.Conv2d(
                old_conv_out.in_channels, target_channels,
                kernel_size=old_conv_out.kernel_size,
                stride=old_conv_out.stride,
                padding=old_conv_out.padding
            ).to(device=device, dtype=old_conv_out.weight.dtype)
            if old_conv_out.bias is not None and target_channels <= old_conv_out.out_channels:
                with torch.no_grad():
                    new_conv_out.bias.data = old_conv_out.bias.data[:target_channels].clone()
            self.vqvae.decoder.conv_out = new_conv_out

    def quantize_residual(
        self, img: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        多尺度token化，返回token索引和累计重建（防信息泄漏 + 潜空间前缀）
        Returns:
            z_list:                [z_1x1, z_2x2, ..., z_16x16]
            fhat_prefix_list:      图像域前缀 [B,1,H,W] in [0,1]
            fhat_lat_prefix_list:  潜空间前缀 [B,D,Gmax,Gmax]
            commit_loss, vq_loss:  占位（训练器里再算）
        */
        """
        B, C, H, W = img.shape

        # 1) 编码 + 量化
        z_list = self.vqvae.img_to_idxBl(img)

        # 断言与维度检查
        assert len(z_list) == len(self.quantizer.v_patch_nums), \
            f"Scale count mismatch: {len(z_list)} vs {len(self.quantizer.v_patch_nums)}"
        for i, z in enumerate(z_list):
            pn = self.quantizer.v_patch_nums[i]
            expected_len = pn * pn
            assert z.shape[1] == expected_len, f"Scale {i}: token length {z.shape[1]} != {expected_len} (grid {pn}×{pn})"
            assert z.dtype == torch.long, f"Scale {i}: expected long tensor, got {z.dtype}"

        # 2) 逐尺度累计（防泄漏：先记prefix，再累加本层）
        D = self.quantizer.embedding.embedding_dim
        Gmax = self.quantizer.v_patch_nums[-1]
        f_hat_cumulative = torch.zeros(B, D, Gmax, Gmax, device=img.device)

        fhat_prefix_list: List[torch.Tensor] = []
        fhat_lat_prefix_list: List[torch.Tensor] = []
        SN = len(self.quantizer.v_patch_nums)

        for i, z_tokens in enumerate(z_list):
            with torch.no_grad():
                fhat_lat_prefix_list.append(f_hat_cumulative.detach().clone())
                if hasattr(self.vqvae, 'fhat_to_img'):
                    img_prefix = self.vqvae.fhat_to_img(f_hat_cumulative.detach())
                else:
                    img_prefix = self.vqvae.decoder(self.vqvae.post_quant_conv(f_hat_cumulative.detach()))
                img_prefix = img_prefix.clamp(-1, 1)
                img_prefix_01 = (img_prefix + 1) * 0.5
            fhat_prefix_list.append(img_prefix_01)

            pn = self.quantizer.v_patch_nums[i]
            z_embedded = self.quantizer.embedding(z_tokens).transpose(1, 2).reshape(B, D, pn, pn)
            if pn != Gmax:
                z_upsampled = F.interpolate(z_embedded, size=(Gmax, Gmax), mode='bilinear', align_corners=False)
            else:
                z_upsampled = z_embedded
            z_processed = self.quantizer.quant_resi[i/(SN-1)](z_upsampled)
            f_hat_cumulative = f_hat_cumulative + z_processed

        assert len(fhat_prefix_list) == len(z_list), "Prefix count mismatch"
        assert len(fhat_lat_prefix_list) == len(z_list), "Latent prefix count mismatch"

        commit_loss = torch.tensor(0.0, device=img.device, requires_grad=False)
        vq_loss = torch.tensor(0.0, device=img.device, requires_grad=False)
        return z_list, fhat_prefix_list, fhat_lat_prefix_list, commit_loss, vq_loss

    def embed_and_upsample(self, z_s: torch.Tensor, si: int) -> torch.Tensor:
        """token嵌入并上采样"""
        B = z_s.shape[0]
        D = self.quantizer.embedding.embedding_dim
        pn = self.quantizer.v_patch_nums[si]
        Gmax = self.quantizer.v_patch_nums[-1]

        z_embedded = self.quantizer.embedding(z_s).transpose(1, 2).reshape(B, D, pn, pn)
        if pn != Gmax:
            z_upsampled = F.interpolate(z_embedded, size=(Gmax, Gmax), mode='bilinear', align_corners=False)
        else:
            z_upsampled = z_embedded
        SN = len(self.quantizer.v_patch_nums)
        up_feature = self.quantizer.quant_resi[si/(SN-1)](z_upsampled)
        return up_feature

    # 兼容旧接口（4返回值）
    def quantize_residual_simple(self, img: torch.Tensor):
        z_list, fhat_prefix_list, _, commit_loss, vq_loss = self.quantize_residual(img)
        return z_list, fhat_prefix_list, commit_loss, vq_loss

    # 仅获取潜空间前缀
    def get_latent_prefixes(self, img: torch.Tensor) -> List[torch.Tensor]:
        _, _, fhat_lat_prefix_list, _, _ = self.quantize_residual(img)
        return fhat_lat_prefix_list

    def decode(self, f_hat_latent: torch.Tensor) -> torch.Tensor:
        """潜空间 → 图像域"""
        if hasattr(self.vqvae, 'fhat_to_img'):
            return self.vqvae.fhat_to_img(f_hat_latent)
        else:
            return self.vqvae.decoder(self.vqvae.post_quant_conv(f_hat_latent)).clamp(-1, 1)

    def forward(self, img: torch.Tensor):
        return self.vqvae(img)


def create_rvq_adapter_from_pretrained(
    vae_ckpt_path: str = 'vae_ch160v4096z32.pth',
    device: str = 'cuda',
    target_channels: int = 1,
) -> RVQUltrasoundAdapter:
    """从预训练VAE创建RVQ适配器"""
    try:
        from models import build_vae_var
    except ImportError:
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from models import build_vae_var

    vae_local, _ = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        num_classes=1000, depth=16,
    )

    if os.path.exists(vae_ckpt_path):
        vae_local.load_state_dict(torch.load(vae_ckpt_path, map_location='cpu'), strict=True)
        print(f"✅ Loaded pretrained VAE from {vae_ckpt_path}")
    else:
        print(f"⚠️  VAE checkpoint not found: {vae_ckpt_path}")

    adapter = RVQUltrasoundAdapter(vae_local, target_channels=target_channels)
    return adapter


def create_rvq_adapter_minimal(
    vocab_size: int = 4096,
    patch_nums: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
    device: str = 'cuda',
    target_channels: int = 1,
) -> RVQUltrasoundAdapter:
    """创建最小配置的RVQ适配器（不依赖预训练权重）"""
    vqvae = VQVAE(
        vocab_size=vocab_size,
        z_channels=32,
        ch=160,
        v_patch_nums=patch_nums,
        test_mode=False,
    ).to(device)
    adapter = RVQUltrasoundAdapter(vqvae, target_channels=target_channels).to(device)
    return adapter


if __name__ == "__main__":
    print("🔄 RVQ Ultrasound Adapter - Optimized Test")
    print("=" * 50)

    try:
        print("✅ VAR modules imported successfully")
        print(f"   VQVAE: {VQVAE}")
        print(f"   VectorQuantizer2: {VectorQuantizer2}")
    except NameError as e:
        print(f"❌ Module import failed: {e}")
        exit(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📱 Device: {device}")

    try:
        print("\n🔧 Creating optimized RVQ adapter...")
        adapter = create_rvq_adapter_minimal(device=device)
        print("✅ Adapter created successfully")

        print("🔍 Architecture verification...")
        model_device = next(adapter.parameters()).device
        D = adapter.quantizer.embedding.embedding_dim
        Gmax = adapter.quantizer.v_patch_nums[-1]
        print(f"   Model device: {model_device}")
        print(f"   Embedding dim: {D}")
        print(f"   Max grid size: {Gmax}")
        assert str(model_device).startswith(device.split(':')[0]), "Device mismatch"
        print("✅ Architecture verified")

        print("🔄 Testing with random data...")
        B = 2
        img = torch.randn(B, 1, 256, 256, device=device)
        print(f"   Input shape: {img.shape}, device: {img.device}")

        z_list, fhat_prefix_list, fhat_lat_prefix_list, commit_loss, vq_loss = adapter.quantize_residual(img)
        print(f"✅ quantize_residual (ENGINEERING enhanced):")
        print(f"   z_list: {len(z_list)} scales")
        print(f"   fhat_prefix: {len(fhat_prefix_list)} images [0,1] range")
        print(f"   fhat_lat_prefix: {len(fhat_lat_prefix_list)} latents [B,D,Gmax,Gmax]")
        print(f"   losses: commit={commit_loss.item():.6f}, vq={vq_loss.item():.6f}")
        print(f"   prefix images range: [{fhat_prefix_list[0].min().item():.3f}, {fhat_prefix_list[0].max().item():.3f}]")

        print(f"🔍 Enhanced verification:")
        print(f"   prefix[0] (before scale 0): img_mean={fhat_prefix_list[0].mean().item():.6f}, lat_norm={fhat_lat_prefix_list[0].norm().item():.6f}")
        print(f"   prefix[1] (before scale 1): img_mean={fhat_prefix_list[1].mean().item():.6f}, lat_norm={fhat_lat_prefix_list[1].norm().item():.6f}")
        print(f"   prefix[9] (before scale 9): img_mean={fhat_prefix_list[9].mean().item():.6f}, lat_norm={fhat_lat_prefix_list[9].norm().item():.6f}")

        assert all((p.min().item() >= 0.0) and (p.max().item() <= 1.0) for p in fhat_prefix_list), "Image prefix not in [0,1] range"
        assert all(lat.shape == (B, D, Gmax, Gmax) for lat in fhat_lat_prefix_list), "Latent prefix shape mismatch"
        print(f"   ✅ Image prefixes in [0,1]: OK")
        print(f"   ✅ Latent prefixes shape [{B},{D},{Gmax},{Gmax}]: OK")
        print(f"   prefix[0] mean abs: {fhat_prefix_list[0].abs().mean().item():.6f}")

        print(f"🔍 Token scale analysis:")
        for si, z_s in enumerate(z_list):
            pn = adapter.quantizer.v_patch_nums[si]
            print(f"   Scale {si}: token shape {z_s.shape} (grid {pn}×{pn})")

        f_hat = torch.zeros(B, D, Gmax, Gmax, device=device)
        print(f"\n🔄 Progressive reconstruction (all upsample to {Gmax}×{Gmax}):")
        for si, z_s in enumerate(z_list):
            up_feature = adapter.embed_and_upsample(z_s, si)
            f_hat = f_hat + up_feature
            pn = adapter.quantizer.v_patch_nums[si]
            print(f"   Scale {si}: {z_s.shape} → embed({pn}×{pn}) → upsample → {up_feature.shape}")

        final_img = adapter.decode(f_hat)
        print(f"✅ embed_and_upsample + decode (optimized):")
        print(f"   cumulative f_hat: {f_hat.shape}")
        print(f"   final_img: {final_img.shape}, device: {final_img.device}")
        print(f"   final image range: [{final_img.min().item():.3f}, {final_img.max().item():.3f}]")

        assert len(z_list) == 10, f"Expected 10 scales, got {len(z_list)}"
        assert len(fhat_prefix_list) == 10, f"Expected 10 prefix images, got {len(fhat_prefix_list)}"
        assert final_img.shape == (B, 1, 256, 256), f"Wrong output shape: {final_img.shape}"
        assert final_img.device == img.device, f"Device mismatch: input {img.device}, output {final_img.device}"
        assert torch.isfinite(final_img).all(), "Output contains NaN/Inf"
        assert f_hat.shape == (B, D, Gmax, Gmax), f"Wrong f_hat shape: {f_hat.shape}"

        mse = F.mse_loss(final_img, img).item()
        print(f"✅ Reconstruction MSE: {mse:.6f}")

        print("\n🎉 ENGINEERING enhanced test passed! All improvements working:")
        print("🔑 ✅ Information leakage prevented")
        print("🔑 ✅ Latent prefixes provided")
        print("🔑 ✅ Prefix images normalized to [0,1]")
        print("🔑 ✅ Assertions & dimension checks OK")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
# --- end file ---
