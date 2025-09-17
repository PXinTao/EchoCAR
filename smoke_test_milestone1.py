#!/usr/bin/env python3
"""
里程碑1简化烟雾测试 - 基于继承的RVQ适配器
直接在VAR项目内运行
"""

import torch
import torch.nn.functional as F
import os
import sys

def test_milestone1_minimal():
    """简化的里程碑1测试"""
    print("=" * 60)
    print("🚀 里程碑1：RVQ适配器 - 简化测试")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📱 Device: {device}")
    
    # 测试参数
    B = 4  # 小batch避免内存问题
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    
    try:
        # 方式1：尝试从预训练创建（如果有VAE权重）
        print("\n🔧 Attempting to create from pretrained...")
        try:
            from models.rvq_ultrasound import create_rvq_adapter_from_pretrained
            adapter = create_rvq_adapter_from_pretrained(device=device)
            print("✅ Created from pretrained VAE")
        except Exception as e:
            print(f"⚠️  Pretrained creation failed: {e}")
            print("🔧 Falling back to minimal creation...")
            from models.rvq_ultrasound import create_rvq_adapter_minimal
            adapter = create_rvq_adapter_minimal(device=device)
            print("✅ Created minimal adapter")
        
        # 基本信息
        total_params = sum(p.numel() for p in adapter.parameters())
        print(f"📊 Total parameters: {total_params/1e6:.1f}M")
        
    except Exception as e:
        print(f"❌ Failed to create adapter: {e}")
        print("💡 Make sure models/rvq_ultrasound.py exists")
        return False
    
    try:
        # 核心功能测试
        print(f"\n🔧 Testing core functions with batch_size={B}...")
        
        # 创建测试数据：灰度超声图像
        img = torch.randn(B, 1, 256, 256).to(device)
        print(f"   Input: {img.shape}")
        
        # 测试1: quantize_residual
        z_list, fhat_prefix_list, commit_loss, vq_loss = adapter.quantize_residual(img)
        
        print(f"✅ quantize_residual:")
        print(f"   z_list: {len(z_list)} scales (expected: {len(patch_nums)})")
        print(f"   fhat_prefix: {len(fhat_prefix_list)} images")
        print(f"   losses: commit={commit_loss.item():.4f}, vq={vq_loss.item():.4f}")
        
        # 快速验证形状
        assert len(z_list) == len(patch_nums), f"z_list length mismatch"
        assert len(fhat_prefix_list) == len(patch_nums), f"fhat_prefix length mismatch"
        
        for i, (z_s, expected_pn) in enumerate(zip(z_list, patch_nums)):
            expected_len = expected_pn * expected_pn
            assert z_s.shape == (B, expected_len), f"Scale {i} shape mismatch: {z_s.shape} vs {(B, expected_len)}"
        
        for i, fhat in enumerate(fhat_prefix_list):
            assert fhat.shape == (B, 1, 256, 256), f"Prefix {i} shape mismatch: {fhat.shape}"
        
        # 测试2: embed_and_upsample + decode
        f_hat = torch.zeros(B, adapter.quantizer.Cvae, patch_nums[-1], patch_nums[-1]).to(device)
        
        for si, z_s in enumerate(z_list):
            up_feature = adapter.embed_and_upsample(z_s, si)
            f_hat = f_hat + up_feature
        
        final_img = adapter.decode(f_hat)
        
        print(f"✅ embed_and_upsample + decode:")
        print(f"   cumulative f_hat: {f_hat.shape}")
        print(f"   final_img: {final_img.shape}")
        
        assert final_img.shape == (B, 1, 256, 256), f"Final image shape mismatch: {final_img.shape}"
        
        # 测试3: 重建质量基本检查
        mse = F.mse_loss(final_img, img).item()
        print(f"✅ Reconstruction MSE: {mse:.6f}")
        
        # 确保数值稳定
        assert torch.isfinite(final_img).all(), "Final image contains non-finite values"
        
    except Exception as e:
        print(f"❌ Core function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 完成度检查
    print(f"\n✅ DoD (Definition of Done) 检查:")
    print(f"   ✓ len(z_list) == len(scales): {len(z_list)} == {len(patch_nums)}")
    print(f"   ✓ z_s.shape == [B, g*g] for each scale")
    print(f"   ✓ len(fhat_prefix_list) == len(scales)")
    print(f"   ✓ fhat_prefix shape == [B,1,H,W] for each")
    print(f"   ✓ decode(f_hat) outputs [B,1,256,256]")
    
    print("\n" + "=" * 60)
    print("🎉 里程碑1 - 基于继承的RVQ适配器测试通过！")
    print("=" * 60)
    print("\n🔑 关键接口确认:")
    print("   quantize_residual(img) -> (z_list, fhat_prefix_list, commit_loss, vq_loss)")
    print("   embed_and_upsample(z_s, si) -> up_feature")
    print("   decode(f_hat_latent) -> img")
    print("\n🚀 准备好进入里程碑2：条件编码器！")
    
    return True


def test_with_var_dependency():
    """如果有完整VAR环境的测试"""
    print("🔍 Checking VAR dependencies...")
    
    try:
        from models.vqvae import VQVAE
        from models.quant import VectorQuantizer2
        print("✅ VAR core modules found")
        return True
    except ImportError as e:
        print(f"❌ VAR modules not found: {e}")
        print("💡 Please ensure you're running from VAR project root")
        return False


if __name__ == "__main__":
    # 检查依赖
    if not test_with_var_dependency():
        print("\n⚠️  Please run this script from VAR project root directory")
        sys.exit(1)
    
    # 运行测试
    success = test_milestone1_minimal()
    sys.exit(0 if success else 1)