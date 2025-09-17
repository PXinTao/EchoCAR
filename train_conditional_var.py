import os, glob, math, time, argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# === 你的工程内模块 ===
from models.rvq_ultrasound import create_rvq_adapter_minimal
from models.cond_encoders import CondEncPack
from models.conditional_var_block import wrap_var_with_condition
from models.text_adapter import TextCondAdapter

# 原仓库提供的构建器（VAR 主体）
try:
    from models import build_vae_var
except Exception as e:
    raise RuntimeError("未找到 build_vae_var（请确认 models/__init__.py 暴露了该函数）") from e

# CAMUS医学数据集
from camus_dataset import CAMUSMedicalDataset  # 假设保存为 camus_dataset.py


# ---------------------------
# Warmup + Cosine LR调度器
# ---------------------------
class WarmupCosine:
    def __init__(self, optimizer, warmup, total_steps, min_lr=1e-6):
        self.opt = optimizer
        self.warmup = warmup
        self.total = total_steps
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]
        self.min_lr = min_lr
        self.step_num = 0

    def step(self):
        self.step_num += 1
        t = self.step_num
        for i, g in enumerate(self.opt.param_groups):
            base = self.base_lrs[i]
            if t <= self.warmup:
                lr = base * t / max(1, self.warmup)
            else:
                progress = (t - self.warmup) / max(1, self.total - self.warmup)
                lr = self.min_lr + 0.5*(base - self.min_lr)*(1 + math.cos(math.pi*progress))
            g['lr'] = lr


# ---------------------------
# 训练主流程
# ---------------------------
def train(args):
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    os.makedirs(args.out, exist_ok=True)

    # 1) CAMUS医学数据集
    train_ds = CAMUSMedicalDataset(
        data_root=args.data, 
        size=args.size, 
        split="train",
        use_text=args.use_text
    )
    val_ds = CAMUSMedicalDataset(
        data_root=args.data, 
        size=args.size, 
        split="val",
        use_text=args.use_text
    )
    
    train_dl = DataLoader(
        train_ds, batch_size=args.bs, shuffle=True, 
        num_workers=args.workers, pin_memory=True, drop_last=True
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.bs, shuffle=False, 
        num_workers=args.workers//2, pin_memory=True, drop_last=False
    )
    
    steps_per_epoch = len(train_dl)
    total_steps = steps_per_epoch * args.epochs
    print(f"[Data] Train: {len(train_ds)}, Val: {len(val_ds)}")
    print(f"[Steps] {steps_per_epoch}/epoch × {args.epochs} epochs = {total_steps} total")

    # 2) RVQ 适配器
    adapter = create_rvq_adapter_minimal(
        vocab_size=args.vocab_size,
        patch_nums=(1,2,3,4,5,6,8,10,13,16),
        device=device,
        target_channels=1
    )
    adapter.eval()

    # 3) 构建 VAR 并获取维度
    vae_tmp, base_var = build_vae_var(
        V=args.vocab_size, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=tuple(adapter.quantizer.v_patch_nums),
        num_classes=1000, depth=args.depth
    )
    
    if args.var_ckpt and os.path.exists(args.var_ckpt):
        base_var.load_state_dict(torch.load(args.var_ckpt, map_location='cpu'), strict=False)
        print(f"[VAR] loaded checkpoint: {args.var_ckpt}")

    var_d_model = getattr(base_var, 'C', getattr(base_var, 'embed_dim', 1024))
    print(f"[Model] VAR d_model={var_d_model}, VQ Cvae=32")

    # 4) 条件编码器
    cond_pack = CondEncPack(d_model=var_d_model).to(device).train()

    # 5) 文本适配器（如果启用文本）
    text_adapter = None
    if args.use_text:
        text_adapter = TextCondAdapter(
            d_model=var_d_model,
            model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_vision-vit_base_patch16_224",
            normalize=True,
            proj_hidden=0,
            dropout_p=args.text_dropout,
            gate_init=-2.0,
            freeze_clip=True,
            dtype="float32"
        ).to(device).train()
        print(f"[Text] BiomedCLIP adapter initialized")

    # 6) 包装条件化VAR
    cond_var = wrap_var_with_condition(
        base_var,
        open_cross_n=args.open_cross_n,
        fuse_text_init=-6.0
    ).to(device).train()

    # 7) 参数选择与冻结
    if args.freeze_backbone:
        # 冻结VAR主体，只训练cross-attention
        for p in cond_var.parameters():
            p.requires_grad = False
        trainable_params = []
        for n, p in cond_var.named_parameters():
            if any(k in n for k in ['cross', 'q_proj', 'k_proj', 'v_proj', 'out_proj', 'ln_q', 'ln_k', 'ln_v', 'gate', 'text_gate']):
                p.requires_grad = True
                trainable_params.append(p)
        
        # 条件编码器参数
        cond_params = list(cond_pack.parameters())
        trainable_params.extend(cond_params)
        
        # 文本适配器参数（如果有）
        if text_adapter is not None:
            text_params = [p for p in text_adapter.parameters() if p.requires_grad]
            trainable_params.extend(text_params)
            
        params_to_train = trainable_params
        print(f"[Freeze] Trainable: {sum(p.numel() for p in params_to_train)/1e6:.2f}M params")
    else:
        # 训练所有参数
        all_params = list(cond_var.parameters()) + list(cond_pack.parameters())
        if text_adapter is not None:
            all_params.extend([p for p in text_adapter.parameters() if p.requires_grad])
        params_to_train = all_params
        print(f"[Full] Trainable: {sum(p.numel() for p in params_to_train)/1e6:.2f}M params")

    # 8) 优化器与调度器
    opt = torch.optim.AdamW(params_to_train, lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95))
    sched = WarmupCosine(opt, warmup=args.warmup, total_steps=total_steps, min_lr=args.min_lr)
    scaler = GradScaler(enabled=not args.no_amp)

    print(f"[Device] {device} | AMP: {not args.no_amp}")

    # 9) 训练循环
    step = 0
    best_val_loss = float('inf')
    
    for ep in range(1, args.epochs+1):
        # ===== 训练 =====
        cond_var.train()
        if text_adapter: text_adapter.train()
        cond_pack.train()
        
        epoch_start = time.time()
        train_metrics = {"loss": 0.0, "acc": 0.0, "samples": 0}
        
        for batch_idx, batch in enumerate(train_dl):
            # 加载数据
            img = batch["img"].to(device, non_blocking=True)        # [B,1,H,W] in [-1,1]
            edge = batch["edge"].to(device, non_blocking=True)      # [B,1,H,W] in [0,1]
            mask = batch["mask"].to(device, non_blocking=True)      # [B,1,H,W] in [0,1]
            sketch = batch["sketch"].to(device, non_blocking=True)  # [B,1,H,W] in [0,1]
            fan = batch["fan"].to(device, non_blocking=True)        # [B,1,H,W] in [0,1]
            
            B = img.size(0)
            
            with torch.no_grad():
                # RVQ编码：图像 → token序列
                z_list, fhat_prefix_list, fhat_lat_prefix_list, _, _ = adapter.quantize_residual(img)
                gt_BL = torch.cat(z_list, dim=1)  # 目标token [B, L]
                x_BLCv_wo_first_l = adapter.vqvae.quantize.idxBl_to_var_input(z_list)  # VAR输入
                
                # 构建条件字典
                conds = {
                    'edge': edge,
                    'mask': mask, 
                    'sketch': sketch,
                    'fan': fan,
                    'coarse_img': fhat_prefix_list[0],  # 第0层前缀，避免信息泄漏
                    # 'coarse_lat': fhat_lat_prefix_list[0],  # 可选：潜空间前缀
                }
                
                # 条件编码：原始图像 → (K,V)对
                grid = adapter.quantizer.v_patch_nums[-1]
                kv_pairs = cond_pack(conds, grid=grid)
                
            # 文本条件编码
            text_BD = None
            if args.use_text and text_adapter is not None:
                text_descs = batch["text_desc"]  # List[str]
                if any(t is not None for t in text_descs):
                    with torch.autocast(device_type=device.type, enabled=not args.no_amp):
                        text_BD = text_adapter(text_descs, device=device)  # [B, d_model]

            # 标签（医学图像通常用虚拟类别0）
            label_B = torch.zeros(B, dtype=torch.long, device=device)

            # 前向传播
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=not args.no_amp):
                logits_BLV = cond_var(
                    label_B=label_B,
                    x_BLCv_wo_first_l=x_BLCv_wo_first_l,
                    kv_pairs=kv_pairs,
                    text_BD=text_BD,
                    open_cross_n=args.open_cross_n
                )
                
                # 交叉熵损失
                loss = F.cross_entropy(
                    logits_BLV.reshape(-1, logits_BLV.size(-1)),
                    gt_BL.reshape(-1)
                )

            # 反向传播
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params_to_train, args.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()

            # 统计
            with torch.no_grad():
                pred_tokens = logits_BLV.argmax(dim=-1)
                acc = (pred_tokens == gt_BL).float().mean().item()
                
            train_metrics["loss"] += loss.item()
            train_metrics["acc"] += acc
            train_metrics["samples"] += B

            # 日志
            if step % args.log_intv == 0:
                # 门控值监控
                gates = []
                for n, p in cond_var.named_parameters():
                    if n.endswith('gate'):
                        gates.append(torch.sigmoid(p.detach()).mean().item())
                
                gate_info = f"gate={np.mean(gates):.3f}" if gates else "no_gates"
                text_gate = f"txt_gate={torch.sigmoid(cond_var.text_gate).item():.3f}" if args.use_text else ""
                
                print(f"[ep{ep:02d} step{step:06d}] "
                      f"loss={loss.item():.4f} acc={acc:.3f} "
                      f"{gate_info} {text_gate} lr={opt.param_groups[0]['lr']:.2e}")

            step += 1

        # 训练epoch统计
        train_loss = train_metrics["loss"] / len(train_dl)
        train_acc = train_metrics["acc"] / len(train_dl)
        epoch_time = time.time() - epoch_start
        
        print(f"[Epoch {ep}] Train - Loss: {train_loss:.4f}, Acc: {train_acc:.3f}, Time: {epoch_time:.1f}s")

        # ===== 验证 =====
        if ep % args.val_intv == 0:
            cond_var.eval()
            if text_adapter: text_adapter.eval()
            cond_pack.eval()
            
            val_metrics = {"loss": 0.0, "acc": 0.0, "samples": 0}
            
            with torch.no_grad():
                for batch in val_dl:
                    img = batch["img"].to(device, non_blocking=True)
                    edge = batch["edge"].to(device, non_blocking=True)
                    mask = batch["mask"].to(device, non_blocking=True)
                    sketch = batch["sketch"].to(device, non_blocking=True)
                    fan = batch["fan"].to(device, non_blocking=True)
                    B = img.size(0)
                    
                    # RVQ编码
                    z_list, fhat_prefix_list, _, _, _ = adapter.quantize_residual(img)
                    gt_BL = torch.cat(z_list, dim=1)
                    x_BLCv_wo_first_l = adapter.vqvae.quantize.idxBl_to_var_input(z_list)
                    
                    # 条件编码
                    conds = {
                        'edge': edge, 'mask': mask, 'sketch': sketch, 'fan': fan,
                        'coarse_img': fhat_prefix_list[0],
                    }
                    kv_pairs = cond_pack(conds, grid=grid)
                    
                    # 文本编码
                    text_BD = None
                    if args.use_text and text_adapter is not None:
                        text_descs = batch["text_desc"]
                        if any(t is not None for t in text_descs):
                            text_BD = text_adapter(text_descs, device=device)
                    
                    label_B = torch.zeros(B, dtype=torch.long, device=device)
                    
                    # 前向
                    with autocast(enabled=not args.no_amp):
                        logits_BLV = cond_var(
                            label_B=label_B,
                            x_BLCv_wo_first_l=x_BLCv_wo_first_l,
                            kv_pairs=kv_pairs,
                            text_BD=text_BD,
                            open_cross_n=args.open_cross_n
                        )
                        
                        loss = F.cross_entropy(
                            logits_BLV.reshape(-1, logits_BLV.size(-1)),
                            gt_BL.reshape(-1)
                        )
                    
                    pred_tokens = logits_BLV.argmax(dim=-1)
                    acc = (pred_tokens == gt_BL).float().mean().item()
                    
                    val_metrics["loss"] += loss.item()
                    val_metrics["acc"] += acc
                    val_metrics["samples"] += B
            
            val_loss = val_metrics["loss"] / len(val_dl)
            val_acc = val_metrics["acc"] / len(val_dl)
            
            print(f"[Epoch {ep}] Val - Loss: {val_loss:.4f}, Acc: {val_acc:.3f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_ckpt = os.path.join(args.out, "best_model.pt")
                torch.save({
                    "epoch": ep,
                    "cond_var": cond_var.state_dict(),
                    "cond_pack": cond_pack.state_dict(),
                    "text_adapter": text_adapter.state_dict() if text_adapter else None,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "args": vars(args)
                }, best_ckpt)
                print(f"[Save] Best model saved: {best_ckpt}")

        # 定期保存checkpoint
        if ep % args.save_intv == 0:
            ckpt_path = os.path.join(args.out, f"epoch_{ep:03d}.pt")
            torch.save({
                "epoch": ep,
                "cond_var": cond_var.state_dict(),
                "cond_pack": cond_pack.state_dict(),
                "text_adapter": text_adapter.state_dict() if text_adapter else None,
                "optimizer": opt.state_dict(),
                "scheduler": sched,
                "args": vars(args)
            }, ckpt_path)
            print(f"[Save] Checkpoint: {ckpt_path}")

    print("Training completed!")


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="CAMUS数据集根目录")
    p.add_argument("--out", type=str, default="ckpts_camus", help="checkpoint输出目录")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--cpu", action="store_true")

    # 模型参数
    p.add_argument("--vocab_size", type=int, default=4096)
    p.add_argument("--depth", type=int, default=16)
    p.add_argument("--open_cross_n", type=int, default=3)
    p.add_argument("--freeze_backbone", action="store_true", help="仅训练cross模块")
    p.add_argument("--var_ckpt", type=str, default="", help="预训练VAR权重")
    
    # 文本条件
    p.add_argument("--use_text", action="store_true", help="启用文本条件")
    p.add_argument("--text_dropout", type=float, default=0.1, help="文本条件dropout")

    # 日志与保存
    p.add_argument("--log_intv", type=int, default=50)
    p.add_argument("--val_intv", type=int, default=2, help="验证间隔")
    p.add_argument("--save_intv", type=int, default=5, help="保存间隔")

    return p.parse_args()


if __name__ == "__main__":
    import numpy as np
    args = build_args()
    train(args)