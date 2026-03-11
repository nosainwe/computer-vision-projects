"""
train.py
--------
Main training script for CLIP from scratch on Flickr30k.

Usage:
    python train.py --data_dir ./data --epochs 20 --batch_size 64

Training loop:
    1. Load Flickr30k image-caption pairs
    2. Encode images with ViT, encode captions with QLoRA LM
    3. Project both to 512-dim shared space
    4. Compute symmetric InfoNCE loss over (B x B) similarity matrix
    5. Backprop — only LoRA adapters, projection heads, and ViT are updated
       (quantized LM base weights are frozen by bitsandbytes)
"""

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.model import CLIPModel
from src.loss import CLIPLoss, contrastive_accuracy
from src.dataset import get_dataloader


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train CLIP from scratch on Flickr30k")

    # Data
    p.add_argument("--data_dir",    type=str, default="./data",
                   help="Root dir containing flickr30k_images/ and results.csv")
    p.add_argument("--num_workers", type=int, default=4)

    # Model
    p.add_argument("--embed_dim",   type=int, default=512,
                   help="Shared embedding dimension")
    p.add_argument("--lora_rank",   type=int, default=16)
    p.add_argument("--lora_alpha",  type=int, default=32)
    p.add_argument("--max_text_len",type=int, default=77)

    # Training
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--weight_decay",type=float, default=0.1)
    p.add_argument("--grad_clip",   type=float, default=1.0)
    p.add_argument("--warmup_steps",type=int,   default=200,
                   help="Linear LR warmup steps")

    # Checkpointing
    p.add_argument("--save_dir",    type=str,   default="./checkpoints")
    p.add_argument("--save_every",  type=int,   default=5,
                   help="Save checkpoint every N epochs")
    p.add_argument("--resume",      type=str,   default=None,
                   help="Path to checkpoint directory to resume from")

    # Misc
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--log_every",   type=int,   default=50,
                   help="Log training stats every N steps")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lr(optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


def warmup_lr(step: int, warmup_steps: int, base_lr: float) -> float:
    """Linear warmup schedule, used in the first warmup_steps steps."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device,
                    epoch, args, global_step):
    model.train()
    total_loss = 0.0
    total_acc_i2t = 0.0
    total_acc_t2i = 0.0
    n_batches = 0
    t0 = time.time()

    for batch_idx, batch in enumerate(loader):
        images = batch["images"].to(device)
        captions = batch["captions"]

        # Tokenize captions on-device via the text encoder's tokenizer
        tokens = model.text_encoder.tokenize(captions, device)

        # Forward
        out = model(images, tokens["input_ids"], tokens["attention_mask"])
        loss = criterion(out["logits"])

        # Backward
        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # LR warmup (manual override for first warmup_steps)
        if global_step < args.warmup_steps:
            lr = warmup_lr(global_step, args.warmup_steps, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        # Metrics
        accs = contrastive_accuracy(out["logits"])
        total_loss     += loss.item()
        total_acc_i2t  += accs["acc_i2t"]
        total_acc_t2i  += accs["acc_t2i"]
        n_batches += 1
        global_step += 1

        if (batch_idx + 1) % args.log_every == 0:
            elapsed = time.time() - t0
            print(
                f"  Epoch {epoch:3d} | Step {batch_idx+1:4d}/{len(loader)} "
                f"| Loss {loss.item():.4f} "
                f"| i2t {accs['acc_i2t']:.3f} "
                f"| t2i {accs['acc_t2i']:.3f} "
                f"| τ {out['temperature']:.4f} "
                f"| lr {get_lr(optimizer):.2e} "
                f"| {elapsed:.1f}s"
            )
            t0 = time.time()

    return {
        "loss":     total_loss / n_batches,
        "acc_i2t":  total_acc_i2t / n_batches,
        "acc_t2i":  total_acc_t2i / n_batches,
    }, global_step


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc_i2t = 0.0
    total_acc_t2i = 0.0
    n_batches = 0

    for batch in loader:
        images = batch["images"].to(device)
        tokens = model.text_encoder.tokenize(batch["captions"], device)
        out = model(images, tokens["input_ids"], tokens["attention_mask"])
        loss = criterion(out["logits"])
        accs = contrastive_accuracy(out["logits"])

        total_loss    += loss.item()
        total_acc_i2t += accs["acc_i2t"]
        total_acc_t2i += accs["acc_t2i"]
        n_batches += 1

    return {
        "loss":    total_loss / n_batches,
        "acc_i2t": total_acc_i2t / n_batches,
        "acc_t2i": total_acc_t2i / n_batches,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Data
    print("\nLoading dataset...")
    train_loader = get_dataloader(args.data_dir, "train",
                                  args.batch_size, args.num_workers)
    val_loader   = get_dataloader(args.data_dir, "val",
                                  args.batch_size, args.num_workers)

    # Model
    print("\nBuilding model...")
    model = CLIPModel(
        embed_dim=args.embed_dim,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        max_text_len=args.max_text_len,
    )

    if args.resume:
        print(f"Resuming from {args.resume}")
        model.load(args.resume)

    # Optimizer — apply weight decay only to non-bias, non-norm params
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}
    param_groups = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(param_groups, lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = CLIPLoss()

    # Training
    print(f"\nStarting training: {args.epochs} epochs\n{'='*60}")
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_stats, global_step = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch, args, global_step
        )

        val_stats = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"\nEpoch {epoch:3d}/{args.epochs} | "
            f"Train loss {train_stats['loss']:.4f} | "
            f"Val loss {val_stats['loss']:.4f} | "
            f"Val i2t {val_stats['acc_i2t']:.3f} | "
            f"Val t2i {val_stats['acc_t2i']:.3f}\n"
        )

        # Save best checkpoint
        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            model.save(os.path.join(args.save_dir, "best"))
            print(f"  ✓ New best model saved (val_loss={best_val_loss:.4f})")

        # Periodic save
        if epoch % args.save_every == 0:
            model.save(os.path.join(args.save_dir, f"epoch_{epoch:02d}"))
            print(f"  ✓ Checkpoint saved: epoch_{epoch:02d}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    model.save(os.path.join(args.save_dir, "final"))


if __name__ == "__main__":
    main()
