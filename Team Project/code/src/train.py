"""
Training loop for DiT-based diffusion model on MVTec AD.

Usage:
    python -m src.train --data_root data/mvtec --category hazelnut --epochs 200
    python -m src.train --data_root data/mvtec --category bottle --model tiny --epochs 100
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from src.dataset import MVTecDataset, get_dataloaders
from src.diffusion import GaussianDiffusion, cosine_beta_schedule
from src.dit import DiT_S, DiT_Tiny


def parse_args():
    parser = argparse.ArgumentParser(description="Train DiT diffusion model on MVTec AD")
    parser.add_argument("--data_root", type=str, required=True, help="Path to mvtec/ folder")
    parser.add_argument("--category", type=str, required=True, help="MVTec category name")
    parser.add_argument("--output_dir", type=str, default="output/checkpoints", help="Checkpoint output dir")
    parser.add_argument("--model", type=str, default="small", choices=["small", "tiny"], help="DiT variant")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=50, help="Save checkpoint every N epochs")
    parser.add_argument("--amp", action="store_true", default=True, help="Use automatic mixed precision")
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def get_lr_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    """Linear warmup then cosine decay."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    train_loader,
    optimizer,
    scaler: GradScaler,
    device: str,
    use_amp: bool,
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in train_loader:
        images = batch.to(device)
        B = images.shape[0]

        # Sample random timesteps
        t = torch.randint(0, diffusion.timesteps, (B,), device=device)

        # Sample noise and create noised images
        noise = torch.randn_like(images)
        x_t = diffusion.q_sample(images, t, noise)

        # Predict noise
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            predicted_noise = model(x_t, t)
            loss = nn.functional.mse_loss(predicted_noise, noise)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save training checkpoint."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, path)


def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Category: {args.category}")
    print(f"Model: DiT-{args.model.capitalize()}")

    # Data
    train_loader, _ = get_dataloaders(
        args.data_root, args.category,
        img_size=args.img_size, batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Train batches: {len(train_loader)}")

    # Model
    if args.model == "small":
        model = DiT_S(img_size=args.img_size).to(device)
    else:
        model = DiT_Tiny(img_size=args.img_size).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    # Diffusion
    betas = cosine_beta_schedule(args.timesteps)
    diffusion = GaussianDiffusion(betas, device=device)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_lr_scheduler(optimizer, args.warmup_epochs, args.epochs)
    scaler = GradScaler(enabled=args.amp)

    # Output directory
    output_dir = Path(args.output_dir) / args.category
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_loss = float("inf")
    history = []

    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        avg_loss = train_one_epoch(
            model, diffusion, train_loader, optimizer, scaler, device, args.amp
        )
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        history.append({"epoch": epoch, "loss": avg_loss, "lr": lr, "time": elapsed})

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{args.epochs} | Loss: {avg_loss:.6f} | LR: {lr:.6f} | Time: {elapsed:.1f}s")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, optimizer, epoch, avg_loss, output_dir / "best.pt")

        # Periodic save
        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, avg_loss, output_dir / f"epoch_{epoch:04d}.pt")

    # Save final
    save_checkpoint(model, optimizer, args.epochs, avg_loss, output_dir / "final.pt")

    # Save training history
    with open(output_dir / "history.json", "w") as f:
        json.dump({"args": vars(args), "history": history}, f, indent=2)

    print("-" * 60)
    print(f"Training complete. Best loss: {best_loss:.6f}")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
