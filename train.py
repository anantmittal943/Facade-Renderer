"""
Facade-Renderer: Training Pipeline
Pix2Pix Conditional GAN (cGAN) for Facade Generation.
"""

import os
import sys
import time
import argparse
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models import UNetGenerator, PatchGANDiscriminator, weights_init_normal
from dataset import FacadesDataset, find_facades_dir, get_dataloader


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Converts [-1, 1] normalized tensor to [0, 1] range for visualization."""
    return torch.clamp((tensor + 1.0) / 2.0, 0.0, 1.0)


def save_sample_progress(
    generator: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    save_path: str,
    num_samples: int = 4,
):
    """
    Generates and saves visual side-by-side comparison:
    [Condition (Blueprint) | Real Target (Facade) | Generated Facade]
    """
    generator.eval()
    samples_list = []

    with torch.no_grad():
        for i, (condition, real_target) in enumerate(val_loader):
            if i >= num_samples:
                break
            condition = condition.to(device)
            real_target = real_target.to(device)
            fake_target = generator(condition)

            # Denormalize each to [0, 1]
            cond_denorm = denormalize(condition)
            real_denorm = denormalize(real_target)
            fake_denorm = denormalize(fake_target)

            # Stack horizontally: condition, ground truth, prediction
            for b in range(min(condition.size(0), num_samples - len(samples_list) // 3)):
                samples_list.extend([cond_denorm[b], real_denorm[b], fake_denorm[b]])
                if len(samples_list) >= num_samples * 3:
                    break

    if samples_list:
        grid = torch.stack(samples_list, dim=0)
        # nrow=3 arranges each sample row as [Condition, Real, Generated]
        save_image(grid, save_path, nrow=3, padding=4, normalize=False)

    generator.train()


def train(args):
    # Setup directories
    samples_dir = os.path.join(args.output_dir, "samples")
    checkpoints_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Resolve dataset directory
    data_dir = args.data_dir
    if not data_dir or not os.path.exists(data_dir):
        detected = find_facades_dir(args.data_dir)
        if detected:
            data_dir = detected
        else:
            raise FileNotFoundError(
                f"Could not locate CMP Facades dataset at: {args.data_dir}. "
                "Please run `python download_data.py` first to download the dataset into D:\\Datasets."
            )

    print("=" * 60)
    print("Facade-Renderer: Pix2Pix cGAN Training")
    print("=" * 60)
    print(f"Dataset root:    {data_dir}")
    print(f"Output dir:      {args.output_dir}")
    print(f"Epochs:          {args.epochs}")
    print(f"Batch size:      {args.batch_size}")
    print(f"Direction:       {args.direction} (Blueprint -> Facade Photo)")
    print(f"Learning rate:   {args.lr} (Adam beta1={args.beta1})")
    print(f"L1 lambda:       {args.lambda_l1}")

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute device:  {device}")
    if device.type == "cuda":
        print(f"GPU device:      {torch.cuda.get_device_name(0)}")
        print(f"VRAM available:  {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print("=" * 60)

    # Initialize Networks
    generator = UNetGenerator(in_channels=3, out_channels=3).to(device)
    discriminator = PatchGANDiscriminator(in_channels=6).to(device)

    # Weight initialization
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume_generator and os.path.exists(args.resume_generator):
        print(f"Resuming generator from: {args.resume_generator}")
        generator.load_state_dict(torch.load(args.resume_generator, map_location=device))
    if args.resume_discriminator and os.path.exists(args.resume_discriminator):
        print(f"Resuming discriminator from: {args.resume_discriminator}")
        discriminator.load_state_dict(torch.load(args.resume_discriminator, map_location=device))

    # Loss Functions
    # Pix2Pix uses BCEWithLogitsLoss for adversarial loss (numerically stable)
    criterion_GAN = nn.BCEWithLogitsLoss().to(device)
    # L1 Loss preserves sharp edges and structural alignment
    criterion_L1 = nn.L1Loss().to(device)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    # DataLoaders
    train_loader = get_dataloader(
        root_dir=data_dir,
        split="train",
        batch_size=args.batch_size,
        direction=args.direction,
        img_size=args.img_size,
        shuffle=True,
    )

    # Validation loader for tracking visual progress
    val_loader = None
    try:
        val_loader = get_dataloader(
            root_dir=data_dir,
            split="val",
            batch_size=1,
            direction=args.direction,
            img_size=args.img_size,
            shuffle=False,
        )
    except Exception:
        # If val split not separated, use train loader for sample generation
        val_loader = train_loader

    print(f"Training samples: {len(train_loader.dataset)} (Batches per epoch: {len(train_loader)})")

    # Training Loop
    total_steps = 0
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        epoch_l1_loss = 0.0

        generator.train()
        discriminator.train()

        for i, (condition, real_target) in enumerate(train_loader):
            if args.max_steps and (i + 1) > args.max_steps:
                break
            condition = condition.to(device)
            real_target = real_target.to(device)

            # -----------------------------------------------------------
            #  1) Train Generator: minimize -log(D(x, G(x))) + lambda * ||y - G(x)||_1
            # -----------------------------------------------------------
            optimizer_G.zero_grad()

            fake_target = generator(condition)
            pred_fake_for_G = discriminator(condition, fake_target)
            valid_targets = torch.ones_like(pred_fake_for_G, device=device)
            fake_targets = torch.zeros_like(pred_fake_for_G, device=device)

            # Adversarial Loss (fool discriminator)
            loss_G_adversarial = criterion_GAN(pred_fake_for_G, valid_targets)

            # L1 Reconstruction Loss
            loss_G_L1 = criterion_L1(fake_target, real_target)

            # Combined Generator Loss
            loss_G = loss_G_adversarial + (args.lambda_l1 * loss_G_L1)
            loss_G.backward()
            optimizer_G.step()

            # -----------------------------------------------------------
            #  2) Train Discriminator: maximize log(D(x, y)) + log(1 - D(x, G(x)))
            # -----------------------------------------------------------
            optimizer_D.zero_grad()

            # Real pair: (condition, real_target)
            pred_real = discriminator(condition, real_target)
            loss_D_real = criterion_GAN(pred_real, valid_targets)

            # Fake pair: (condition, fake_target.detach())
            pred_fake = discriminator(condition, fake_target.detach())
            loss_D_fake = criterion_GAN(pred_fake, fake_targets)

            # Total D loss: averaged
            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            loss_D.backward()
            optimizer_D.step()

            # Accumulate metrics
            epoch_d_loss += loss_D.item()
            epoch_g_loss += loss_G.item()
            epoch_l1_loss += loss_G_L1.item()
            total_steps += 1

            if (i + 1) % args.log_interval == 0 or (i + 1) == len(train_loader):
                elapsed = time.time() - start_time
                print(
                    f"Epoch [{epoch:03d}/{args.epochs:03d}] "
                    f"Step [{i + 1:03d}/{len(train_loader):03d}] | "
                    f"Loss D: {loss_D.item():.4f} (Real: {loss_D_real.item():.4f}, Fake: {loss_D_fake.item():.4f}) | "
                    f"Loss G: {loss_G.item():.4f} (Adv: {loss_G_adversarial.item():.4f}, L1: {loss_G_L1.item():.4f}) | "
                    f"Time: {elapsed:.1f}s"
                )

        # End of Epoch logging
        avg_d = epoch_d_loss / len(train_loader)
        avg_g = epoch_g_loss / len(train_loader)
        avg_l1 = epoch_l1_loss / len(train_loader)
        print(f"--- Epoch {epoch} Complete --- Avg D: {avg_d:.4f} | Avg G: {avg_g:.4f} | Avg L1: {avg_l1:.4f}")

        # Save Visual Sample Progress
        if epoch % args.sample_interval == 0 or epoch == 1:
            sample_file = os.path.join(samples_dir, f"sample_epoch_{epoch:03d}.png")
            save_sample_progress(generator, val_loader, device, sample_file, num_samples=4)
            print(f"[Visual] Saved sample comparison to: {sample_file}")

        # Save Checkpoints
        if epoch % args.checkpoint_interval == 0 or epoch == args.epochs:
            torch.save(generator.state_dict(), os.path.join(checkpoints_dir, f"generator_epoch_{epoch:03d}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(checkpoints_dir, f"discriminator_epoch_{epoch:03d}.pth"))
            # Save latest checkpoint pointers
            torch.save(generator.state_dict(), os.path.join(checkpoints_dir, "generator_latest.pth"))
            torch.save(discriminator.state_dict(), os.path.join(checkpoints_dir, "discriminator_latest.pth"))
            print(f"[Checkpoint] Saved model checkpoint for epoch {epoch} to {checkpoints_dir}")

    print("\nTraining completed successfully!")
    print(f"Final generator model saved to: {os.path.join(checkpoints_dir, 'generator_latest.pth')}")


def main():
    parser = argparse.ArgumentParser(description="Train Pix2Pix Facade-Renderer")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to CMP Facades dataset (auto-detected if omitted)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save checkpoints and visual samples")
    parser.add_argument("--epochs", type=int, default=200, help="Total training epochs (default: 200)")
    parser.add_argument("--max_steps", type=int, default=None, help="Optional max training steps per epoch (useful for debugging)")
    parser.add_argument("--batch_size", type=int, default=1, help="Mini-batch size (1 to 4 recommended, default: 1)")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for Adam optimizer (default: 0.0002)")
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam beta1 momentum term (default: 0.5)")
    parser.add_argument("--lambda_l1", type=float, default=100.0, help="Weight for L1 pixelwise reconstruction loss (default: 100)")
    parser.add_argument("--img_size", type=int, default=256, help="Target image resolution (default: 256)")
    parser.add_argument("--direction", type=str, default="BtoA", choices=["BtoA", "AtoB"], help="Translation direction (default: BtoA = blueprint->photo)")
    parser.add_argument("--log_interval", type=int, default=20, help="Log step interval within an epoch (default: 20)")
    parser.add_argument("--sample_interval", type=int, default=5, help="Epoch interval to save sample visual grids (default: 5)")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="Epoch interval to save model checkpoints (default: 10)")
    parser.add_argument("--resume_generator", type=str, default=None, help="Path to checkpoint to resume generator")
    parser.add_argument("--resume_discriminator", type=str, default=None, help="Path to checkpoint to resume discriminator")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
