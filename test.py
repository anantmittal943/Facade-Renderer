"""
Facade-Renderer: Inference & Evaluation Script
Tests a trained U-Net Generator on test set or individual image and saves side-by-side comparisons.
"""

import os
import argparse
import math
from typing import Optional

from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.utils import save_image

from models import UNetGenerator
from dataset import FacadesDataset, find_facades_dir, get_dataloader
from train import denormalize


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Calculates Peak Signal to Noise Ratio (PSNR) between two [0, 1] images."""
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return float("inf")
    return 20 * math.log10(1.0 / math.sqrt(mse))


def evaluate_dataset(
    generator: nn.Module,
    data_dir: str,
    output_dir: str,
    device: torch.device,
    direction: str = "BtoA",
    num_samples: Optional[int] = None,
):
    """Evaluates generator on test dataset and saves comparison grids."""
    test_loader = get_dataloader(
        root_dir=data_dir,
        split="test",
        batch_size=1,
        direction=direction,
        shuffle=False,
    )

    results_dir = os.path.join(output_dir, "comparisons")
    generated_only_dir = os.path.join(output_dir, "generated")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(generated_only_dir, exist_ok=True)

    generator.eval()
    total_l1 = 0.0
    total_psnr = 0.0
    count = 0

    print(f"Running inference on {len(test_loader)} test samples...")

    with torch.no_grad():
        for i, (condition, real_target) in enumerate(test_loader):
            if num_samples and i >= num_samples:
                break

            condition = condition.to(device)
            real_target = real_target.to(device)

            # Generate output
            fake_target = generator(condition)

            # Denormalize to [0, 1]
            cond_denorm = denormalize(condition)
            real_denorm = denormalize(real_target)
            fake_denorm = denormalize(fake_target)

            # Compute metrics
            l1_err = torch.mean(torch.abs(fake_denorm - real_denorm)).item()
            psnr = calculate_psnr(fake_denorm, real_denorm)
            total_l1 += l1_err
            total_psnr += psnr
            count += 1

            # Save comparison grid: [Condition, Real, Generated]
            comparison = torch.cat([cond_denorm, real_denorm, fake_denorm], dim=3)
            save_image(comparison, os.path.join(results_dir, f"test_{i+1:04d}_comparison.png"))

            # Save generated image separately
            save_image(fake_denorm, os.path.join(generated_only_dir, f"test_{i+1:04d}_generated.png"))

            if (i + 1) % 10 == 0 or (i + 1) == len(test_loader):
                print(f"Processed [{i+1:03d}/{len(test_loader):03d}] | Current L1: {l1_err:.4f} | PSNR: {psnr:.2f} dB")

    avg_l1 = total_l1 / count if count > 0 else 0.0
    avg_psnr = total_psnr / count if count > 0 else 0.0

    print("=" * 60)
    print("Evaluation Summary:")
    print(f"Total samples evaluated: {count}")
    print(f"Mean L1 Error:          {avg_l1:.4f}")
    print(f"Mean PSNR:              {avg_psnr:.2f} dB")
    print(f"Side-by-side results:   {results_dir}")
    print(f"Generated facades only: {generated_only_dir}")
    print("=" * 60)


def predict_single_image(
    generator: nn.Module,
    image_path: str,
    output_path: str,
    device: torch.device,
    is_paired: bool = True,
    direction: str = "BtoA",
):
    """Runs inference on a single test image (either composite paired image or raw blueprint)."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    if is_paired and w >= 2 * h:
        # Sliced composite pair
        half_w = w // 2
        left = img.crop((0, 0, half_w, h))
        right = img.crop((half_w, 0, w, h))
        blueprint = right if direction.upper() == "BTOA" else left
    else:
        # Standalone blueprint image
        blueprint = img

    blueprint_resized = TF.resize(blueprint, [256, 256], interpolation=T.InterpolationMode.BICUBIC)
    blueprint_tensor = TF.to_tensor(blueprint_resized)
    blueprint_tensor = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(blueprint_tensor).unsqueeze(0).to(device)

    generator.eval()
    with torch.no_grad():
        generated = generator(blueprint_tensor)
        generated_denorm = denormalize(generated)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_image(generated_denorm, output_path)
    print(f"Generated facade saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Facade-Renderer cGAN")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to generator checkpoint (.pth)")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to CMP Facades dataset (auto-detected if omitted)")
    parser.add_argument("--output_dir", type=str, default="outputs/test_results", help="Directory to save test outputs")
    parser.add_argument("--input_image", type=str, default=None, help="Optional path to a single input blueprint image")
    parser.add_argument("--direction", type=str, default="BtoA", choices=["BtoA", "AtoB"], help="Translation direction")
    parser.add_argument("--num_samples", type=int, default=None, help="Limit number of test samples to evaluate")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"Compute device:     {device}")

    generator = UNetGenerator(in_channels=3, out_channels=3).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    generator.load_state_dict(state_dict)
    generator.eval()

    if args.input_image:
        out_file = os.path.join(args.output_dir, "single_prediction.png")
        predict_single_image(generator, args.input_image, out_file, device, direction=args.direction)
    else:
        data_dir = args.data_dir or find_facades_dir()
        if not data_dir:
            raise FileNotFoundError("Could not locate CMP Facades dataset. Specify --data_dir.")
        evaluate_dataset(generator, data_dir, args.output_dir, device, direction=args.direction, num_samples=args.num_samples)


if __name__ == "__main__":
    main()
