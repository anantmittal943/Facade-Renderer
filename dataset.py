"""
Facade-Renderer: Dataset & DataLoader
Handles paired CMP Facades images (blueprint/segmentation map -> photo).
"""

import os
import re
import glob
import random
from typing import Tuple, Optional
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF


def natural_sort_key(s: str):
    """Sort strings containing numbers in natural numeric order (1, 2, ... 10, 106)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def find_facades_dir(base_hint: Optional[str] = None) -> Optional[str]:
    """
    Auto-detects the CMP Facades dataset directory across common locations.
    Checks user-specified path, D:\\Datasets, and current working directory.
    """
    candidates = []
    if base_hint:
        candidates.extend([
            base_hint,
            os.path.join(base_hint, "facades"),
            os.path.join(base_hint, "datasets", "vikramtiwari", "pix2pix-dataset", "versions", "2", "facades"),
            os.path.join(base_hint, "datasets", "vikramtiwari", "pix2pix-dataset", "facades"),
            os.path.join(base_hint, "pix2pix-dataset", "facades"),
        ])
    
    # Standard locations
    candidates.extend([
        r"D:\Datasets\facades",
        r"D:\Datasets\datasets\vikramtiwari\pix2pix-dataset\versions\2\facades",
        r"D:\Datasets\datasets\vikramtiwari\pix2pix-dataset\facades",
        r"D:\Datasets\pix2pix-dataset\facades",
        r"D:\Datasets",
        "./facades",
        "./data/facades",
    ])

    for path in candidates:
        if os.path.exists(path):
            # Check if this folder contains train/val/test subdirectories or images directly
            if os.path.exists(os.path.join(path, "train")):
                return os.path.abspath(path)
            # Or if it's already the split folder
            if glob.glob(os.path.join(path, "*.jpg")) or glob.glob(os.path.join(path, "*.png")):
                return os.path.abspath(path)

    return None


class FacadesDataset(Dataset):
    """
    Dataset for CMP Facades paired samples.
    Each sample contains input blueprint and real photo stored side-by-side (512x256).
    
    Standard Pix2Pix layout:
      - Left half (A): Real building photograph
      - Right half (B): Blueprint / semantic label map
    
    Default direction: 'BtoA' (Blueprint -> Real Photo)
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        direction: str = "BtoA",
        img_size: int = 256,
    ):
        super().__init__()
        self.direction = direction.upper()
        self.split = split
        self.img_size = img_size

        # Resolve directory: check root_dir/split or root_dir directly
        split_dir = os.path.join(root_dir, split)
        if os.path.exists(split_dir):
            files = (
                glob.glob(os.path.join(split_dir, "*.jpg")) +
                glob.glob(os.path.join(split_dir, "*.png"))
            )
        else:
            files = (
                glob.glob(os.path.join(root_dir, "*.jpg")) +
                glob.glob(os.path.join(root_dir, "*.png"))
            )
        self.image_paths = sorted(files, key=natural_sort_key)

        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No images found for split '{split}' in: {root_dir}. "
                f"Please ensure dataset is downloaded using download_data.py."
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        composite_img = Image.open(img_path).convert("RGB")

        # Slice composite image into left and right halves
        w, h = composite_img.size
        half_w = w // 2
        left_img = composite_img.crop((0, 0, half_w, h))
        right_img = composite_img.crop((half_w, 0, w, h))

        # Determine condition and target based on direction
        # BtoA: Right (blueprint) -> Left (photo)
        # AtoB: Left (photo) -> Right (blueprint)
        if self.direction == "BTOA":
            condition = right_img
            target = left_img
        else:
            condition = left_img
            target = right_img

        # Data Preprocessing & Augmentations
        if self.split == "train":
            # 1. Random Jitter: resize to 286x286, then random crop to 256x256
            jitter_size = int(self.img_size * 286 / 256)
            condition = TF.resize(condition, [jitter_size, jitter_size], interpolation=T.InterpolationMode.BICUBIC)
            target = TF.resize(target, [jitter_size, jitter_size], interpolation=T.InterpolationMode.BICUBIC)

            # Synchronous random crop
            i, j, h_crop, w_crop = T.RandomCrop.get_params(
                condition, output_size=(self.img_size, self.img_size)
            )
            condition = TF.crop(condition, i, j, h_crop, w_crop)
            target = TF.crop(target, i, j, h_crop, w_crop)

            # Synchronous random horizontal flip
            if random.random() > 0.5:
                condition = TF.hflip(condition)
                target = TF.hflip(target)
        else:
            # Val/Test: deterministic resize to target size
            condition = TF.resize(condition, [self.img_size, self.img_size], interpolation=T.InterpolationMode.BICUBIC)
            target = TF.resize(target, [self.img_size, self.img_size], interpolation=T.InterpolationMode.BICUBIC)

        # Convert to Tensor [0, 1] and Normalize to [-1, 1]
        condition_tensor = TF.to_tensor(condition)
        target_tensor = TF.to_tensor(target)

        # Normalize with mean=0.5, std=0.5 for all 3 channels
        normalize = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        condition_tensor = normalize(condition_tensor)
        target_tensor = normalize(target_tensor)

        return condition_tensor, target_tensor


def get_dataloader(
    root_dir: str,
    split: str = "train",
    batch_size: int = 1,
    direction: str = "BtoA",
    img_size: int = 256,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
) -> DataLoader:
    """
    Creates and returns a PyTorch DataLoader for the Facades dataset.
    """
    if shuffle is None:
        shuffle = (split == "train")

    dataset = FacadesDataset(
        root_dir=root_dir,
        split=split,
        direction=direction,
        img_size=img_size,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == "train"),
    )
