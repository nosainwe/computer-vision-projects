"""
dataset.py
----------
Flickr30k dataset loader for CLIP training.

Dataset: Young et al. (2014) "From image descriptions to visual denotations"
  https://shannon.cs.illinois.edu/DenotationGraph/
  ~31,000 images, 5 human-written captions each (155,000 total pairs)

Download instructions: see data/README.md

CSV format expected:
    image_name | comment_number | comment
    1000092795.jpg | 0 | Two young guys with shaggy hair...
    1000092795.jpg | 1 | Two young men in jeans...

During training, we sample ONE random caption per image per batch
to maximize data diversity. All 5 captions are used across epochs.
"""

import os
import random
from pathlib import Path

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ---------------------------------------------------------------------------
# Image Transforms
# ---------------------------------------------------------------------------

def get_train_transform(img_size: int = 224) -> transforms.Compose:
    """Standard CLIP training augmentations."""
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0),
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_val_transform(img_size: int = 224) -> transforms.Compose:
    """Deterministic transform for evaluation."""
    return transforms.Compose([
        transforms.Resize(img_size + 32, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Flickr30kDataset(Dataset):
    """
    Flickr30k image-caption dataset.

    In training mode, one caption is sampled randomly from the 5 available
    per image on each access. In eval mode, all captions are used.

    Args:
        data_dir (str): Root directory containing flickr30k_images/ and results.csv
        split    (str): 'train', 'val', or 'test'
        transform:      torchvision transform applied to images
        train_ratio (float): fraction of data used for training
        val_ratio   (float): fraction for validation (rest is test)
        seed        (int): random seed for reproducible splits
    """

    # Official Flickr30k has no predefined split; we define one here.
    TRAIN_RATIO = 0.85
    VAL_RATIO   = 0.075
    # TEST_RATIO  = 0.075 (implicit)

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform=None,
        seed: int = 42,
    ):
        assert split in ("train", "val", "test"), \
            f"split must be one of train/val/test, got '{split}'"

        self.img_dir = Path(data_dir) / "flickr30k_images"
        self.transform = transform or (
            get_train_transform() if split == "train" else get_val_transform()
        )
        self.is_train = (split == "train")

        # Load and parse CSV
        csv_path = Path(data_dir) / "results.csv"
        df = pd.read_csv(csv_path, sep="|", header=0,
                         names=["image_name", "comment_number", "comment"])

        # Clean up whitespace artifacts from the separator
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        # Group: image_name → list of captions
        grouped = df.groupby("image_name")["comment"].apply(list).reset_index()
        grouped.columns = ["image_name", "captions"]

        # Reproducible split
        rng = random.Random(seed)
        images = grouped["image_name"].tolist()
        rng.shuffle(images)

        n = len(images)
        n_train = int(n * self.TRAIN_RATIO)
        n_val   = int(n * self.VAL_RATIO)

        split_images = {
            "train": set(images[:n_train]),
            "val":   set(images[n_train:n_train + n_val]),
            "test":  set(images[n_train + n_val:]),
        }[split]

        self.data = grouped[grouped["image_name"].isin(split_images)].reset_index(drop=True)
        print(f"Flickr30k [{split}]: {len(self.data)} images loaded")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data.iloc[idx]
        img_path = self.img_dir / row["image_name"]

        # Load image
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        # Sample one caption (random during train, first during eval)
        if self.is_train:
            caption = random.choice(row["captions"])
        else:
            caption = row["captions"][0]

        return {"image": img, "caption": caption, "image_name": row["image_name"]}


# ---------------------------------------------------------------------------
# Collate + DataLoader factory
# ---------------------------------------------------------------------------

def collate_fn(batch: list[dict]) -> dict:
    """Stack images into a tensor; keep captions as a list of strings."""
    return {
        "images":      torch.stack([b["image"] for b in batch]),
        "captions":    [b["caption"] for b in batch],
        "image_names": [b["image_name"] for b in batch],
    }


def get_dataloader(
    data_dir: str,
    split: str = "train",
    batch_size: int = 64,
    num_workers: int = 4,
    transform=None,
    seed: int = 42,
) -> DataLoader:
    """Convenience factory for getting a DataLoader for a given split."""
    dataset = Flickr30kDataset(data_dir, split=split,
                               transform=transform, seed=seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=(split == "train"),  # keep batch sizes uniform during training
    )
