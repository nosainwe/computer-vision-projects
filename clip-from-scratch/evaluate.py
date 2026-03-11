"""
evaluate.py
-----------
Recall@K evaluation for image-text retrieval on Flickr30k test split.

Metrics reported:
    - Image→Text Recall@1, @5, @10
    - Text→Image Recall@1, @5, @10

Standard protocol: uses ALL 5 captions per image (not just one),
giving 5x as many text queries as images.

Usage:
    python evaluate.py --checkpoint ./checkpoints/best --data_dir ./data
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from src.model import CLIPModel
from src.dataset import get_dataloader, Flickr30kDataset, get_val_transform


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_dir",   type=str, default="./data")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers",type=int, default=4)
    return p.parse_args()


@torch.no_grad()
def compute_embeddings(model, dataset, device, batch_size: int = 64):
    """
    Compute all image and text embeddings for the test set.
    Returns:
        img_embs:  (N_images, D)
        txt_embs:  (N_images * 5, D)  — 5 captions per image
        gt_labels: (N_images * 5,)    — which image each caption belongs to
    """
    from torch.utils.data import DataLoader
    from src.dataset import collate_fn

    # We need all 5 captions per image, so we build a custom loader
    all_img_embs = []
    all_txt_embs = []
    gt_labels = []

    model.eval()
    img_idx = 0

    for row_idx in tqdm(range(len(dataset.data)), desc="Encoding"):
        row = dataset.data.iloc[row_idx]

        from pathlib import Path
        from PIL import Image
        img_path = dataset.img_dir / row["image_name"]
        img = Image.open(img_path).convert("RGB")
        img_t = dataset.transform(img).unsqueeze(0).to(device)

        # Encode image
        img_emb = model.encode_image(img_t)   # (1, D)
        all_img_embs.append(img_emb.cpu())

        # Encode all captions
        for cap in row["captions"]:
            tokens = model.text_encoder.tokenize([cap], device)
            txt_emb = model.encode_text(tokens["input_ids"], tokens["attention_mask"])
            all_txt_embs.append(txt_emb.cpu())
            gt_labels.append(img_idx)

        img_idx += 1

    return (
        torch.cat(all_img_embs, dim=0),    # (N, D)
        torch.cat(all_txt_embs, dim=0),    # (N*5, D)
        torch.tensor(gt_labels),           # (N*5,)
    )


def recall_at_k(scores: torch.Tensor, gt: torch.Tensor,
                ks: list[int] = [1, 5, 10]) -> dict:
    """
    scores: (N_queries, N_targets) — higher = more similar
    gt:     (N_queries,) — index of correct target for each query
    """
    results = {}
    ranked = scores.argsort(dim=-1, descending=True)

    for k in ks:
        top_k = ranked[:, :k]
        hits = (top_k == gt.unsqueeze(1)).any(dim=1).float()
        results[f"R@{k}"] = hits.mean().item() * 100

    return results


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Loading checkpoint from {args.checkpoint}...")
    model = CLIPModel()
    model.load(args.checkpoint)
    model.to(device)
    model.eval()

    # Load test dataset (eval transform, all captions)
    dataset = Flickr30kDataset(
        args.data_dir, split="test",
        transform=get_val_transform()
    )

    # Embed everything
    print("\nComputing embeddings...")
    img_embs, txt_embs, gt_labels = compute_embeddings(model, dataset, device)

    # Similarity matrix
    print("\nComputing similarity matrix...")
    scores_i2t = img_embs @ txt_embs.T          # (N_img, N_txt)
    scores_t2i = txt_embs @ img_embs.T          # (N_txt, N_img)

    # Build image ground truth for text queries
    # Each image i has 5 corresponding caption indices: [5i, 5i+1, ..., 5i+4]
    N_img = img_embs.size(0)
    img_gt = torch.arange(N_img).repeat_interleave(5)  # for t2i

    # Recall@K
    print("\n" + "="*50)
    print("Retrieval Results — Flickr30k Test Set")
    print("="*50)

    i2t = recall_at_k(scores_i2t, gt_labels.view(N_img, 5)[:, 0])
    print("\nImage → Text Retrieval:")
    for k, v in i2t.items():
        print(f"  {k}: {v:.1f}%")

    t2i = recall_at_k(scores_t2i, img_gt)
    print("\nText → Image Retrieval:")
    for k, v in t2i.items():
        print(f"  {k}: {v:.1f}%")

    print("="*50)


if __name__ == "__main__":
    main()
