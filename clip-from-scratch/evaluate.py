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
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--data_dir",    type=str, default="./data")
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
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

    all_img_embs = []
    all_txt_embs = []
    gt_labels = []

    model.eval()
    img_idx = 0

    # iterating row by row instead of batching — needed because we want all 5 captions per image
    # batching would mix captions from different images and complicate the gt_labels bookkeeping
    for row_idx in tqdm(range(len(dataset.data)), desc="Encoding"):
        row = dataset.data.iloc[row_idx]

        from pathlib import Path
        from PIL import Image

        img_path = dataset.img_dir / row["image_name"]
        img = Image.open(img_path).convert("RGB")

        # unsqueeze to add the batch dim — model expects (B, C, H, W) not (C, H, W)
        img_t = dataset.transform(img).unsqueeze(0).to(device)

        img_emb = model.encode_image(img_t)   # (1, D)
        all_img_embs.append(img_emb.cpu())

        # encoding each of the 5 captions separately — keeping them associated with img_idx
        for cap in row["captions"]:
            tokens = model.text_encoder.tokenize([cap], device)
            txt_emb = model.encode_text(tokens["input_ids"], tokens["attention_mask"])
            all_txt_embs.append(txt_emb.cpu())
            gt_labels.append(img_idx)

        img_idx += 1

    return (
        torch.cat(all_img_embs, dim=0),    # (N, D)
        torch.cat(all_txt_embs, dim=0),    # (N*5, D)
        torch.tensor(gt_labels),           # (N*5,) — maps each caption back to its image
    )


def recall_at_k(scores: torch.Tensor, gt: torch.Tensor,
                ks: list[int] = [1, 5, 10]) -> dict:
    # argsort descending gives ranked candidates — highest similarity first
    results = {}
    ranked = scores.argsort(dim=-1, descending=True)

    for k in ks:
        top_k = ranked[:, :k]
        # checking if the correct target appears anywhere in the top k — any() across dim=1
        hits = (top_k == gt.unsqueeze(1)).any(dim=1).float()
        results[f"R@{k}"] = hits.mean().item() * 100

    return results


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint from {args.checkpoint}...")
    model = CLIPModel()
    model.load(args.checkpoint)
    model.to(device)
    model.eval()

    # using val transform here — no augmentation at eval time, ever
    dataset = Flickr30kDataset(
        args.data_dir, split="test",
        transform=get_val_transform()
    )

    print("\nComputing embeddings...")
    img_embs, txt_embs, gt_labels = compute_embeddings(model, dataset, device)

    # dot product similarity — embeddings are L2 normalized so this equals cosine sim
    print("\nComputing similarity matrix...")
    scores_i2t = img_embs @ txt_embs.T    # (N_img, N_txt) — each image vs all captions
    scores_t2i = txt_embs @ img_embs.T    # (N_txt, N_img) — each caption vs all images

    N_img = img_embs.size(0)

    # for t2i: caption i belongs to image i//5 — repeat_interleave builds that mapping
    img_gt = torch.arange(N_img).repeat_interleave(5)

    print("\n" + "="*50)
    print("Retrieval Results — Flickr30k Test Set")
    print("="*50)

    # for i2t: using first caption per image as the gt query — standard Flickr30k protocol
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
