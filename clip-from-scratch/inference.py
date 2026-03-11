"""
inference.py
------------
Run zero-shot image-to-text retrieval on custom inputs.

Given an image and a list of candidate captions, ranks them by
cosine similarity in the shared embedding space.

Usage:
    python inference.py \\
        --image path/to/image.jpg \\
        --captions "A dog playing fetch" "A cat on a sofa" "A person cycling" \\
        --checkpoint ./checkpoints/best
"""

import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.model import CLIPModel
from src.dataset import get_val_transform


def parse_args():
    p = argparse.ArgumentParser(
        description="Rank captions against an image using the trained CLIP model"
    )
    p.add_argument("--image",      type=str, required=True,
                   help="Path to the query image")
    p.add_argument("--captions",   type=str, nargs="+", required=True,
                   help="Candidate captions to rank")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to the checkpoint directory")
    return p.parse_args()


@torch.no_grad()
def rank_captions(model, image_path: str, captions: list[str],
                  device: torch.device) -> list[tuple[str, float]]:
    """
    Rank a list of captions against a single image.

    Returns a list of (caption, score) tuples sorted by score descending.
    """
    model.eval()

    # Load and preprocess image
    transform = get_val_transform()
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)   # (1, 3, 224, 224)

    # Encode image
    img_emb = model.encode_image(img_t)              # (1, 512)

    # Encode all captions
    tokens = model.text_encoder.tokenize(captions, device)
    txt_emb = model.encode_text(
        tokens["input_ids"], tokens["attention_mask"]
    )                                                # (N, 512)

    # Cosine similarity (already normalized, so just dot product)
    scores = (img_emb @ txt_emb.T).squeeze(0)        # (N,)
    scores = scores.cpu().tolist()

    ranked = sorted(zip(captions, scores), key=lambda x: x[1], reverse=True)
    return ranked


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    model = CLIPModel()
    model.load(args.checkpoint)
    model.to(device)

    # Rank captions
    print(f"\nImage: {args.image}")
    print(f"Ranking {len(args.captions)} caption(s)...\n")

    results = rank_captions(model, args.image, args.captions, device)

    print("Results (ranked by similarity):")
    print("-" * 60)
    for i, (caption, score) in enumerate(results, 1):
        bar = "█" * int(score * 40)
        print(f"  #{i}  [{score:.4f}]  {bar}")
        print(f"       {caption}")
    print("-" * 60)


if __name__ == "__main__":
    main()
