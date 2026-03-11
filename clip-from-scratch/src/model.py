"""
model.py
--------
Full CLIP model: vision encoder + text encoder + dual projection heads.

Both encoders are mapped into a shared 512-dimensional embedding space
via learned linear projection heads. A single learned temperature
parameter scales the cosine similarity logits before the InfoNCE loss.

Architecture inspired by:
  Radford et al. (2021) "Learning Transferable Visual Models From
  Natural Language Supervision" https://arxiv.org/abs/2103.00020
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision_encoder import VisionTransformer
from .text_encoder import TextEncoder


class ProjectionHead(nn.Module):
    """
    Two-layer MLP projection head mapping encoder output → shared space.

    Structure: Linear → GELU → LayerNorm → Linear
    The final output is NOT normalized here — normalization happens
    in the forward pass of CLIPModel before similarity computation.
    """

    def __init__(self, in_dim: int, out_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CLIPModel(nn.Module):
    """
    CLIP: Contrastive Language-Image Pretraining from scratch.

    Components:
        vision_encoder  — ViT-B/16, random init, outputs 768-dim
        text_encoder    — sihab/slm-1.0, 4-bit QLoRA, outputs hidden_dim
        img_proj        — ProjectionHead: 768 → embed_dim
        txt_proj        — ProjectionHead: hidden_dim → embed_dim
        temperature     — learned scalar log(1/τ), initialized at log(1/0.07)
    """

    def __init__(
        self,
        embed_dim: int = 512,
        # ViT config
        img_size: int = 224,
        patch_size: int = 16,
        vit_depth: int = 12,
        vit_heads: int = 12,
        vit_embed_dim: int = 768,
        # Text encoder config
        lm_model_id: str = "sihab/slm-1.0",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        max_text_len: int = 77,
    ):
        super().__init__()

        # --- Vision Tower ---
        self.vision_encoder = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=vit_embed_dim,
            depth=vit_depth,
            num_heads=vit_heads,
        )

        # --- Text Tower ---
        self.text_encoder = TextEncoder(
            model_id=lm_model_id,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            max_length=max_text_len,
        )

        # --- Projection Heads ---
        self.img_proj = ProjectionHead(vit_embed_dim, embed_dim)
        self.txt_proj = ProjectionHead(self.text_encoder.hidden_dim, embed_dim)

        # --- Learned Temperature ---
        # Initialized to log(1/0.07) ≈ 2.659 — same as original CLIP.
        # We store log(τ) and exponentiate in the loss to keep τ > 0.
        self.temperature = nn.Parameter(
            torch.ones([]) * math.log(1 / 0.07)
        )

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, 224, 224)
        Returns:
            (B, embed_dim) — L2-normalized image embeddings
        """
        feats = self.vision_encoder(images)   # (B, 768)
        emb = self.img_proj(feats)            # (B, embed_dim)
        return F.normalize(emb, dim=-1)

    def encode_text(self, input_ids: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids:      (B, seq_len)
            attention_mask: (B, seq_len)
        Returns:
            (B, embed_dim) — L2-normalized text embeddings
        """
        feats = self.text_encoder(input_ids, attention_mask)   # (B, hidden_dim)
        emb = self.txt_proj(feats)                             # (B, embed_dim)
        return F.normalize(emb, dim=-1)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict:
        """
        Full forward pass for training.

        Returns a dict with:
            img_emb   — (B, embed_dim) normalized image embeddings
            txt_emb   — (B, embed_dim) normalized text embeddings
            logits    — (B, B) scaled similarity matrix
            temperature — current τ value (for logging)
        """
        img_emb = self.encode_image(images)
        txt_emb = self.encode_text(input_ids, attention_mask)

        # Scale by learned temperature
        # We clamp log(τ) to prevent it collapsing to near-zero
        log_tau = self.temperature.clamp(max=4.6052)   # τ ≤ 100
        logits = (img_emb @ txt_emb.T) / log_tau.exp()

        return {
            "img_emb": img_emb,
            "txt_emb": txt_emb,
            "logits": logits,
            "temperature": log_tau.exp().item(),
        }

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save(self, path: str):
        """
        Save all model components.

        ⚠️  The QLoRA text encoder CANNOT be torch.save()'d as a whole.
            We save LoRA adapters separately via HuggingFace's save_pretrained,
            and save the remaining non-quantized components with torch.save.
        """
        import os
        os.makedirs(path, exist_ok=True)

        # LoRA adapters + tokenizer
        self.text_encoder.save(f"{path}/text_lora")

        # Everything else
        torch.save({
            "vision_encoder": self.vision_encoder.state_dict(),
            "img_proj":        self.img_proj.state_dict(),
            "txt_proj":        self.txt_proj.state_dict(),
            "temperature":     self.temperature.data,
        }, f"{path}/clip_components.pt")

        print(f"Checkpoint saved to {path}/")

    def load(self, path: str):
        """Load a saved checkpoint back into this model instance."""
        from .text_encoder import TextEncoder

        # Reload LoRA adapters onto a fresh quantized base
        self.text_encoder = TextEncoder.load(f"{path}/text_lora")

        # Load remaining components
        ckpt = torch.load(f"{path}/clip_components.pt", map_location="cpu")
        self.vision_encoder.load_state_dict(ckpt["vision_encoder"])
        self.img_proj.load_state_dict(ckpt["img_proj"])
        self.txt_proj.load_state_dict(ckpt["txt_proj"])
        self.temperature.data = ckpt["temperature"]

        print(f"Checkpoint loaded from {path}/")
