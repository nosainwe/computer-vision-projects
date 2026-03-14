"""
vision_encoder.py
-----------------
Vision Transformer (ViT) built from scratch.

Architecture follows Dosovitskiy et al. (2020):
  "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
  https://arxiv.org/abs/2010.11929

No pretrained weights — trained from random initialization on Flickr30k
via contrastive learning with the text encoder.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """
    Split image into non-overlapping patches and linearly embed each one.

    For a 224x224 image with patch_size=16:
        num_patches = (224/16)^2 = 196
        Each patch is 16*16*3 = 768-dimensional before projection.
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        # catching this early — silent failure here would produce wrong num_patches downstream
        assert img_size % patch_size == 0, \
            f"Image size {img_size} must be divisible by patch size {patch_size}"

        self.img_size    = img_size
        self.patch_size  = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # conv with kernel=stride=patch_size is equivalent to cutting the image into patches
        # and projecting each one — cleaner than doing it with reshapes and einops
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.projection(x)    # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)          # (B, embed_dim, num_patches) — collapsing spatial dims
        x = x.transpose(1, 2)     # (B, num_patches, embed_dim) — seq-first for transformer
        return x


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):

    def __init__(self, embed_dim: int = 768, num_heads: int = 12,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5    # 1/sqrt(d_k) — prevents softmax from saturating

        # fusing Q, K, V into one linear is faster than three separate ones
        self.qkv       = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # compute Q, K, V in one matmul then split — avoids three separate forward passes
        qkv = self.qkv(x)                                       # (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                       # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)                                 # each: (B, heads, N, head_dim)

        # scaled dot-product — scale before softmax to keep gradients healthy
        attn = (q @ k.transpose(-2, -1)) * self.scale           # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v)                                          # (B, heads, N, head_dim)
        x = x.transpose(1, 2).reshape(B, N, C)                 # merge heads back into C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    Standard ViT Transformer block:
        LayerNorm -> MHSA -> residual
        LayerNorm -> MLP  -> residual
    """

    def __init__(self, embed_dim: int = 768, num_heads: int = 12,
                 mlp_ratio: float = 4.0, drop: float = 0.0, attn_drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadSelfAttention(embed_dim, num_heads, attn_drop, drop)
        self.norm2 = nn.LayerNorm(embed_dim)

        # mlp_ratio=4 means the hidden layer is 4x wider than embed_dim — standard ViT config
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pre-norm formulation (norm before sublayer, not after) — more stable than post-norm
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Vision Transformer (ViT)
# ---------------------------------------------------------------------------

class VisionTransformer(nn.Module):
    """
    ViT-Base/16 — trained from scratch.

    Default config matches ViT-B/16:
        img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12

    The [CLS] token representation at the final layer is used as the
    image embedding, following the original paper.
    """

    def __init__(
        self,
        img_size: int         = 224,
        patch_size: int       = 16,
        in_channels: int      = 3,
        embed_dim: int        = 768,
        depth: int            = 12,
        num_heads: int        = 12,
        mlp_ratio: float      = 4.0,
        drop_rate: float      = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches      = self.patch_embed.num_patches

        # cls_token is a learnable parameter — gets prepended to every sequence
        # the model learns to aggregate global info into it through attention
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # learned positional encoding — +1 for the cls token slot
        # fixed sinusoidal would also work but learned is simpler to implement
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop  = nn.Dropout(drop_rate)

        # stacking blocks as nn.Sequential so forward just calls them in order
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop_rate, attn_drop_rate)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        # trunc_normal with std=0.02 is the standard ViT init — don't deviate from this
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                # norm weight=1, bias=0 is the correct starting point — don't randomize these
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                # fan_out init for the patch projection conv
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224) image tensor

        Returns:
            cls_embedding: (B, embed_dim) -- the [CLS] token representation
        """
        B = x.shape[0]

        x = self.patch_embed(x)                  # (B, 196, 768) — 196 patches for 224x224

        # expand cls_token across the batch — can't just broadcast, need explicit expand
        cls = self.cls_token.expand(B, -1, -1)   # (B, 1, 768)
        x   = torch.cat([cls, x], dim=1)         # (B, 197, 768) — cls goes first

        # adding positional encoding after concatenation so cls gets its own position slot
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.blocks(x)
        x = self.norm(x)

        # only returning the cls token — this is the whole image's representation
        # the rest are patch tokens, useful for dense tasks but not needed here
        return x[:, 0]                           # (B, 768)
