"""
loss.py
-------
Symmetric InfoNCE (cross-modal contrastive) loss.

The loss was introduced in:
  van den Oord et al. (2018) "Representation Learning with Contrastive
  Predictive Coding" https://arxiv.org/abs/1807.03748

And applied to vision-language contrastive learning in:
  Radford et al. (2021) CLIP https://arxiv.org/abs/2103.00020

How it works:
  Given a batch of B (image, caption) pairs, we construct a B×B matrix
  of cosine similarities. The ground-truth pairs are on the diagonal.
  The loss is the mean of two cross-entropy losses:
    - image-to-text: for each image, the correct caption is the target
    - text-to-image: for each caption, the correct image is the target

  In-batch negatives: all other captions in the batch serve as
  negatives for each image, and vice versa. This means larger
  batch sizes → more negatives → harder problem → better representations.

Temperature τ:
  - High τ (e.g. 1.0): soft distribution, easy negatives, slow learning
  - Low τ (e.g. 0.01): sharp distribution, hard negatives, unstable gradients
  - We learn τ as a parameter (stored as log(τ) for numerical stability)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPLoss(nn.Module):
    """
    Symmetric InfoNCE loss for contrastive image-text training.

    The logits matrix is computed OUTSIDE this module (in CLIPModel.forward)
    so that the temperature is applied there and remains a proper
    nn.Parameter that receives gradients.

    Args:
        None

    Forward:
        logits (torch.Tensor): (B, B) scaled cosine similarity matrix
                               Already divided by τ.

    Returns:
        loss (torch.Tensor): scalar mean of i2t and t2i cross-entropy losses
    """

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        B = logits.size(0)
        labels = torch.arange(B, device=logits.device)

        # Image-to-text: for each image row, the diagonal is the positive
        loss_i2t = F.cross_entropy(logits, labels)

        # Text-to-image: for each text column, the diagonal is the positive
        loss_t2i = F.cross_entropy(logits.T, labels)

        return (loss_i2t + loss_t2i) / 2


def contrastive_accuracy(logits: torch.Tensor) -> dict:
    """
    Compute in-batch retrieval accuracy for logging.

    Returns accuracy for both directions (i→t and t→i).
    Not a training objective — just a useful diagnostic metric.
    """
    B = logits.size(0)
    labels = torch.arange(B, device=logits.device)

    i2t_preds = logits.argmax(dim=1)
    t2i_preds = logits.argmax(dim=0)

    return {
        "acc_i2t": (i2t_preds == labels).float().mean().item(),
        "acc_t2i": (t2i_preds == labels).float().mean().item(),
    }
