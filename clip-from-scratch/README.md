# CLIP from Scratch 🔗

> A ground-up implementation of Contrastive Language–Image Pretraining (CLIP) — no OpenAI weights, no HuggingFace CLIP wrappers. Every component hand-rolled and understood.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Dataset: Flickr30k](https://img.shields.io/badge/Dataset-Flickr30k-yellow)](https://shannon.cs.illinois.edu/DenotationGraph/)

---

## What This Is

This project is a **personal learning implementation** of the CLIP architecture from the original paper:
> Radford et al., *"Learning Transferable Visual Models From Natural Language Supervision"*, OpenAI 2021.

The goal was not to replicate OpenAI's scale, but to deeply understand every design decision by building it myself — from patch embeddings to contrastive loss to QLoRA serialization quirks.

**Trained on Flickr30k image-caption pairs. After 20 epochs, the model correctly identifies the right caption for an image, ranking it above unrelated distractors.**

---

## Architecture

```
  IMAGE ──► ViT Encoder ──► Projection Head (→ 512d) ──┐
                                                         ├──► Cosine Similarity ──► InfoNCE Loss
  TEXT  ──► LM Encoder  ──► Projection Head (→ 512d) ──┘
             (QLoRA 4-bit)
```

### Vision Encoder — Vision Transformer (ViT)
- Patch size: 16×16, image size: 224×224 → **196 patch tokens**
- 12 Transformer blocks, hidden dim 768, 12 attention heads
- Trained **from random init** (no ImageNet pretraining)
- CLS token pooled → 768-dim representation → projected to 512-dim

### Text Encoder — `sihab/slm-1.0` (1.5B causal LM)
- 1.5B parameter causal language model
- Quantized to **4-bit with bitsandbytes** (NF4)
- Fine-tuned with **QLoRA** (rank=16, α=32, dropout=0.05)
- Last-token pooled → projected to 512-dim

### Shared Embedding Space
- Both modalities projected into a **512-dimensional unit hypersphere**
- L2 normalization applied before similarity computation
- Learned temperature parameter `τ` (initialized at `log(1/0.07)`)

### Loss — Symmetric InfoNCE
```
ℒ = ( CE(logits, labels) + CE(logits.T, labels) ) / 2
```
Where `logits = (img_emb @ text_emb.T) / exp(τ)`

---

## Results

After 20 epochs on Flickr30k:

| Query Image | Matched Caption | Similarity |
|---|---|---|
| 🐕 Golden retriever in snow | *"A golden retriever catches a tennis ball in the snow"* | **0.91** |
| ← same image | *"A man riding a bicycle"* | 0.23 |
| ← same image | *"A cat sleeping on a couch"* | 0.18 |

The model learns a strong alignment between visual and textual representations with no pretrained vision backbone.

---

## Project Structure

```
clip-from-scratch/
│
├── src/
│   ├── model.py          # Full CLIP model (encoders + projection heads)
│   ├── vision_encoder.py # Vision Transformer (ViT) from scratch
│   ├── text_encoder.py   # QLoRA-wrapped causal LM encoder
│   ├── loss.py           # Symmetric InfoNCE contrastive loss
│   ├── dataset.py        # Flickr30k dataset loader & preprocessing
│   └── utils.py          # Checkpointing, logging, evaluation helpers
│
├── train.py              # Main training script
├── evaluate.py           # Image-to-text & text-to-image retrieval eval
├── inference.py          # Run retrieval on custom image/caption pairs
│
├── data/
│   └── README.md         # How to download and set up Flickr30k
│
├── checkpoints/          # Saved model weights (gitignored)
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/<your-username>/clip-from-scratch.git
cd clip-from-scratch

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **GPU Note:** A CUDA-capable GPU with at least 12GB VRAM is recommended. The 4-bit quantized text encoder reduces memory significantly compared to full-precision.

---

## Dataset Setup

This project uses **Flickr30k** (~31,000 images, 5 captions each).

```bash
# Download instructions in data/README.md
# After download, your data/ directory should look like:
data/
├── flickr30k_images/   # ~31k .jpg files
└── results.csv         # image_name | comment_number | comment
```

See [`data/README.md`](data/README.md) for full download instructions.

---

## Training

```bash
python train.py \
  --data_dir ./data \
  --epochs 20 \
  --batch_size 64 \
  --lr 1e-4 \
  --save_dir ./checkpoints
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 20 | Number of training epochs |
| `--batch_size` | 64 | Batch size (reduce if OOM) |
| `--lr` | 1e-4 | Learning rate |
| `--lora_rank` | 16 | QLoRA adapter rank |
| `--embed_dim` | 512 | Shared embedding dimension |
| `--temperature` | 0.07 | Initial temperature (learned) |

---

## Evaluation

```bash
python evaluate.py --checkpoint ./checkpoints/epoch_20
```

Runs image→text and text→image retrieval on the Flickr30k test split and reports **Recall@1, Recall@5, Recall@10**.

---

## Inference

```bash
python inference.py \
  --image path/to/your/image.jpg \
  --captions "A dog playing fetch" "A cat on a sofa" "A person cycling" \
  --checkpoint ./checkpoints/epoch_20
```

---

## Checkpointing — A Key Lesson

> ⚠️ **You cannot `torch.save()` a QLoRA model directly.** The 4-bit quantized base weights don't serialize cleanly.

The correct approach is to save the LoRA adapters and non-quantized heads separately:

```python
# Saving
model.text_encoder.lora_adapters.save_pretrained(path + "/lora")
torch.save({
    "vision": model.vision_encoder.state_dict(),
    "temperature": model.temperature,
    "img_proj": model.img_proj.state_dict(),
    "txt_proj": model.txt_proj.state_dict(),
}, path + "/clip_heads.pt")

# Loading
base = AutoModelForCausalLM.from_pretrained("sihab/slm-1.0", load_in_4bit=True)
model.text_encoder = PeftModel.from_pretrained(base, path + "/lora")
```

---

## Key Learnings

### 1. QLoRA Serialization
`torch.save()` on a quantized model fails silently or produces unloadable checkpoints. Always save LoRA adapters via `save_pretrained()` and reload on top of a freshly quantized base.

### 2. Temperature Scaling
The temperature `τ` in InfoNCE loss matters enormously:
- **Too high** → model can't separate hard negatives → loss plateaus
- **Too low** → gradients vanish → training stalls
- **Solution:** make `τ` a learned `nn.Parameter`, let it find its own optimum

### 3. CLIP Is a Ranker, Not a Generator
CLIP learns to score (image, text) pairs — it has no decoder and generates nothing. To go from *"which caption fits this image?"* to *"describe this image,"* you need a generative head. This is exactly what **LLaVA** does with its MLP projection bridge into a decoder LM.

---

## Limitations

- Trained only on Flickr30k (31k images) — OpenAI's CLIP used 400M image-text pairs
- ViT trained from scratch; production CLIP uses pretrained vision backbones
- No hard-negative mining; random in-batch negatives only
- Text encoder is a causal LM (decoder), not a bidirectional encoder — last-token pooling is a reasonable but imperfect approximation

---

## Next Steps

- [ ] Add **projection bridge** from ViT CLS token → causal LM input space (mini-LLaVA)
- [ ] Enable **image captioning** via autoregressive decoding
- [ ] Experiment with **hard negative mining** in the contrastive loss
- [ ] Try a **bidirectional text encoder** for better pooling

---

## Acknowledgements

This project would not have been possible without the following:

| Resource | Contribution |
|---|---|
| [Radford et al. (2021)](https://arxiv.org/abs/2103.00020) | Original CLIP paper — the architecture and InfoNCE loss formulation |
| [Dosovitskiy et al. (2020)](https://arxiv.org/abs/2010.11929) | *An Image is Worth 16×16 Words* — ViT architecture |
| [Hu et al. (2021)](https://arxiv.org/abs/2106.09685) | LoRA paper — low-rank adapter methodology |
| [Dettmers et al. (2023)](https://arxiv.org/abs/2305.14314) | QLoRA paper — 4-bit quantization + LoRA |
| [`sihab/slm-1.0`](https://huggingface.co/sihab/slm-1.0) | 1.5B causal LM used as the text encoder backbone |
| [Flickr30k Dataset](https://shannon.cs.illinois.edu/DenotationGraph/) | Young et al. (2014) — image-caption training data |
| [HuggingFace `transformers`](https://github.com/huggingface/transformers) | Tokenizer, model loading utilities |
| [HuggingFace `peft`](https://github.com/huggingface/peft) | QLoRA / PEFT adapter framework |
| [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) | 4-bit quantization backend |
| [OpenCLIP](https://github.com/mlfoundations/open_clip) | Reference implementation for architecture patterns |

> This is an **independent learning project** with no affiliation with OpenAI, HuggingFace, or the authors of the above papers.

---

## Citation

If you find this useful as a reference implementation:

```bibtex
@misc{clip-from-scratch,
  author       = {sihab},
  title        = {CLIP from Scratch: A Ground-Up Contrastive Learning Implementation},
  year         = {2025},
  howpublished = {\url{https://github.com/<your-username>/clip-from-scratch}}
}
```

---

## License

MIT — see [LICENSE](LICENSE) for details.
