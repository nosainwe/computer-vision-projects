# CLIP from Scratch 🔗

> A ground-up implementation of Contrastive Language-Image Pretraining (CLIP), built without OpenAI weights or HuggingFace CLIP wrappers.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Dataset: Flickr30k](https://img.shields.io/badge/Dataset-Flickr30k-yellow)](https://shannon.cs.illinois.edu/DenotationGraph/)

---

## Overview

This project is a personal learning implementation of the CLIP architecture from:

> Radford et al., *"Learning Transferable Visual Models From Natural Language Supervision"*, OpenAI, 2021

The aim was not to match OpenAI's scale. The aim was to understand the full pipeline by building it myself: patch embeddings, transformer encoders, contrastive training, temperature scaling, and QLoRA checkpointing.

Trained on Flickr30k image-caption pairs, the model learns to rank the correct caption above unrelated distractors after 20 epochs.

---

## Architecture

```text
  IMAGE -> ViT Encoder -> Projection Head (512d) ->|
                                                   |-> Cosine Similarity -> InfoNCE Loss
  TEXT  -> LM Encoder  -> Projection Head (512d) ->|
           (QLoRA 4-bit)
```

### Vision Encoder: Vision Transformer (ViT)

- Patch size: `16 x 16`
- Input image size: `224 x 224`
- Total patch tokens: `196`
- Transformer blocks: `12`
- Hidden dimension: `768`
- Attention heads: `12`
- Trained from random initialisation, with no ImageNet pretraining
- CLS token output pooled and projected from `768 -> 512`

### Text Encoder: `sihab/slm-1.0`

- 1.5B parameter causal language model
- Quantised to 4-bit using `bitsandbytes` with NF4
- Fine-tuned with QLoRA:
  - rank = `16`
  - alpha = `32`
  - dropout = `0.05`
- Last-token representation pooled and projected from LM hidden size to `512`

### Shared Embedding Space

- Image and text embeddings are projected into the same `512-dimensional` space
- L2 normalisation is applied before similarity computation
- A learned temperature parameter `τ` scales the logits
- Initial temperature setup: `log(1 / 0.07)`

### Loss Function: Symmetric InfoNCE

```text
L = ( CE(logits, labels) + CE(logits.T, labels) ) / 2
```

Where:

```text
logits = (img_emb @ text_emb.T) / exp(τ)
```

This trains the model in both directions:

- image -> text
- text -> image

---

## Why this project exists

Most CLIP repos hide the moving parts behind wrappers. That’s fine if your only goal is to run inference. It’s bad if you actually want to understand what the model is doing.

I built this to force myself through the full stack:

- how images become patch tokens
- how text gets pooled into a single embedding
- how contrastive loss works in practice
- why temperature matters
- why QLoRA saving and loading can go wrong

---

## Results

After 20 epochs on Flickr30k:

| Query Image | Candidate Caption | Similarity |
|---|---|---:|
| 🐕 Golden retriever in snow | *"A golden retriever catches a tennis ball in the snow"* | **0.91** |
| same image | *"A man riding a bicycle"* | 0.23 |
| same image | *"A cat sleeping on a couch"* | 0.18 |

The model learns a useful shared embedding space even though the vision encoder starts from random initialisation.

---

## Project Structure

```text
clip-from-scratch/
│
├── src/
│   ├── model.py
│   ├── vision_encoder.py
│   ├── text_encoder.py
│   ├── loss.py
│   ├── dataset.py
│   └── utils.py
│
├── train.py
├── evaluate.py
├── inference.py
│
├── data/
│   └── README.md
│
├── checkpoints/
├── requirements.txt
└── README.md
```

### File Breakdown

- `model.py`  
  Full CLIP model definition, including encoders and projection heads.

- `vision_encoder.py`  
  Vision Transformer implementation for image encoding.

- `text_encoder.py`  
  QLoRA-wrapped causal language model used as the text encoder.

- `loss.py`  
  Symmetric InfoNCE contrastive loss.

- `dataset.py`  
  Flickr30k loading, image preprocessing, and caption tokenisation.

- `utils.py`  
  Logging, checkpoint helpers, and evaluation support.

- `train.py`  
  Main training entry point.

- `evaluate.py`  
  Retrieval evaluation for image-to-text and text-to-image matching.

- `inference.py`  
  Runs retrieval on custom image and caption inputs.

---

## Installation

```bash
git clone https://github.com/<your-username>/clip-from-scratch.git
cd clip-from-scratch

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Hardware Note

A CUDA-capable GPU with at least 12 GB VRAM is recommended.

The 4-bit quantised text encoder cuts memory use hard compared to full precision, but this is still not a tiny project.

---

## Dataset Setup

This project uses **Flickr30k**, which contains roughly 31,000 images and 5 captions per image.

Expected layout:

```text
data/
├── flickr30k_images/
└── results.csv
```

See `data/README.md` for download and setup instructions.

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

### Key Arguments

| Argument | Default | Description |
|---|---:|---|
| `--epochs` | 20 | Number of training epochs |
| `--batch_size` | 64 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--lora_rank` | 16 | QLoRA adapter rank |
| `--embed_dim` | 512 | Shared embedding dimension |
| `--temperature` | 0.07 | Initial temperature value |

If you hit out-of-memory errors, reduce batch size first.

---

## Evaluation

```bash
python evaluate.py --checkpoint ./checkpoints/epoch_20
```

This runs retrieval evaluation on the Flickr30k test split and reports:

- Recall@1
- Recall@5
- Recall@10

The point is simple: does the model rank the right caption or image near the top?

---

## Inference

```bash
python inference.py \
  --image path/to/your/image.jpg \
  --captions "A dog playing fetch" "A cat on a sofa" "A person cycling" \
  --checkpoint ./checkpoints/epoch_20
```

This lets you test retrieval on your own image-caption candidates.

---

## Checkpointing: the annoying part that matters

A normal `torch.save()` workflow is not enough here.

The 4-bit quantised base model does not serialize cleanly if you try to save the full wrapped model directly. That can leave you with checkpoints that either fail silently or cannot reload properly.

### Correct Saving Approach

```python
model.text_encoder.lora_adapters.save_pretrained(path + "/lora")

torch.save({
    "vision": model.vision_encoder.state_dict(),
    "temperature": model.temperature,
    "img_proj": model.img_proj.state_dict(),
    "txt_proj": model.txt_proj.state_dict(),
}, path + "/clip_heads.pt")
```

### Correct Loading Approach

```python
base = AutoModelForCausalLM.from_pretrained(
    "sihab/slm-1.0",
    load_in_4bit=True
)

model.text_encoder = PeftModel.from_pretrained(base, path + "/lora")
```

This is one of the biggest practical lessons in the project.

---

## Key Learnings

### 1. QLoRA checkpointing is not plug-and-play

Quantised models are awkward to save and reload. If you treat them like ordinary PyTorch modules, you can waste hours.

The safe route is:

- save LoRA adapters with `save_pretrained()`
- reload adapters on top of a fresh quantised base model
- save non-quantised heads separately

### 2. Temperature scaling matters more than it looks

The temperature parameter `τ` changes how sharply the model separates positive and negative pairs.

- Too high: the logits get too soft and the model struggles to separate hard negatives
- Too low: the logits get too sharp and optimisation can stall

The best fix was to make `τ` a learned `nn.Parameter` and let training adjust it.

### 3. CLIP is a ranker, not a generator

CLIP does not generate captions.

It learns to score image-text pairs. That means it can answer:

- which caption matches this image?
- which image matches this text?

It cannot answer:

- describe this image in a new sentence

If you want generation, you need a decoder-style setup, such as the kind of bridge used in models like LLaVA.

### 4. A causal LM can work as text encoder, but it is not ideal

This project uses last-token pooling from a decoder-style language model. It works, but it is not the same as using a bidirectional text encoder built for representation learning.

That trade-off is worth understanding, not hiding.

---

## Limitations

This repo is useful, but it is not pretending to be production CLIP.

- Trained only on Flickr30k, around 31k images
- OpenAI CLIP used hundreds of millions of image-text pairs
- Vision encoder trained from scratch instead of starting from strong visual pretraining
- No hard-negative mining
- In-batch negatives only
- Text encoder is causal, not bidirectional
- Last-token pooling is workable, but imperfect

That doesn’t kill the project. It just sets the right expectations.

---

## Next Steps

- [ ] Add a projection bridge from ViT CLS token to causal LM input space
- [ ] Extend the setup toward mini-LLaVA style captioning
- [ ] Try hard negative mining
- [ ] Test a bidirectional text encoder for better pooling
- [ ] Add cleaner retrieval metrics reporting and example visualisations

---

## Acknowledgements

| Resource | Contribution |
|---|---|
| [Radford et al. (2021)](https://arxiv.org/abs/2103.00020) | Original CLIP paper and contrastive setup |
| [Dosovitskiy et al. (2020)](https://arxiv.org/abs/2010.11929) | Vision Transformer architecture |
| [Hu et al. (2021)](https://arxiv.org/abs/2106.09685) | LoRA method |
| [Dettmers et al. (2023)](https://arxiv.org/abs/2305.14314) | QLoRA and 4-bit finetuning |
| [`sihab/slm-1.0`](https://huggingface.co/sihab/slm-1.0) | Text encoder backbone |
| [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/) | Image-caption dataset |
| [HuggingFace Transformers](https://github.com/huggingface/transformers) | Model and tokenizer utilities |
| [HuggingFace PEFT](https://github.com/huggingface/peft) | Adapter framework |
| [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) | 4-bit quantisation backend |
| [OpenCLIP](https://github.com/mlfoundations/open_clip) | Useful implementation reference points |

This is an independent learning project. It is not affiliated with OpenAI, HuggingFace, or the authors above.

---

## Citation

If you use this as a learning reference:

```bibtex
@misc{clip-from-scratch,
  author       = {sihab},
  title        = {CLIP from Scratch: A Ground-Up Contrastive Learning Implementation},
  year         = {2025},
  howpublished = {\url{https://github.com/<nosainwe>/clip-from-scratch}}
}
```

---

## License

MIT. See [LICENSE](LICENSE) for details.
