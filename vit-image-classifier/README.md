# 🔍 ViT Vision - Image Classifier

A production-ready, zero-dependency web app that classifies images using **Google's Vision Transformer (ViT)** model via the HuggingFace Inference API.

> Built to make the theory from [Vision Transformers](https://vizuara.substack.com) tangible and interactive.

![Demo screenshot](screenshot.png)

---

## ✨ Features

- **Drag & drop** any image to classify
- **Paste** an image from your clipboard (Ctrl+V / Cmd+V)
- **Camera capture** - take a photo and classify it instantly
- **Sample images** - one-click demo with curated examples
- **ViT Patch Grid toggle** - visualize how ViT slices your image into 14×14 patches
- **Top-5 predictions** with animated confidence bars
- Zero build step - pure HTML/CSS/JS, open `index.html` and go

---

## 🧠 The Model

| Property | Value |
|---|---|
| Model | `google/vit-base-patch16-224` |
| Pretrained on | ImageNet-21k |
| Fine-tuned on | ImageNet-1k |
| Input size | 224×224 px |
| Patch size | 16×16 px |
| Patches per image | 196 (14×14 grid) + 1 CLS token |
| Embedding dim | 768 |
| Attention heads | 12 |
| Encoder layers | 12 |
| Parameters | ~86M |
| Output classes | 1,000 (ImageNet categories) |

---

## 🚀 Quick Start

### 1. Get a free HuggingFace token

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Click **New token** → choose **Read** scope → copy it

### 2. Open the app

```bash
git clone https://github.com/YOUR_USERNAME/vit-image-classifier.git
cd vit-image-classifier
open index.html   # Mac
# or just double-click index.html in Windows/Linux
```

> **No server required.** This is a static HTML file.

### 3. Paste your token

Enter your `hf_...` token in the token field at the top of the app, then drop an image to classify.

---

## 🌐 Deploy (optional)

Since this is a single HTML file, you can host it anywhere static:

**GitHub Pages:**
```bash
# Push to a repo, then go to:
# Settings → Pages → Source: main branch / root
```

**Netlify** (drag & drop):
- Go to [netlify.com/drop](https://app.netlify.com/drop)
- Drag the `index.html` file onto the page

**Vercel:**
```bash
npx vercel
```

---

## 🏗️ How It Works (ViT Explained)

### From pixels to patches

ViT treats an image the same way a language model treats text - as a **sequence of tokens**.

1. **Divide** the image into a 14×14 grid of non-overlapping 16×16 pixel patches → 196 patches
2. **Flatten** each patch into a vector of length `16×16×3 = 768`
3. **Project** each flattened patch to embedding dimension D=768
4. **Prepend** a special learnable `[CLS]` token → sequence length = 197
5. **Add positional embeddings** so the model knows where each patch lives
6. **Feed through 12 Transformer encoder blocks** - each patch attends to every other
7. **Read the `[CLS]` token output** and pass it to an MLP head → class logits

### Why self-attention beats convolutions for this

A CNN's receptive field grows only as layers stack - distant parts of an image only "meet" deep in the network.

ViT's self-attention is **global from layer 1**: the patch showing a cat's eye can directly attend to the patch showing its whiskers, even on the first layer. This makes ViT especially powerful for:
- Long-range dependencies in images
- Recognition where context matters (e.g. distinguishing a wolf from a husky by background snow)
- Transfer learning from massive datasets

```
Image (224×224×3)
    ↓  slice into patches
196 patches × (16×16×3)
    ↓  flatten + linear projection
196 × 768 patch embeddings
    ↓  prepend CLS token
197 × 768 sequence
    ↓  + positional embeddings
197 × 768 (with position info)
    ↓  12× Transformer Encoder Block
        ├─ Multi-Head Self-Attention (12 heads)
        ├─ Add & LayerNorm
        ├─ Feed-Forward MLP
        └─ Add & LayerNorm
    ↓  CLS token final state (768-dim)
    ↓  MLP head
1000 class logits → softmax → top-5 predictions
```

---

## 📁 Project Structure

```
vit-image-classifier/
├── index.html      ← The entire app (HTML + CSS + JS)
└── README.md       ← This file
```

---

## 🔧 Customization

### Use a different model

Change the `MODEL` constant in the `<script>` block:

```js
// Options:
const MODEL = 'google/vit-base-patch16-224';       // default
const MODEL = 'google/vit-large-patch16-224';      // larger, more accurate
const MODEL = 'microsoft/resnet-50';               // CNN baseline to compare
const MODEL = 'facebook/deit-base-patch16-224';    // Data-efficient ViT
const MODEL = 'openai/clip-vit-base-patch32';      // CLIP (zero-shot)
```

### Add more sample images

In the `.sample-grid` section, add:
```html
<div class="sample-thumb" onclick="loadSample('IMAGE_URL', 'Label')">🖼️</div>
```

### Run your own inference server

To avoid the HuggingFace token requirement, run a local inference server:

```bash
pip install transformers torch pillow flask
```

```python
# server.py
from flask import Flask, request, jsonify
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch, io

app = Flask(__name__)
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
model.eval()

@app.route('/classify', methods=['POST'])
def classify():
    img = Image.open(io.BytesIO(request.data)).convert('RGB')
    inputs = processor(images=img, return_tensors='pt')
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    top5 = probs.topk(5)
    return jsonify([
        {'label': model.config.id2label[i.item()], 'score': s.item()}
        for i, s in zip(top5.indices, top5.values)
    ])

app.run(port=8000)
```

Then change `API_URL` in `index.html` to `http://localhost:8000/classify`.

---

## 📚 Learn More

- [Vision Transformer paper (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929)
- [HuggingFace model card](https://huggingface.co/google/vit-base-patch16-224)
- [Vizuara - Vision Transformers article](https://vizuara.substack.com)
- [An Image is Worth 16×16 Words (illustrated)](https://arxiv.org/abs/2010.11929)

---

## 📜 License

MIT - free to use, modify, and deploy. Attribution appreciated.

---

*Built as a hands-on companion to the Vision Transformers theory.*
