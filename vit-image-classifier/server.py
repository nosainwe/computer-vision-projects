"""
Local ViT Inference Server
--------------------------
Run this to classify images locally without a HuggingFace API token.
The model (~350MB) is downloaded once and cached.

Requirements:
    pip install transformers torch pillow flask flask-cors

Usage:
    python server.py

Then open index.html and set the API URL to http://localhost:8000/classify
(edit the API_URL constant in the <script> block of index.html)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch
import io
import sys

app = Flask(__name__)
CORS(app)  # allow requests from the browser

MODEL_NAME = "google/vit-base-patch16-224"

print(f"Loading model: {MODEL_NAME}")
print("(This may take a moment on first run — model is ~350MB)")

try:
    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    model = ViTForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    sys.exit(1)


@app.route("/classify", methods=["POST", "OPTIONS"])
def classify():
    if request.method == "OPTIONS":
        return "", 200

    try:
        image_bytes = request.data
        if not image_bytes:
            return jsonify({"error": "No image data received"}), 400

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits

        probs = torch.softmax(logits, dim=-1)[0]
        top_k = probs.topk(5)

        results = [
            {
                "label": model.config.id2label[idx.item()],
                "score": round(score.item(), 6),
            }
            for idx, score in zip(top_k.indices, top_k.values)
        ]

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME})


if __name__ == "__main__":
    print("\n🚀 Starting local inference server at http://localhost:8000")
    print("   Update API_URL in index.html to: http://localhost:8000/classify\n")
    app.run(host="0.0.0.0", port=8000, debug=False)
