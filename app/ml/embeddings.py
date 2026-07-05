"""
embeddings.py
-------------
Multimodal embedding functions using Google SigLIP (so400m-patch14-384).

WHAT CHANGED vs. the old CLIP version:
--------------------------------------
Before (OpenAI CLIP ViT-B/32):
  - Library:    `clip` (installed from GitHub, fragile)
  - Model:      ViT-B/32 (2021, 88M params)
  - Embeddings: 512 dimensions
  - Tokenizer:  clip.tokenize()
  - Encoder:    model.encode_text() / model.encode_image()

After (Google SigLIP so400m-patch14-384):
  - Library:    `transformers` (standard Hugging Face, from PyPI)
  - Model:      SigLIP so400m (2024, 400M params — 4.5x bigger)
  - Embeddings: 1152 dimensions (higher = more information captured)
  - Processor:  AutoProcessor handles BOTH text tokenization AND image preprocessing
  - Encoder:    model.get_text_features() / model.get_image_features()

WHY SIGLIP IS BETTER:
  - Uses "sigmoid loss" instead of CLIP's "softmax loss" — more robust
  - Trained on billions of image-text pairs (vs CLIP's 400M)
  - ~75% zero-shot ImageNet accuracy (vs CLIP's ~63%)
  - Better at distinguishing similar domains (e.g. sports vs bollywood)
  - Installed cleanly from PyPI (no GitHub repo cloning)
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from transformers import AutoModel, AutoProcessor

# ── Device selection ──────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load SigLIP model ────────────────────────────────────────────────────────
# This downloads the model weights on first run (~1.6 GB) and caches them.
# Subsequent runs load instantly from the Hugging Face cache.
MODEL_NAME = "google/siglip-so400m-patch14-384"

print(f"[INFO] Loading SigLIP model ({MODEL_NAME}) on {DEVICE}...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
print(f"[SUCCESS] SigLIP model loaded — embedding dimension: 1152")


def embed_text(text: str) -> np.ndarray:
    """
    Convert text into a 1152-dimension embedding vector using SigLIP.

    Old CLIP way:
        tokens = clip.tokenize([text]).to(DEVICE)
        embedding = model.encode_text(tokens)

    New SigLIP way:
        inputs = processor(text=[text], ...)
        embedding = model.get_text_features(**inputs)
    """
    # `processor` handles tokenization (text → token IDs + attention mask)
    inputs = processor(
        text=[text],
        return_tensors="pt",       # return PyTorch tensors
        padding="max_length",      # pad to model's max length
        truncation=True,           # truncate if text is too long
    )

    # Move all tensors to the same device as the model
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        kwargs = {"input_ids": inputs["input_ids"]}
        if "attention_mask" in inputs:
            kwargs["attention_mask"] = inputs["attention_mask"]
            
        # get_text_features returns (1, 1152) tensor
        embedding = model.get_text_features(**kwargs)
        if hasattr(embedding, "pooler_output"):
            embedding = embedding.pooler_output
        elif not isinstance(embedding, torch.Tensor) and hasattr(embedding, "last_hidden_state"):
            embedding = embedding.last_hidden_state.mean(dim=1)
        
        # L2 normalize so dot product = cosine similarity
        embedding = F.normalize(embedding, p=2, dim=-1)

    return embedding.cpu().numpy()[0]                     # (1152,)


def embed_image(image_path: str) -> np.ndarray:
    """
    Convert an image into a 1152-dimension embedding vector using SigLIP.

    Old CLIP way:
        image = preprocess(Image.open(path)).unsqueeze(0).to(DEVICE)
        embedding = model.encode_image(image)

    New SigLIP way:
        inputs = processor(images=image, ...)
        embedding = model.get_image_features(**inputs)
    """
    image = Image.open(image_path).convert("RGB")

    # `processor` handles resizing to 384x384, normalization, etc.
    inputs = processor(
        images=image,
        return_tensors="pt",
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        embedding = model.get_image_features(
            pixel_values=inputs["pixel_values"],
        )
        embedding = F.normalize(embedding, p=2, dim=-1)

    return embedding.cpu().numpy()[0]                     # (1152,)


def embed_video(video_path: str, frame_interval: int = 30) -> np.ndarray:
    """
    Convert a video into a 1152-dimension embedding by averaging frame embeddings.

    Same logic as before — extract every Nth frame, embed each one, average.
    Only the image embedding function changed (CLIP → SigLIP).
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frame_embeddings = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Convert BGR (OpenCV) → RGB (PIL)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            # Use SigLIP's processor for image preprocessing
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            with torch.no_grad():
                emb = model.get_image_features(
                    pixel_values=inputs["pixel_values"],
                )
                emb = F.normalize(emb, p=2, dim=-1)

            frame_embeddings.append(emb.cpu().numpy()[0])

        frame_count += 1

    cap.release()

    if len(frame_embeddings) == 0:
        raise ValueError("No frames extracted from video")

    # Average all frame embeddings
    video_embedding = np.mean(frame_embeddings, axis=0)

    # Re-normalize the averaged vector
    video_embedding = video_embedding / np.linalg.norm(video_embedding)

    return video_embedding                                # (1152,)


def embed_multimodal(image_path: str, text: str) -> np.ndarray:
    """
    Combine image and text embeddings into a single normalized vector.
    Used for posts that have both image and caption.
    """
    img_vec = embed_image(image_path)
    text_vec = embed_text(text)

    combined = (img_vec + text_vec) / 2
    combined = combined / np.linalg.norm(combined)

    return combined                                       # (1152,)
