import torch
import clip
import numpy as np
from PIL import Image
import cv2

# Decide device (CPU or GPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model
model, preprocess = clip.load("ViT-B/32", device=DEVICE)
model.eval()

def embed_text(text: str) -> np.ndarray:
    """
    Convert text into a 512-dimension embedding vector using CLIP
    """
    with torch.no_grad():
        tokens = clip.tokenize([text]).to(DEVICE)
        embedding = model.encode_text(tokens)

        # Normalize the vector
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

    return embedding.cpu().numpy()[0]

def embed_image(image_path: str) -> np.ndarray:
    """
    Convert an image into a 512-dimension embedding vector using CLIP
    """
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        embedding = model.encode_image(image_input)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

    return embedding.cpu().numpy()[0]

def embed_video(video_path: str, frame_interval: int = 30) -> np.ndarray:
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
            # Convert BGR (OpenCV) to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            image_input = preprocess(image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                emb = model.encode_image(image_input)
                emb = emb / emb.norm(dim=-1, keepdim=True)

            frame_embeddings.append(emb.cpu().numpy()[0])

        frame_count += 1

    cap.release()

    if len(frame_embeddings) == 0:
        raise ValueError("No frames extracted from video")

    # Average all frame embeddings
    video_embedding = np.mean(frame_embeddings, axis=0)

    # Normalize final vector
    video_embedding = video_embedding / np.linalg.norm(video_embedding)

    return video_embedding

def embed_multimodal(image_path: str, text: str) -> np.ndarray:
    """
    Combine image and text embeddings into a single normalized vector.
    Used for posts that have both image and caption.
    """

    img_vec = embed_image(image_path)
    text_vec = embed_text(text)

    combined = (img_vec + text_vec) / 2
    combined = combined / np.linalg.norm(combined)

    return combined
