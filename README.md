# Xoodrip Content Intelligence Service

This repository contains a **multimodal content intelligence microservice** for Xoodrip.
It dynamically categorizes posts (text, images, videos) without predefined categories
using online clustering, zero-shot domain classification, and exposes the functionality via secured APIs.

---

## 🚀 Key Features

- **Dynamic Clustering:** Automatically creates new categories or updates existing ones using incremental mean clustering.
- **Multimodal Support:** Embeds text, images, and videos (via frame sampling) into a unified 512-dimensional semantic space.
- **Zero-Shot Domain Classification:** Uses OpenAI's **CLIP** model to classify content into 12 distinct domains (sports, bollywood, politics, tech, etc.) without training a custom classifier.
- **Domain-Aware Gating:** Prevents posts from different domains (e.g., cricket and a tech startup) from ever merging into the same category, even if semantically similar.
- **Persistence:** All learned categories and centroids are saved in a SQLite database (`xoodrip.db`), surviving server restarts.
- **Auto-Naming:** Generates human-readable category names automatically using TF-IDF term extraction once a category has 3+ posts.
- **Secure:** API key authentication for secure backend-to-backend communication.

---

## 🏗️ Architecture Overview

The system is built as a plug-and-play microservice:

```text
Xoodrip Backend  
  │
  ├── 1. POST /analyze/[text|image|video] (Protected by X-Api-Key)
  │
Content Intelligence Service
  │
  ├── 2. Embeddings (CLIP ViT-B/32) -> 512-dim vector
  ├── 3. Domain Inference (Zero-shot CLIP comparison)
  ├── 4. Online Clustering (Find nearest DB centroid or create new)
  │
  └── 5. Returns: { category_id, name, domain, similarity, is_new }
```

---

## 🧰 Tech Stack

- **Framework:** FastAPI, Uvicorn
- **Machine Learning:** PyTorch, OpenAI CLIP, scikit-learn
- **Database:** SQLite (via SQLAlchemy ORM)
- **Media Processing:** OpenCV (video frames), Pillow (images)

---

## ⚙️ Setup & Installation

### 1. Prerequisites
- Python 3.12+
- (Optional but recommended) A dedicated virtual environment

### 2. Install Dependencies
Because OpenAI's CLIP model is not hosted on PyPI, you must install it directly from GitHub, followed by the rest of the dependencies:

```bash
# 1. Install CLIP directly from GitHub
pip install git+https://github.com/openai/CLIP.git

# 2. Install all other pinned dependencies
pip install -r requirements.txt
```

---

## 🚀 Running the Service

Start the FastAPI development server:

```bash
uvicorn app.main:app --reload
```

The service will run at `http://localhost:8000`.
- **Health Check:** `GET /health`
- **Interactive API Docs (Swagger UI):** `http://localhost:8000/docs`

### Authentication
All `/analyze` endpoints require an API key passed in the headers.
For local development, the fallback key is `dev-secret-key`. In production, set the environment variable:
```bash
export XOODRIP_API_KEY="your-secure-production-key"
```

---

## 🔌 API Endpoints

All endpoints are `POST` and accept form data (`multipart/form-data` or `application/x-www-form-urlencoded`).

### Headers Required (All Endpoints)
```http
X-Api-Key: dev-secret-key
```

### 1. Analyze Text
`POST /analyze/text`

**Form Data:**
- `text` (string, required): The content of the post.
- `include_scores` (boolean, optional): Set to `true` to get raw CLIP confidence scores for all domains.

### 2. Analyze Image
`POST /analyze/image`

**Form Data:**
- `image` (file, required): The image file to upload.
- `caption` (string, optional): Relevant text. If provided, the system uses a multimodal embedding (average of image + text).
- `include_scores` (boolean, optional)

### 3. Analyze Video
`POST /analyze/video`

**Form Data:**
- `video` (file, required): The video file (e.g., .mp4).
- `caption` (string, optional): Relevant text.
- `include_scores` (boolean, optional)

### Example Response:
```json
{
  "category_id": 4,
  "is_new": false,
  "similarity": 0.82,
  "domain": "sports",
  "name": "cricket & ipl"
}
```

---

## 🧪 Testing

The project uses `pytest` for rigorous integration testing (verifying domain inference and score thresholds without requiring manual API calls).

To run the test suite:
```bash
python -m pytest tests/ -v
```

*Note: The first run will download the CLIP model weights (~350MB) if they are not already cached.*
