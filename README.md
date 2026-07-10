# Xoodrip Content Intelligence

A multimodal AI microservice that analyzes social media posts (text, images, videos) and automatically classifies them into domains and fine-grained categories using zero-shot learning.

Built as the content intelligence engine for the **GrowinBharat** platform.

## What It Does

1. **Domain Classification** — Assigns each post to one of 13 broad domains (sports, bollywood, politics, tech, etc.) using zero-shot SigLIP inference.
2. **Dynamic Categorization** — Groups similar posts into fine-grained categories within each domain using online cosine-similarity clustering.
3. **Auto-Naming** — Generates human-readable category names (e.g., "Cricket & Ipl", "Budget & Ministry") from accumulated post texts using TF-IDF.

> A single domain like **sports** can contain multiple categories — one for cricket, another for football, another for tennis — all discovered automatically from the content.

## Tech Stack

| Layer | Technology |
|---|---|
| **API Framework** | FastAPI (async) |
| **Server** | Uvicorn (ASGI) |
| **ML Model** | Google SigLIP `so400m-patch14-384` via Hugging Face Transformers |
| **ML Runtime** | PyTorch |
| **Embeddings** | 1152-dimensional L2-normalized vectors |
| **Database** | MongoDB Atlas (async via Motor) |
| **Config** | Pydantic Settings + `.env` |
| **Text Features** | scikit-learn TF-IDF |
| **Image Processing** | Pillow |
| **Video Processing** | OpenCV |
| **Testing** | pytest |
| **Language** | Python 3.12+ |

## Supported Domains

The zero-shot classifier covers 13 domains with no custom training data:

| Domain | Domain | Domain |
|---|---|---|
| 🏏 sports | 🎬 bollywood | 🗳️ politics |
| 🏛️ government | 💻 tech | 🚀 startup |
| 🍔 food | ✈️ travel | 💪 fitness |
| 👗 fashion | 😂 memes | 📺 tv_series |
| 📦 general | | |

## Architecture

```
GrowinBharat Backend
    │
    │  POST /analyze/text
    │  POST /analyze/image
    │  POST /analyze/video
    │  Header: X-Api-Key
    ▼
FastAPI Content Intelligence Service
    │
    │  1. Convert input → SigLIP embedding (1152-d vector)
    │  2. Classify domain via zero-shot prompt comparison
    │  3. Load category centroids from MongoDB
    │  4. Find nearest category (cosine similarity)
    │  5. Domain gating — block cross-domain merges
    │  6. Update existing category or create a new one
    │  7. Auto-name category once 3+ posts are grouped
    ▼
Response:
{
  "category_id": "668f...",
  "is_new": false,
  "similarity": 0.82,
  "domain": "sports",
  "name": "Cricket & Ipl"
}
```

## Project Structure

```
app/
  config.py              # Pydantic settings — reads .env
  main.py                # FastAPI entry point + MongoDB lifespan
  api/
    analyze.py           # /analyze/text, /image, /video endpoints
    auth.py              # API key authentication
  db/
    mongodb.py           # Motor async connection manager
    model.py             # Category document helpers
  ml/
    embeddings.py        # SigLIP text/image/video embeddings
    domain.py            # Zero-shot domain classifier
    clustering.py        # Online category assignment (CategoryManager)
    naming.py            # TF-IDF category naming
    similarity.py        # Cosine similarity helper
    sample.mp4           # Sample video used for testing
  utils/                 # Helper utilities (currently empty)

.env                     # Secrets — MongoDB URI, API key (git-ignored)
requirements.txt         # Python dependencies (includes pytest)
```

## Setup

### 1. Clone and create a virtual environment

```powershell
git clone https://github.com/YashMittal1503/xoodrip-content-intelligence.git
cd xoodrip-content-intelligence

python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
MONGO_URI=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/
MONGO_DB_NAME=xoodrip_intelligence
XOODRIP_API_KEY=dev-secret-key
```

Replace `<username>`, `<password>`, and `<cluster>` with your MongoDB Atlas credentials.

### 4. Start the server

```bash
uvicorn app.main:app --reload
```

The first start downloads the SigLIP model weights (~1.6 GB) from Hugging Face. Subsequent starts load from cache.

You should see:

```
[INFO] Loading SigLIP model (google/siglip-so400m-patch14-384) on cpu...
[SUCCESS] SigLIP model loaded — embedding dimension: 1152
[INFO] Building domain embeddings...
[SUCCESS] Domain embeddings built for 13 domains
✅ Connected to MongoDB Atlas — database: xoodrip_intelligence
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Useful Routes

| Route | Description |
|---|---|
| `GET /` | Service status |
| `GET /health` | Health check |
| `GET /docs` | Swagger UI (interactive API docs) |

## Authentication

All `/analyze/*` endpoints require an API key header:

```http
X-Api-Key: dev-secret-key
```

The key is configured via `XOODRIP_API_KEY` in `.env`. For local development, it defaults to `dev-secret-key`.

## API Endpoints

### `POST /analyze/text`

Analyze a text post and assign it to a category.

**Form fields:**
- `text` (required) — the post content

**Query params:**
- `include_scores` (optional, default `false`) — return per-domain confidence scores

**Example:**

```bash
curl -X POST "http://localhost:8000/analyze/text?include_scores=true" \
  -H "X-Api-Key: dev-secret-key" \
  -F "text=Virat Kohli hits a century in IPL 2025"
```

---

### `POST /analyze/image`

Analyze an uploaded image, optionally with a caption.

**Form fields:**
- `image` (required) — image file
- `caption` (optional) — text caption to combine with the image

**Query params:**
- `include_scores` (optional, default `false`) — return per-domain confidence scores

**Example:**

```bash
curl -X POST "http://localhost:8000/analyze/image?include_scores=true" \
  -H "X-Api-Key: dev-secret-key" \
  -F "image=@test_images/cricket.jpeg" \
  -F "caption=Cricket match highlights"
```

---

### `POST /analyze/video`

Analyze an uploaded video by sampling frames.

**Form fields:**
- `video` (required) — video file
- `caption` (optional) — text caption

**Query params:**
- `include_scores` (optional, default `false`) — return per-domain confidence scores

**Example:**

```bash
curl -X POST "http://localhost:8000/analyze/video?include_scores=true" \
  -H "X-Api-Key: dev-secret-key" \
  -F "video=@app/ml/sample.mp4" \
  -F "caption=Short video post"
```

## Response Format

```json
{
  "category_id": "668fa1b2c3d4e5f6a7b8c9d0",
  "is_new": false,
  "similarity": 0.82,
  "domain": "sports",
  "name": "Cricket & Ipl"
}
```

With `include_scores=true`:

```json
{
  "category_id": "668fa1b2c3d4e5f6a7b8c9d0",
  "is_new": true,
  "similarity": 0.0,
  "domain": "sports",
  "name": null,
  "domain_scores": {
    "sports": 0.31,
    "bollywood": 0.22,
    "government": 0.18,
    "politics": 0.16,
    "...": "..."
  }
}
```

## How Categorization Works

```
Input Post
    │
    ▼
┌─────────────────────────┐
│  SigLIP Embedding       │  Text, image, or video → 1152-d vector
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│  Domain Classification  │  Compare against 13 pre-computed domain
│  (zero-shot)            │  centroids + keyword boosting
└─────────┬───────────────┘
          │
          ▼
┌─────────────────────────┐
│  Category Matching      │  Load centroids from MongoDB,
│  (cosine similarity)    │  find best match within the same domain
└─────────┬───────────────┘
          │
    ┌─────┴──────┐
    │            │
  Match?      No match
    │            │
    ▼            ▼
  Update      Create new
  centroid     category
  & count      in MongoDB
    │            │
    └─────┬──────┘
          │
          ▼
   Return result JSON
```

**Key mechanisms:**

- **Domain gating** — A cricket post will never merge into a Bollywood category, even if embeddings are close.
- **Dynamic threshold** — Sports categories use a looser threshold (0.70 vs 0.78) because sports posts vary more. Mature categories (3+ posts) use a stricter threshold.
- **Incremental centroid** — `new = (old × count + new_embedding) / (count + 1)`. No need to store all historical embeddings.
- **Auto-naming** — Once a category accumulates 3+ texts, TF-IDF extracts the top keywords and joins them (e.g., "Cricket & Ipl & Match").

## MongoDB Document Schema

Categories are stored in the `categories` collection:

```json
{
  "_id": ObjectId("668fa1b2c3d4e5f6a7b8c9d0"),
  "name": "Cricket & Ipl",
  "domain": "sports",
  "count": 5,
  "centroid": [0.012, -0.034, 0.056, ...],
  "texts": [
    "Virat Kohli hits a century in IPL",
    "India wins cricket world cup",
    "..."
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `_id` | ObjectId | Auto-generated unique identifier |
| `name` | string or null | Human-readable name, generated after 3+ posts |
| `domain` | string | Broad domain (sports, tech, etc.) |
| `count` | integer | Number of posts assigned to this category |
| `centroid` | array of floats | Average SigLIP embedding (1152 values) |
| `texts` | array of strings | Sample post texts for naming |

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MONGO_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `MONGO_DB_NAME` | `xoodrip_intelligence` | Database name |
| `XOODRIP_API_KEY` | `dev-secret-key` | API key for `/analyze/*` routes |

All variables are read from `.env` via Pydantic Settings.

## GitHub

```
https://github.com/YashMittal1503/xoodrip-content-intelligence.git
```
