"""
analyze.py
----------
API routes for analyzing text, image, and video content.

WHAT CHANGED vs. the old version:
  - Removed the global `category_manager = CategoryManager(...)` singleton.
    A global singleton was dangerous because it only existed in memory — 
    any restart would wipe all learned categories.

  - Now each API request gets its own fresh `CategoryManager` instance
    that is wired to the database via a `db` session (injected by FastAPI's
    dependency injection system using `Depends(get_db)`).

  - The DB session is automatically opened before the request and closed
    after it completes (handled by `get_db()` in database.py).
"""

from fastapi import APIRouter, UploadFile, File, Form, Depends
from typing import Optional
from sqlalchemy.orm import Session
import tempfile
import os

from app.ml.embeddings import embed_text, embed_image, embed_video, embed_multimodal
from app.ml.clustering import CategoryManager
from app.ml.domain import get_domain_scores
from app.api.auth import verify_api_key
from app.db.database import get_db

router = APIRouter()


@router.post("/text", dependencies=[Depends(verify_api_key)])
def analyze_text(
    text: str = Form(...),
    db: Session = Depends(get_db),
    include_scores: bool = False,          # ← optional: return per-domain scores
):
    """
    Analyze a piece of text and assign it to a category.

    Set ?include_scores=true to also receive a breakdown of how confident
    CLIP is for every domain (useful for debugging).
    """
    manager = CategoryManager(db=db)
    embedding = embed_text(text)
    result = manager.assign_category(embedding, text)
    if include_scores:
        result["domain_scores"] = get_domain_scores(embedding)
    return result


@router.post("/image", dependencies=[Depends(verify_api_key)])
def analyze_image(
    image: UploadFile = File(...),
    caption: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    include_scores: bool = False,
):
    """
    Analyze an uploaded image (optionally with a caption) and assign a category.
    """
    suffix = os.path.splitext(image.filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(image.file.read())
        image_path = tmp.name

    try:
        manager = CategoryManager(db=db)

        if caption:
            embedding = embed_multimodal(image_path, caption)
            result = manager.assign_category(embedding, caption)
        else:
            embedding = embed_image(image_path)
            result = manager.assign_category(embedding, image.filename)

        if include_scores:
            result["domain_scores"] = get_domain_scores(embedding)
    finally:
        os.remove(image_path)

    return result


@router.post("/video", dependencies=[Depends(verify_api_key)])
def analyze_video(
    video: UploadFile = File(...),
    caption: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    include_scores: bool = False,
):
    """
    Analyze an uploaded video and assign a category.
    """
    suffix = os.path.splitext(video.filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(video.file.read())
        video_path = tmp.name

    try:
        manager = CategoryManager(db=db)
        embedding = embed_video(video_path)
        result = manager.assign_category(embedding, caption if caption else video.filename)
        if include_scores:
            result["domain_scores"] = get_domain_scores(embedding)
    finally:
        os.remove(video_path)

    return result
