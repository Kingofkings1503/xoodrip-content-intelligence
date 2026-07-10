"""
analyze.py
----------
API routes for analyzing text, image, and video content.

WHAT CHANGED vs. the old version:
  - All endpoints are now `async def` (were `def`)
    → Required because Motor (MongoDB driver) uses `await` for queries
  - `db: Session = Depends(get_db)` → `db = Depends(get_db)`
  - `CategoryManager(db=db)` now uses Motor instead of SQLAlchemy
  - `manager.assign_category(...)` is now awaited (it's async)

WHY ASYNC?
  - "async def" means the function can "pause" while waiting for I/O
    (like database queries) without blocking the entire server.
  - "await" means "pause here until this finishes, but let other
    requests be handled in the meantime"
  - Without async, if 10 requests come in, they queue up one-by-one.
    With async, the server handles them concurrently.
"""

from fastapi import APIRouter, UploadFile, File, Form, Depends
from typing import Optional
import tempfile
import os

from app.ml.embeddings import embed_text, embed_image, embed_video, embed_multimodal
from app.ml.clustering import CategoryManager
from app.ml.domain import get_domain_scores
from app.api.auth import verify_api_key
from app.db.mongodb import get_db

router = APIRouter()


@router.post("/text", dependencies=[Depends(verify_api_key)])
async def analyze_text(
    text: str = Form(...),
    db=Depends(get_db),
    include_scores: bool = False,
):
    """
    Analyze a piece of text and assign it to a category.

    Set include_scores=true to also receive a breakdown of how confident
    the model is for every domain (useful for debugging).
    """
    manager = CategoryManager(db=db)
    embedding = embed_text(text)
    result = await manager.assign_category(embedding, text)
    if include_scores:
        result["domain_scores"] = get_domain_scores(embedding, text)
    return result


@router.post("/image", dependencies=[Depends(verify_api_key)])
async def analyze_image(
    image: UploadFile = File(...),
    caption: Optional[str] = Form(None),
    db=Depends(get_db),
    include_scores: bool = False,
):
    """
    Analyze an uploaded image (optionally with a caption) and assign a category.
    """
    suffix = os.path.splitext(image.filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await image.read())
        image_path = tmp.name

    try:
        manager = CategoryManager(db=db)

        if caption:
            embedding = embed_multimodal(image_path, caption)
            result = await manager.assign_category(embedding, caption)
            text_for_scores = caption
        else:
            embedding = embed_image(image_path)
            result = await manager.assign_category(embedding, image.filename)
            text_for_scores = image.filename

        if include_scores:
            result["domain_scores"] = get_domain_scores(embedding, text_for_scores)
    finally:
        os.remove(image_path)

    return result


@router.post("/video", dependencies=[Depends(verify_api_key)])
async def analyze_video(
    video: UploadFile = File(...),
    caption: Optional[str] = Form(None),
    db=Depends(get_db),
    include_scores: bool = False,
):
    """
    Analyze an uploaded video and assign a category.
    """
    suffix = os.path.splitext(video.filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await video.read())
        video_path = tmp.name

    try:
        manager = CategoryManager(db=db)
        embedding = embed_video(video_path)
        text_for_scores = caption if caption else video.filename
        result = await manager.assign_category(embedding, text_for_scores)
        if include_scores:
            result["domain_scores"] = get_domain_scores(embedding, text_for_scores)
    finally:
        os.remove(video_path)

    return result
