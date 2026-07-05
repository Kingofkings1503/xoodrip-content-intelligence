"""
auth.py
-------
API key authentication for protecting /analyze/* endpoints.

WHAT CHANGED:
  - Before: Used os.getenv() directly with a hardcoded fallback
  - Now:    Uses settings.XOODRIP_API_KEY from our centralized config
            (which reads from .env automatically)
"""

from fastapi import Header, HTTPException, status
from typing import Optional

from app.config import settings


def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """
    Dependency to verify API key from request headers.
    """
    if x_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key missing"
        )

    if x_api_key != settings.XOODRIP_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )

    return True
