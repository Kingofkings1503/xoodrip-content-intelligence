import os
from fastapi import Header, HTTPException, status
from typing import Optional

# Load API key from environment
API_KEY = os.getenv("XOODRIP_API_KEY")

if not API_KEY:
    # Fallback for local development ONLY
    API_KEY = "dev-secret-key"


def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """
    Dependency to verify API key from request headers
    """
    if x_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key missing"
        )

    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key"
        )

    return True
