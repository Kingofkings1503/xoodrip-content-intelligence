"""
main.py
-------
The entry point of the FastAPI application.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db.database import engine, Base
from app.api.analyze import router as analyze_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Code here runs ONCE when the server starts up.

    `Base.metadata.create_all(engine)` inspects every class that inherits
    from Base (i.e., our Category model) and creates the matching SQL table
    in xoodrip.db if it doesn't already exist.

    Think of it like: "Before we open for business, make sure the filing
    cabinet (database) has the right folders (tables) in it."
    """
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created / verified.")
    yield
    # (anything after `yield` runs on shutdown — nothing needed here)


# Create the FastAPI app, wired to our lifespan startup handler
app = FastAPI(
    title="Xoodrip Content Intelligence",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Allows browser clients (Hoppscotch, frontends) to call this API locally.
# For production, replace allow_origins=["*"] with your actual frontend URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # ← restrict to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Content Intelligence Service",
    }


@app.get("/health")
def health_check():
    return {"healthy": True}


app.include_router(analyze_router, prefix="/analyze", tags=["Analysis"])
