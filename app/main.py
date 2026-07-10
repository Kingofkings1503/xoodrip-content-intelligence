"""
main.py
-------
The entry point of the FastAPI application.

connect_to_mongo() : connects to MongoDB Atlas on startup
close_mongo() : disconnects on shutdown
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db.mongodb import connect_to_mongo, close_mongo
from app.api.analyze import router as analyze_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Code here runs ONCE when the server starts up and shuts down.

    Startup:  Connect to MongoDB Atlas
    Shutdown: Close the connection cleanly
    """
    # ── Startup ──
    await connect_to_mongo()
    yield
    # ── Shutdown ──
    await close_mongo()


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
async def root():
    return {
        "status": "ok",
        "service": "Content Intelligence Service",
    }


@app.get("/health")
async def health_check():
    return {"healthy": True}


app.include_router(analyze_router, prefix="/analyze", tags=["Analysis"])
