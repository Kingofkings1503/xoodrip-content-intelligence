from fastapi import FastAPI

# Create FastAPI app instance
app = FastAPI(title="Xoodrip Content Intelligence")

# Root endpoint
@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Content Intelligence Service"
    }
@app.get("/health")
def health_check():
    return {
        "healthy": True
    }
