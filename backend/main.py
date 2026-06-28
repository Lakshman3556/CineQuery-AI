from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import chat, ingest
from backend import config
import uvicorn

app = FastAPI(
    title="MedQuery AI Backend",
    description="Clinical Knowledge Assistant RAG backend",
    version="1.0.0"
)

# Enable CORS for frontend requests (React app runs on port 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers under the '/api' prefix
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(ingest.router, prefix="/api", tags=["Admin Ingestion"])

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "provider": config.LLM_PROVIDER}

if __name__ == "__main__":
    # Start the server on port 8000
    print("[MAIN] Starting FastAPI server on http://localhost:8000")
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
