"""FastAPI backend for Retinal Image Enhancement."""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import uvicorn
import sys

# Add src and project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

app = FastAPI(
    title="Retinal Image Enhancement API",
    description="Transform low-quality Zeiss Visuscout images to Zeiss Clarus quality",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import routers
from backend.app.routers import inference

app.include_router(inference.router, prefix="/api", tags=["inference"])


@app.get("/")
async def root():
    """Root endpoint - API status."""
    return {
        "status": "online",
        "message": "Retinal Image Enhancement API",
        "version": "1.0.0",
        "endpoints": {
            "process": "/api/process",
            "health": "/api/health",
            "docs": "/docs"
        }
    }


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "retinal-enhancement-api"
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

