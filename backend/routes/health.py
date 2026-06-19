"""FastAPI router — GET /api/health endpoint that returns service status and version."""
from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
async def health_check():
    """Health check endpoint to verify that the application is running."""
    return {"status": "ok", 
            "version": '0.1.0'}