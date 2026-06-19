"""FastAPI routes — GET /api/download/<report_id>/<filename> endpoint that serves saved report files."""
import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from ..config import REPORTS_DIR

router = APIRouter()
@router.get("/download/{report_id}/{filename}")
async def download_file(report_id: str, filename: str):
    file_path = os.path.join(REPORTS_DIR, report_id, filename)
    
    if os.path.exists(file_path):
        return FileResponse(path=file_path, 
                            filename=filename, 
                            media_type='application/octet-stream')
    else:
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")