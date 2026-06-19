"""FastAPI router — POST /api/analyze endpoint that accepts a file upload and returns a FinalReport."""
import json
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from ..config import ALLOWED_FILE_TYPE, MAX_FILE_SIZE

router = APIRouter()
@router.post("/analyze")
async def analyze_file(file: UploadFile=File(...), config: str=Form(...)):
    
    # Validate file type
    if not any(file.filename.endswith(ext) for ext in ALLOWED_FILE_TYPE):
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed types: {ALLOWED_FILE_TYPE}")
    
    try:
        config = json.loads(config)
    except json.JSONDecodeError:
        raise HTTPException(status_code = 400, detail = "Invalid configuration")
    
    # Validate file size
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File size exceeds the maximum limit of {MAX_FILE_SIZE} bytes.")
    
    await file.seek(0)  # Reset file pointer after reading for size validation
    # Process the file and generate the report (placeholder logic)
    report = {
        "summary": "This is a summary of the analysis.",
        "insights": ["Insight 1", "Insight 2"],
        "recommendations": ["Recommendation 1", "Recommendation 2"]
    }
    
    return JSONResponse(content=report)