"""FastAPI route — POST /api/analyze: checks the upload and its settings, then starts the analysis in the background.

It answers at once with a report ID; the website then asks GET /api/status/<report_id> for progress.
"""
import json
from typing import Any, Dict

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import ValidationError

from ..config import ALLOWED_FILE_TYPE, MAX_FILE_SIZE, UPLOAD_CHUNK_BYTES
from ..models import AnalyzeConfig
from ..utils.job_runner import ServerBusyError

router = APIRouter()

TOO_LARGE = f"The file is too large. The limit is {MAX_FILE_SIZE // (1024 * 1024)} MB."


def _parse_config(raw: str) -> AnalyzeConfig:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="The settings (config) are not valid JSON.")
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="The settings (config) must be a JSON object.")
    try:
        return AnalyzeConfig.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}" for error in exc.errors())
        raise HTTPException(status_code=400, detail=f"The settings (config) have problems: {problems}")


def _check_file_type(filename: str) -> None:
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in ALLOWED_FILE_TYPE:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed types: {', '.join(ALLOWED_FILE_TYPE)}")


async def _read_within_limit(file: UploadFile, limit: int) -> bytes:
    """Read the upload 1 MB at a time and stop as soon as it passes the limit, so memory never holds more."""
    pieces = []
    total = 0
    while chunk := await file.read(UPLOAD_CHUNK_BYTES):
        total += len(chunk)
        if total > limit:
            raise HTTPException(status_code=413, detail=TOO_LARGE)
        pieces.append(chunk)
    return b"".join(pieces)


@router.post("/analyze", status_code=202)
async def analyze_file(request: Request, file: UploadFile = File(...), config: str = Form("{}")) -> Dict[str, Any]:
    settings = _parse_config(config)
    filename = (file.filename or "").strip()
    _check_file_type(filename)

    contents = await _read_within_limit(file, request.app.state.max_upload_bytes)
    if not contents:
        raise HTTPException(status_code=400, detail="The file is empty.")

    try:
        job = request.app.state.runner.submit(contents, filename, settings)
    except ServerBusyError:
        raise HTTPException(
            status_code=503,
            detail="Too many files are being analysed right now. Please try again in a minute.",
            headers={"Retry-After": "30"},
        )
    return {"report_id": job.report_id, "status": job.status, "status_url": f"/api/status/{job.report_id}"}
