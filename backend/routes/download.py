"""FastAPI route — GET /api/download/<report_id>/<filename>: sends one report file, only with a valid, unexpired link.

Three locks: the link's signature must be one the server made and still in date; the report ID and file name
must be ones we create; and the file must really be inside the reports folder.
"""
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from ..utils.signed_links import check_link

router = APIRouter()


@router.get("/download/{report_id}/{filename}")
def download_file(report_id: str, filename: str, request: Request, expires: Optional[int] = None, sig: Optional[str] = None) -> FileResponse:
    # The signature is checked first, so someone without a valid link can't even learn which files exist
    check = check_link(report_id, filename, expires, sig)
    if check == "expired":
        raise HTTPException(status_code=410, detail="This download link has expired. Open the report again for a fresh link.")
    if check == "invalid":
        raise HTTPException(status_code=403, detail="This download link is not valid.")

    path = request.app.state.store.file_path(report_id, filename)
    if path is None:
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path=path, filename=filename, media_type="application/octet-stream")
