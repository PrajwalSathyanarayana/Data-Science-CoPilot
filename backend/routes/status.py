"""FastAPI route — GET /api/status/<report_id>: how an analysis is going, and the result with download links once done."""
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request

from ..utils.report_store import is_report_id
from ..utils.signed_links import signed_link

router = APIRouter()


@router.get("/status/{report_id}")
def report_status(report_id: str, request: Request) -> Dict[str, Any]:
    store = request.app.state.store
    job = store.load_job(report_id) if is_report_id(report_id) else None
    if job is None:
        raise HTTPException(status_code=404, detail="No report with this ID.")

    body: Dict[str, Any] = job.model_dump(mode="json", exclude={"owner"})
    if job.status == "done":
        body["result"] = store.load_analysis(report_id)
        # Fresh links each time, so a page opened later still gets working downloads
        body["downloads"] = {name: signed_link(report_id, name) for name in store.available_files(report_id)}
    return body
