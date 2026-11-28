"""FastAPI app factory — creates the app, its storage and background job runner, and registers the routes."""
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import ALLOWED_ORIGINS, MAX_FILE_SIZE, UPLOAD_FORM_ALLOWANCE_BYTES
from .routes import analyze, download, health, status
from .utils.job_runner import JobRunner
from .utils.report_store import ReportStore
from .utils.upload_limit import UploadSizeLimit


def create_app(
    store: Optional[ReportStore] = None,
    runner: Optional[JobRunner] = None,
    max_upload_bytes: int = MAX_FILE_SIZE,
) -> FastAPI:
    """Build the app. Tests pass their own store (a temporary folder) and runner (a fake Claude client)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.store = store or ReportStore()
        # Jobs cut off by the last shutdown would otherwise look like they're still running
        app.state.store.mark_interrupted()
        app.state.runner = runner or JobRunner(app.state.store)
        yield
        app.state.runner.shutdown()

    app = FastAPI(lifespan=lifespan)
    app.state.max_upload_bytes = max_upload_bytes

    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )
    app.add_middleware(UploadSizeLimit, path="/api/analyze", max_body_bytes=max_upload_bytes + UPLOAD_FORM_ALLOWANCE_BYTES)

    app.include_router(analyze.router, prefix="/api", tags=["analyze"])
    app.include_router(status.router, prefix="/api", tags=["status"])
    app.include_router(download.router, prefix="/api", tags=["download"])
    app.include_router(health.router, prefix="/api", tags=["health"])
    return app


app = create_app()
