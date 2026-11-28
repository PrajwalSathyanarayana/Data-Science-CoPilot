"""Utility — saves and loads jobs, reports and uploads. Today on local disk; Supabase replaces only this module.

Every report lives in its own folder, reports/<report_id>/:
- job.json       the job's status (rewritten after every change, so it survives a restart)
- analysis.json  everything the analysis produced
- upload.parquet the cleaned upload, compressed (kept for the chart "download the data" button; never offered back)
- other report files (notebook.ipynb, later report.md and report.pdf) written by their tools

Report IDs and file names coming from a web address are never trusted: IDs must be in the exact format we
create, file names must be on the REPORT_FILES list, and the final path must stay inside the reports folder.
"""
import json
import logging
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import pyarrow

from ..config import REPORT_FILES, REPORTS_DIR
from ..models import AnalysisRun, JobRecord

logger = logging.getLogger(__name__)

REPORT_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
JOB_FILE = "job.json"
ANALYSIS_FILE = "analysis.json"
UPLOAD_FILE = "upload.parquet"
INTERRUPTED_MESSAGE = "The analysis was interrupted because the server restarted. Please upload the file again."


def new_report_id() -> str:
    """A random 32-character ID. There are 2^122 possible values, so nobody can guess another user's report."""
    return uuid.uuid4().hex


def is_report_id(value: str) -> bool:
    return bool(REPORT_ID_PATTERN.fullmatch(value))


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ReportStore:
    """Local-disk storage for jobs and reports. Safe to use from several threads at once."""

    def __init__(self, root: str = REPORTS_DIR) -> None:
        self.root = os.path.realpath(root)
        self._lock = threading.Lock()
        os.makedirs(self.root, exist_ok=True)

    # --- paths ---

    def folder(self, report_id: str) -> str:
        """The report's folder. Refuses anything that isn't one of our IDs."""
        if not is_report_id(report_id):
            raise ValueError("not a valid report ID")
        return os.path.join(self.root, report_id)

    def _inside_root(self, path: str) -> bool:
        real = os.path.realpath(path)
        return os.path.commonpath([real, self.root]) == self.root and real != self.root

    def _write_json(self, path: str, data: Any) -> None:
        # Write to a temporary file first, then swap it in, so a reader never sees half a file
        temporary = f"{path}.{threading.get_ident()}.tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(data, handle, allow_nan=False)
        os.replace(temporary, path)

    # --- jobs ---

    def save_job(self, job: JobRecord) -> None:
        folder = self.folder(job.report_id)
        with self._lock:
            os.makedirs(folder, exist_ok=True)
            self._write_json(os.path.join(folder, JOB_FILE), job.model_dump(mode="json"))

    def load_job(self, report_id: str) -> Optional[JobRecord]:
        if not is_report_id(report_id):
            return None
        path = os.path.join(self.folder(report_id), JOB_FILE)
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as handle:
            return JobRecord.model_validate_json(handle.read())

    def mark_interrupted(self) -> int:
        """At startup: jobs still queued or running were cut off by the restart, so mark them failed."""
        count = 0
        for name in os.listdir(self.root):
            job = self.load_job(name)
            if job is not None and job.status in ("queued", "running"):
                job.status, job.step, job.error, job.finished_at = "failed", "interrupted", INTERRUPTED_MESSAGE, utc_now()
                self.save_job(job)
                count += 1
        if count:
            logger.warning("Marked %d job(s) as interrupted by a restart", count)
        return count

    # --- results ---

    def save_analysis(self, report_id: str, run: AnalysisRun) -> None:
        self._write_json(os.path.join(self.folder(report_id), ANALYSIS_FILE), run.model_dump(mode="json"))

    def load_analysis(self, report_id: str) -> Optional[Dict[str, Any]]:
        path = os.path.join(self.folder(report_id), ANALYSIS_FILE)
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)

    def save_upload(self, report_id: str, df: pd.DataFrame) -> None:
        """Keep the cleaned upload, compressed. Columns mixing numbers and text are saved as text."""
        path = os.path.join(self.folder(report_id), UPLOAD_FILE)
        try:
            df.to_parquet(path, index=False)
        except (pyarrow.ArrowException, TypeError, ValueError):
            as_text = {name: "string" for name in df.columns if df[name].dtype == object}
            df.astype(as_text).to_parquet(path, index=False)

    # --- downloads ---

    def file_path(self, report_id: str, filename: str) -> Optional[str]:
        """The path of a report file to download, or None. Three checks: our ID, a listed name, inside the folder."""
        if not is_report_id(report_id) or filename not in REPORT_FILES:
            return None
        path = os.path.join(self.folder(report_id), filename)
        if not self._inside_root(path) or not os.path.isfile(path):
            return None
        return path

    def available_files(self, report_id: str) -> List[str]:
        return [name for name in REPORT_FILES if self.file_path(report_id, name) is not None]
