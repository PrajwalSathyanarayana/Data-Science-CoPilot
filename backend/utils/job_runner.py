"""Utility — runs each uploaded file's analysis in the background, so the upload request can answer at once.

The upload request gets a report ID straight away; the website then asks for the status every few seconds.
At most ANALYSIS_WORKERS analyses run at the same time, and at most ANALYSIS_QUEUE_MAX wait or run in total,
so a burst of uploads can't exhaust memory or the API budget. Every status change is saved, so a restart
never leaves a job looking like it's still running.
"""
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import Any, Dict, List, Mapping, Optional

import anthropic

from ..agents.orchestrator import run_analysis
from ..config import ANALYSIS_QUEUE_MAX, ANALYSIS_WORKERS, DATASET_NAME_MAX_CHARS
from ..models import AnalyzeConfig, JobRecord
from ..tools.notebook_generator import generate_notebook
from .file_handler import FileLoadError, load_file
from .report_store import ReportStore, new_report_id, utc_now

logger = logging.getLogger(__name__)

UNEXPECTED_ERROR = "Something went wrong while analysing this file. Please try again, or try a different file."


class ServerBusyError(Exception):
    """Too many analyses are waiting already; the upload should be tried again shortly."""


class JobRunner:
    def __init__(
        self,
        store: ReportStore,
        workers: int = ANALYSIS_WORKERS,
        queue_max: int = ANALYSIS_QUEUE_MAX,
        client: Optional[anthropic.Anthropic] = None,
    ) -> None:
        self.store = store
        self.queue_max = queue_max
        self.client = client  # None = the real Claude client; tests pass a fake
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="analysis")
        self._lock = threading.Lock()
        self._active: Dict[str, Future] = {}

    def submit(self, file_bytes: bytes, filename: str, config: AnalyzeConfig, owner: Optional[str] = None) -> JobRecord:
        """Queue one analysis and return its job record at once.

        Raises:
            ServerBusyError: ANALYSIS_QUEUE_MAX analyses are already waiting or running.
        """
        with self._lock:
            if len(self._active) >= self.queue_max:
                raise ServerBusyError()
            job = JobRecord(
                report_id=new_report_id(),
                status="queued",
                step="waiting to start",
                dataset_name=filename[:DATASET_NAME_MAX_CHARS],
                created_at=utc_now(),
                owner=owner,
            )
            self.store.save_job(job)
            future = self._pool.submit(self._run, job, file_bytes, filename, config)
            self._active[job.report_id] = future
            future.add_done_callback(lambda _, report_id=job.report_id: self._finished(report_id))
        logger.info("Queued job %s (%d bytes)", job.report_id, len(file_bytes))
        return job

    def _finished(self, report_id: str) -> None:
        with self._lock:
            self._active.pop(report_id, None)

    def _update(self, job: JobRecord, **changes: Any) -> None:
        for name, value in changes.items():
            setattr(job, name, value)
        self.store.save_job(job)

    def _run(self, job: JobRecord, file_bytes: bytes, filename: str, config: AnalyzeConfig) -> None:
        """One analysis from start to finish. Never raises: every problem ends as a failed job with a safe message."""
        try:
            self._update(job, status="running", step="reading the file")
            loaded = load_file(file_bytes, filename, unit_choices=config.units or None)

            run = run_analysis(
                loaded,
                target_column=config.target_column,
                date_dayfirst=config.date_dayfirst,
                client=self.client,
                progress=lambda step: self._update(job, step=step),
            )

            self._update(job, step="writing the notebook")
            payloads: Mapping[str, Mapping[str, Any]] = {r.tool_name: r.payload for r in run.tool_results if r.success}
            run.tool_results.append(generate_notebook(
                loaded, job.dataset_name, job.report_id, tool_payloads=payloads, output_dir=self.store.folder(job.report_id)
            ))

            self._update(job, step="saving the report")
            self.store.save_upload(job.report_id, loaded.df)
            self.store.save_analysis(job.report_id, run)
            self._update(job, status="done", step="done", finished_at=utc_now())
            logger.info("Job %s done in %d ms", job.report_id, run.step_timings_ms.get("total", 0))
        except FileLoadError as exc:
            # The loader's messages are written for users ("The file has no data rows")
            self._fail(job, str(exc))
        except ValueError as exc:
            # The only ValueError the analysis raises on purpose is an unknown target column
            if "target_column" in str(exc):
                self._fail(job, f"The target column '{config.target_column}' is not in this file.")
            else:
                logger.exception("Job %s failed", job.report_id)
                self._fail(job, UNEXPECTED_ERROR)
        except Exception:
            logger.exception("Job %s failed", job.report_id)
            self._fail(job, UNEXPECTED_ERROR)

    def _fail(self, job: JobRecord, message: str) -> None:
        logger.warning("Job %s failed: %s", job.report_id, message)
        self._update(job, status="failed", step="failed", error=message, finished_at=utc_now())

    # --- for tests, scripts and shutdown ---

    def active_count(self) -> int:
        with self._lock:
            return len(self._active)

    def wait_idle(self, timeout: float) -> bool:
        """Wait until every job has finished. Returns False if some were still running after `timeout` seconds."""
        with self._lock:
            pending: List[Future] = list(self._active.values())
        _, not_done = wait(pending, timeout=timeout)
        return not not_done

    def shutdown(self) -> None:
        # Running jobs are not waited for; on the next start they are marked as interrupted
        self._pool.shutdown(wait=False, cancel_futures=True)
