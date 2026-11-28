"""Smoke test for the API (upload, background job, status, signed downloads) on the real sample files.

Run from the project root (scripted fake LLM, no API cost; reports go to a temporary folder):
    python -m scripts.smoke.smoke_api
Needs DOWNLOAD_SIGNING_KEY in .env.
"""
import asyncio
import json
import logging
import os
import tempfile
import time

from fastapi.testclient import TestClient

from backend.app import create_app
from backend.utils.job_runner import JobRunner
from backend.utils.report_store import ReportStore
from backend.utils.signed_links import signed_link
from scripts.smoke.common import SmokeRunner
from tests.llm_fakes import ScriptedClient, answer, make_response

WEATHER_ANSWER = dict(
    rationale="Storm warnings are rare (4.1%), so that imbalance matters most.",
    focus_columns=["storm_warning", "rainfall_mm", "observed_on"],
    suggested_target="storm_warning",
    flag_order=[1, 3, 2],
)
ANALYSIS_TOOLS = ["dataset_inspector", "missing_value_analyzer", "outlier_analyzer", "correlation_analyzer",
                  "visualization_analyzer", "notebook_generator"]


def sample(name: str) -> bytes:
    with open(f"tests/sample_data/{name}", "rb") as handle:
        return handle.read()


def raw_get(app, path: str, query: bytes = b"") -> int:
    """Send a path exactly as written (browsers would tidy up '..'; an attacker's tool doesn't)."""
    result = {}
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET", "scheme": "http",
             "path": path, "raw_path": path.encode(), "query_string": query, "root_path": "",
             "headers": [(b"host", b"smoke")], "server": ("smoke", 80), "client": ("smoke", 1)}

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            result["status"] = message["status"]

    asyncio.run(app(scope, receive, send))
    return result["status"]


def main() -> None:
    logging.disable(logging.WARNING)
    runner_out = SmokeRunner("API: upload, background job, status, downloads")
    store = ReportStore(tempfile.mkdtemp())
    # The fake planner answers weather, then refuses for the other files (they still get a full fallback report)
    fake = ScriptedClient([answer(**WEATHER_ANSWER)] + [make_response("", stop_reason="refusal")] * 3)
    jobs = JobRunner(store, client=fake)
    app = create_app(store, jobs, max_upload_bytes=50 * 1024 * 1024)

    with TestClient(app) as client:
        runner_out.section("weather_ddmm.csv: upload answers at once, the job runs in the background")
        start = time.perf_counter()
        response = client.post("/api/analyze", files={"file": ("weather_ddmm.csv", sample("weather_ddmm.csv"))},
                               data={"config": json.dumps({"date_dayfirst": True})})
        answered_ms = (time.perf_counter() - start) * 1000
        runner_out.check("status code", 202, response.status_code)
        runner_out.check("answered in under 1 s (before the analysis finished)", True, answered_ms < 1000)
        report_id = response.json()["report_id"]
        jobs.wait_idle(120)
        status = client.get(f"/api/status/{report_id}").json()
        runner_out.check("final status", ("done", None), (status["status"], status["error"]))
        runner_out.check("results", ANALYSIS_TOOLS, [t["tool_name"] for t in status["result"]["tool_results"]])
        runner_out.check("files offered", ["analysis.json", "notebook.ipynb"], sorted(status["downloads"]))
        runner_out.check("files saved", ["analysis.json", "job.json", "notebook.ipynb", "upload.parquet"],
                         sorted(os.listdir(store.folder(report_id))))
        runner_out.check("signed download works", 200, client.get(status["downloads"]["notebook.ipynb"]).status_code)

        runner_out.section("other sample files, two at a time")
        ids = {}
        for name in ("shop_orders.csv", "hospital.xlsx", "survey_with_gaps.csv"):
            ids[name] = client.post("/api/analyze", files={"file": (name, sample(name))}).json()["report_id"]
        jobs.wait_idle(300)
        for name, rid in ids.items():
            final = client.get(f"/api/status/{rid}").json()
            runner_out.check(f"{name}: done, all 6 results", ("done", 6),
                             (final["status"], len(final["result"]["tool_results"])))

        runner_out.section("refused uploads")
        runner_out.check("unsupported type (.txt)", 400,
                         client.post("/api/analyze", files={"file": ("notes.txt", b"x")}).status_code)
        runner_out.check("misspelled setting", 400, client.post(
            "/api/analyze", files={"file": ("w.csv", b"a\n1\n")}, data={"config": '{"target_colum": "a"}'}).status_code)
        too_big = client.post("/api/analyze", files={"file": ("big.csv", b"a,b\n" + b"1,2\n" * 13_200_000)})
        runner_out.check("51 MB file refused from its size header", 413, too_big.status_code)

        runner_out.section("download attacks")
        runner_out.check("/api/download/../.env", 403, raw_get(app, "/api/download/../.env"))
        query = signed_link("..", ".env").split("?")[1].encode()
        runner_out.check("same, with a correctly signed link", 404, raw_get(app, "/api/download/../.env", query))
        runner_out.check("unsigned link to a real report", 403, client.get(f"/api/download/{report_id}/analysis.json").status_code)
        expired = signed_link(report_id, "analysis.json", now=time.time() - 7200)
        runner_out.check("expired link", 410, client.get(expired).status_code)
        runner_out.check("signed link to the stored upload", 404, client.get(signed_link(report_id, "upload.parquet")).status_code)

    runner_out.finish()


if __name__ == "__main__":
    main()
