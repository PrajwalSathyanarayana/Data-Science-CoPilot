"""Tests for the API: upload checks, background jobs, status, signed downloads and the report store.

A fake Claude client and a temporary reports folder are used, so nothing calls the real API or touches reports/.
"""
import asyncio
import json
import os
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.agents.orchestrator import run_analysis
from backend.app import create_app
from backend.models import AnalyzeConfig, JobRecord
from backend.utils.file_handler import load_file
from backend.utils.job_runner import UNEXPECTED_ERROR, JobRunner, ServerBusyError
from backend.utils.report_store import INTERRUPTED_MESSAGE, ReportStore, is_report_id, new_report_id, utc_now
from backend.utils.signed_links import check_link, signed_link
from backend.utils.upload_limit import UploadSizeLimit
from tests.llm_fakes import ScriptedClient, answer

WEATHER_ANSWER = dict(
    rationale="Storm warnings are rare (4.1%), so the imbalance matters most.",
    focus_columns=["storm_warning", "rainfall_mm", "observed_on"],
    suggested_target="storm_warning",
    flag_order=[1, 3, 2],
)
WEATHER = open("tests/sample_data/weather_ddmm.csv", "rb").read()


@pytest.fixture
def store(tmp_path):
    return ReportStore(str(tmp_path))


def make_client(store, answers=1, max_upload_bytes=None, queue_max=10):
    runner = JobRunner(store, client=ScriptedClient([answer(**WEATHER_ANSWER)] * answers), queue_max=queue_max)
    kwargs = {} if max_upload_bytes is None else {"max_upload_bytes": max_upload_bytes}
    return TestClient(create_app(store, runner, **kwargs)), runner


def upload(client, data=WEATHER, name="weather_ddmm.csv", config=None):
    return client.post("/api/analyze", files={"file": (name, data)}, data={"config": json.dumps(config or {})})


def finished_status(client, runner, report_id):
    assert runner.wait_idle(60)
    return client.get(f"/api/status/{report_id}").json()


# --- a normal upload ---

def test_upload_answers_at_once_and_the_job_finishes(store):
    client, runner = make_client(store)
    with client:
        response = upload(client, config={"date_dayfirst": True})
        assert response.status_code == 202
        report_id = response.json()["report_id"]
        assert is_report_id(report_id)
        assert response.json()["status_url"] == f"/api/status/{report_id}"

        status = finished_status(client, runner, report_id)
    assert (status["status"], status["step"], status["error"]) == ("done", "done", None)
    assert [t["tool_name"] for t in status["result"]["tool_results"]] == [
        "dataset_inspector", "missing_value_analyzer", "outlier_analyzer", "correlation_analyzer",
        "visualization_analyzer", "notebook_generator",
    ]
    assert status["result"]["plan"]["target_column"] == "storm_warning"
    assert set(status["downloads"]) == {"analysis.json", "notebook.ipynb"}
    assert "owner" not in status


def test_saved_files_include_the_cleaned_upload(store):
    client, runner = make_client(store)
    with client:
        report_id = upload(client).json()["report_id"]
        finished_status(client, runner, report_id)
    assert sorted(os.listdir(store.folder(report_id))) == ["analysis.json", "job.json", "notebook.ipynb", "upload.parquet"]
    assert len(pd.read_parquet(os.path.join(store.folder(report_id), "upload.parquet"))) == len(load_file(WEATHER, "w.csv").df)


def test_capital_letter_extension_is_accepted(store):
    client, _ = make_client(store)
    with client:
        assert upload(client, name="WEATHER.CSV").status_code == 202


# --- upload checks ---

@pytest.mark.parametrize("name", ["data.txt", "data", "", "report.csv.exe"])
def test_unsupported_file_types_are_refused(store, name):
    client, _ = make_client(store, answers=0)
    with client:
        response = upload(client, name=name)
    assert response.status_code in (400, 422)  # an empty name is refused by the form parser itself


@pytest.mark.parametrize(
    "config, message",
    [
        ("not json", "not valid JSON"),
        ("[1, 2]", "must be a JSON object"),
        ('{"target_colum": "x"}', "target_colum: Extra inputs are not permitted"),
        ('{"hypothesis_count": 0}', "hypothesis_count"),
        ('{"analysis_depth": "deep"}', "analysis_depth"),
    ],
)
def test_bad_settings_are_refused_with_the_reason(store, config, message):
    client, _ = make_client(store, answers=0)
    with client:
        response = client.post("/api/analyze", files={"file": ("w.csv", WEATHER)}, data={"config": config})
    assert response.status_code == 400
    assert message in response.json()["detail"]


def test_empty_file_is_refused(store):
    client, _ = make_client(store, answers=0)
    with client:
        response = upload(client, data=b"")
    assert response.status_code == 400 and response.json()["detail"] == "The file is empty."


def test_too_large_by_its_stated_size_is_refused_before_reading(store):
    # Limit 1,000 bytes + 64 KB room for the form: a 100 KB upload is refused from its size header alone
    client, _ = make_client(store, answers=0, max_upload_bytes=1000)
    with client:
        response = upload(client, data=b"a,b\n" + b"1,2\n" * 25_000)
    assert response.status_code == 413


def test_too_large_while_reading_is_refused(store):
    # 5 KB passes the header check (under 1,000 + 64 KB) but the route stops counting at 1,000
    client, _ = make_client(store, answers=0, max_upload_bytes=1000)
    with client:
        response = upload(client, data=b"a,b\n" + b"1,2\n" * 1250)
    assert response.status_code == 413
    assert response.json()["detail"].startswith("The file is too large")


def test_invalid_size_header_is_refused():
    sent = []

    async def app(scope, receive, send):
        sent.append("reached the app")

    async def send(message):
        if message["type"] == "http.response.start":
            sent.append(message["status"])

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    guard = UploadSizeLimit(app, path="/api/analyze", max_body_bytes=100)
    scope = {"type": "http", "method": "POST", "path": "/api/analyze", "headers": [(b"content-length", b"12abc")]}
    asyncio.run(guard(scope, receive, send))
    assert sent == [400]


def test_busy_server_asks_to_try_again(store):
    client, _ = make_client(store, answers=0, queue_max=0)
    with client:
        response = upload(client)
    assert response.status_code == 503
    assert response.headers["retry-after"] == "30"


# --- failed jobs ---

def test_unknown_target_column_fails_with_a_clear_message(store):
    client, runner = make_client(store, answers=0)
    with client:
        report_id = upload(client, config={"target_column": "nope"}).json()["report_id"]
        status = finished_status(client, runner, report_id)
    assert status["status"] == "failed"
    assert status["error"] == "The target column 'nope' is not in this file."


def test_unreadable_file_fails_with_the_loaders_message(store):
    client, runner = make_client(store, answers=0)
    with client:
        report_id = upload(client, data=b"only,a,header\n").json()["report_id"]
        status = finished_status(client, runner, report_id)
    assert status["status"] == "failed"
    assert status["error"] and status["error"] != UNEXPECTED_ERROR


def test_unexpected_errors_never_reach_the_user(store, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("internal detail C:\\secret\\path")

    monkeypatch.setattr("backend.utils.job_runner.run_analysis", broken)
    client, runner = make_client(store, answers=0)
    with client:
        report_id = upload(client).json()["report_id"]
        status = finished_status(client, runner, report_id)
    assert status["error"] == UNEXPECTED_ERROR


def test_restart_marks_unfinished_jobs_as_interrupted(store):
    for status in ("queued", "running", "done"):
        store.save_job(JobRecord(report_id=new_report_id(), status=status, step="x", dataset_name="d.csv", created_at=utc_now()))
    client, _ = make_client(store, answers=0)
    with client:  # starting the app runs the check
        pass
    jobs = [store.load_job(name) for name in os.listdir(store.root)]
    assert sorted(job.status for job in jobs) == ["done", "failed", "failed"]
    assert all(job.error == INTERRUPTED_MESSAGE for job in jobs if job.status == "failed")


# --- status ---

@pytest.mark.parametrize("report_id", ["0" * 32, "not-an-id", "..", "A" * 32])
def test_unknown_or_malformed_report_ids_get_404(store, report_id):
    client, _ = make_client(store, answers=0)
    with client:
        assert client.get(f"/api/status/{report_id}").status_code == 404


# --- downloads ---

@pytest.fixture
def finished_report(store):
    client, runner = make_client(store)
    with client:
        report_id = upload(client).json()["report_id"]
        status = finished_status(client, runner, report_id)
        yield client, report_id, status["downloads"]


def test_signed_link_downloads_the_file(finished_report):
    client, _, downloads = finished_report
    response = client.get(downloads["analysis.json"])
    assert response.status_code == 200
    assert json.loads(response.content)["plan"]["target_column"] == "storm_warning"


def test_link_without_signature_is_refused(finished_report):
    client, report_id, _ = finished_report
    assert client.get(f"/api/download/{report_id}/analysis.json").status_code == 403


def test_link_for_another_file_is_refused(finished_report):
    client, _, downloads = finished_report
    assert client.get(downloads["notebook.ipynb"].replace("notebook.ipynb", "analysis.json")).status_code == 403


def test_expired_link_says_so(finished_report):
    client, report_id, _ = finished_report
    old = signed_link(report_id, "analysis.json", now=time.time() - 7200)
    response = client.get(old)
    assert response.status_code == 410 and "expired" in response.json()["detail"]


@pytest.mark.parametrize("filename", ["job.json", "upload.parquet", ".env"])
def test_signed_links_to_files_not_on_the_list_get_404(finished_report, filename):
    client, report_id, _ = finished_report
    assert client.get(signed_link(report_id, filename)).status_code == 404


# --- pieces on their own ---

def test_signature_checks():
    now = 1_000_000.0
    link = signed_link("a" * 32, "report.pdf", now=now)
    expires = int(link.split("expires=")[1].split("&")[0])
    sig = link.split("sig=")[1]
    assert check_link("a" * 32, "report.pdf", expires, sig, now=now) == "ok"
    assert check_link("b" * 32, "report.pdf", expires, sig, now=now) == "invalid"
    assert check_link("a" * 32, "report.pdf", expires + 1, sig, now=now) == "invalid"
    assert check_link("a" * 32, "report.pdf", expires, sig, now=expires + 1) == "expired"
    assert check_link("a" * 32, "report.pdf", None, sig) == "invalid"


@pytest.mark.parametrize("bad", ["..", "../x", "A" * 32, "a" * 31, "a" * 33, ""])
def test_store_refuses_anything_but_our_ids(store, bad):
    with pytest.raises(ValueError):
        store.folder(bad)
    assert store.file_path(bad, "analysis.json") is None


def test_mixed_type_columns_are_saved_as_text(store):
    report_id = new_report_id()
    os.makedirs(store.folder(report_id))
    store.save_upload(report_id, pd.DataFrame({"mixed": [1, "a", 2.5]}))
    assert pd.read_parquet(os.path.join(store.folder(report_id), "upload.parquet"))["mixed"].tolist() == ["1", "a", "2.5"]


def test_runner_refuses_when_full(store):
    runner = JobRunner(store, queue_max=0)
    with pytest.raises(ServerBusyError):
        runner.submit(WEATHER, "w.csv", AnalyzeConfig())
    runner.shutdown()


def test_progress_names_each_step():
    steps = []
    run_analysis(load_file(WEATHER, "w.csv"), client=ScriptedClient([answer(**WEATHER_ANSWER)]), progress=steps.append)
    assert steps == ["looking at the data", "planning the analysis", "running the analysis tools", "drawing the charts"]
