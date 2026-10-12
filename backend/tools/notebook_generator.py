"""Tool — writes a Jupyter notebook that loads the upload exactly as DS Copilot did and re-checks each finding with short pandas code."""
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import nbformat
import pandas as pd

from ..config import CHART_MAX_BOXES, OUTLIER_IQR_K, OUTLIER_RARE_PCT, REPORTS_DIR
from ..models import ToolResult
from ..utils.file_handler import LoadedFile
from ..utils.formatting import format_pct
from ..utils.text_safety import to_plain_markdown
from .correlation_analyzer import report_pairs

logger = logging.getLogger(__name__)

TOOL_NAME = "notebook_generator"
NOTEBOOK_FILE = "notebook.ipynb"
REPORT_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")  # report ids become folder names, so nothing like "../"
ROWS_SHOWN = 10  # unusual rows shown by the row-check cell
MISSING_COLUMNS_SHOWN = 20


class _Notebook:
    """Collects cells. Every code cell is compile-checked; `checks` records the numbers each check cell should print."""

    def __init__(self) -> None:
        self.cells: List[Any] = []
        self.checks: List[Dict[str, Any]] = []

    def markdown(self, text: str) -> None:
        self.cells.append(nbformat.v4.new_markdown_cell(text))

    def code(self, source: str, finding: Optional[str] = None, expect: Sequence[str] = ()) -> None:
        compile(source, f"<cell {len(self.cells)}>", "exec")  # SyntaxError here means a generator bug
        if finding is not None:
            self.checks.append({"cell": len(self.cells), "finding": finding, "expect": list(expect)})
        self.cells.append(nbformat.v4.new_code_cell(source))


# --- writing code safely ---

def _col(df: pd.DataFrame, position: int) -> str:
    """Code for one column: by name, or by position when two columns share the name."""
    name = str(df.columns[position])
    if sum(1 for c in df.columns if str(c) == name) > 1:
        return f"df.iloc[:, {position}]"
    return f"df[{name!r}]"


def _md(text: Any) -> str:
    """Text from the upload or the LLM, made safe for a Markdown cell."""
    return to_plain_markdown(str(text)).replace("\n", " ")


def _event_value(series: pd.Series) -> Any:
    """The value the correlation tool coded as 1 in a two-value column: the less common one (ties: the larger)."""
    numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
    values = series.dropna() if numeric else series.dropna().astype(str)
    counts = values.value_counts().sort_index()
    return counts.index[1] if counts.iloc[1] <= counts.iloc[0] else counts.index[0]


def _coded(df: pd.DataFrame, position: int) -> str:
    """Code turning a two-value column into 1 (the tool's coded value) and 0, keeping gaps."""
    series = df.iloc[:, position]
    event = _event_value(series)
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return f"({_col(df, position)} == {float(event)!r}).astype(float).where({_col(df, position)}.notna())"
    return f"({_col(df, position)}.astype(str) == {str(event)!r}).astype(float).where({_col(df, position)}.notna())"


# --- sections ---

def _load_cells(nb: _Notebook, loaded: LoadedFile, dataset_name: str) -> None:
    nb.markdown(
        f"# DS Copilot analysis: {_md(dataset_name)}\n\n"
        "Put the uploaded file next to this notebook and run all cells. The file is read exactly as "
        "DS Copilot read it, and each finding in the report is followed by a short cell that re-computes it."
    )
    nb.code("import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\npd.set_option('display.max_columns', 50)")

    if loaded.file_type == "csv":
        encoding = "utf-8-sig" if loaded.encoding == "utf-8" else loaded.encoding
        read = f"df = pd.read_csv(FILE, sep={loaded.delimiter!r}, encoding={encoding!r})"
    else:
        read = f"df = pd.read_excel(FILE, sheet_name={loaded.sheet_name!r})"
    columns = [str(c) for c in loaded.df.columns]
    nb.code(
        f"FILE = {dataset_name!r}\n{read}\n\n"
        "# Column names as DS Copilot cleaned them (trimmed, lowercase, blanks named, duplicates numbered)\n"
        f"df.columns = {columns!r}\n"
        "print(df.shape)\ndf.head()",
        finding="file loads with the same shape",
        expect=[f"({loaded.df.shape[0]}, {loaded.df.shape[1]})"],
    )


def _missing_cells(nb: _Notebook, df: pd.DataFrame, missing: Mapping[str, Any]) -> None:
    columns = [c for c in missing.get("columns", []) if c["missing"] or c["placeholders"]]
    if not columns:
        return
    nb.markdown("## Missing values")
    shown = sorted(columns, key=lambda c: -(c["missing_pct"] + c["placeholder_pct"]))[:MISSING_COLUMNS_SHOWN]
    nb.markdown("\n".join(
        f"- **{_md(c['name'])}**: {c['missing']:,} missing ({format_pct(c['missing_pct'])})"
        + (f", {c['placeholders']:,} possible placeholders" if c["placeholders"] else "")
        for c in shown
    ))
    nb.code(
        "missing = df.isna().sum()\nmissing = missing[missing > 0].sort_values(ascending=False)\n"
        "for name, count in missing.items():\n    print(f'{name}: {count} missing')\n"
        "if len(missing):\n    missing.head(20).plot.bar(title='Missing values per column', ylabel='rows')\n    plt.show()",
        finding="missing counts",
        expect=[f"{c['name']}: {c['missing']} missing" for c in shown if c["missing"]],
    )
    with_placeholders = [c for c in shown if c["placeholders"]]
    if with_placeholders:
        names = ", ".join(_md(c["name"]) for c in with_placeholders)
        nb.markdown(f"Values that probably mean \"no data\" (counted, not changed) in: {names}")
        positions = {str(n): i for i, n in enumerate(df.columns)}
        lines = ["PLACEHOLDERS = " + repr(sorted({v.lower() for c in with_placeholders for v in c["placeholder_values"] if v != "(blank)"}))]
        for c in with_placeholders:
            lines += [
                f"text = {_col(df, positions[c['name']])}.dropna().astype(str).str.strip()",
                f"print({c['name']!r}, int(((text == '') | text.str.lower().isin(PLACEHOLDERS)).sum()), 'possible placeholders')",
            ]
        nb.code("\n".join(lines), finding="placeholder counts", expect=[f"{c['name']} {c['placeholders']} possible placeholders" for c in with_placeholders])

    positions = {str(n): i for i, n in enumerate(df.columns)}
    for pair in missing.get("missing_together", [])[:5]:
        a, b = pair["columns"]
        nb.markdown(f"**{_md(a)}** and **{_md(b)}** are missing in the same rows ({pair['overlap']:.0%} overlap)")
        nb.code(
            f"both = ({_col(df, positions[a])}.isna() & {_col(df, positions[b])}.isna()).sum()\nprint('missing in both:', both)",
            finding=f"{a} & {b} missing together", expect=[f"missing in both: {pair['both_missing']}"],
        )


def _relationship_cells(nb: _Notebook, df: pd.DataFrame, correlation: Mapping[str, Any]) -> None:
    pairs = report_pairs(correlation)
    if not pairs:
        return
    nb.markdown("## Relationships between columns")
    for pair in pairs:
        (a, b), (i, j), measure = pair["columns"], pair["positions"], pair["measure"]
        label = f"**{_md(a)}** and **{_md(b)}**"
        warning = ""
        if pair["near_duplicate"]:
            warning = " Near-duplicate" + (": involves the target, possible leakage." if pair["involves_target"] else ".")

        if measure == "pearson_spearman":
            nb.markdown(f"{label}: Pearson {pair['pearson']:.2f}, Spearman {pair['spearman']:.2f}." + warning
                        + (f" Banded score {pair['banded']:.2f}: related, but not in a straight line." if pair.get("curve") else ""))
            x, y = _col(df, i), _col(df, j)
            lines = [
                f"x, y = {x}, {y}",
                "print(f\"Pearson {x.corr(y):.4f}\")",
                "print(f\"Spearman {x.corr(y, method='spearman'):.4f}\")",
            ]
            expect = [f"Pearson {pair['pearson']:.4f}"] + ([f"Spearman {pair['spearman']:.4f}"] if pair.get("spearman_exact") else [])
            if pair.get("curve"):
                lines += [
                    "# Average of y in 10 equal-sized bands of x: a curve shows up here even when Pearson is near 0",
                    "bands = pd.qcut(x, 10, duplicates='drop')",
                    "print(y.groupby(bands, observed=True).mean().round(2))",
                ]
            lines += [f"plt.scatter(x, y, s=4, alpha=0.5)\nplt.xlabel({a!r}); plt.ylabel({b!r}); plt.show()"]
            nb.code("\n".join(lines), finding=f"{a} & {b}", expect=expect)

        elif measure in ("point_biserial", "phi"):
            nb.markdown(f"{label}: {pair['value']:.2f} ({measure.replace('_', '-')}); {_md(pair['detail'])}.{warning}")
            first = _coded(df, i) if pair["kinds"][0] == "two-value" else _col(df, i)
            second = _coded(df, j) if pair["kinds"][1] == "two-value" else _col(df, j)
            lines = [
                "# Two-value columns as 1 (the less common value) and 0; then an ordinary correlation",
                f"r = {first}.corr({second})",
                "print(f\"r {r:.4f}\")",
            ]
            if measure == "phi":
                lines.append(f"pd.crosstab({_col(df, i)}, {_col(df, j)})")
            else:
                number, group = (i, j) if pair["kinds"][0] == "number" else (j, i)
                lines.append(f"{_col(df, number)}.groupby({_col(df, group)}).mean()")
            nb.code("\n".join(lines), finding=f"{a} & {b}", expect=[f"r {pair['value']:.4f}"])

        elif measure == "eta":
            number, group = (i, j) if pair["kinds"][0] == "number" else (j, i)
            nb.markdown(f"{label}: eta {pair['value']:.2f} (adjusted for chance); {_md(pair['detail'])}.{warning}")
            nb.code(
                f"averages = {_col(df, number)}.groupby({_col(df, group)}).agg(['mean', 'count']).sort_values('mean', ascending=False)\n"
                "averages['mean'].head(20).plot.bar(ylabel='average'); plt.show()\naverages",
                finding=f"{a} & {b}",
            )
        else:
            nb.markdown(f"{label}: Cramer's V {pair['value']:.2f} (corrected for small samples).{warning}")
            nb.code(f"pd.crosstab({_col(df, i)}, {_col(df, j)}, normalize='index').round(3)", finding=f"{a} & {b}")


def _outlier_cells(nb: _Notebook, df: pd.DataFrame, outliers: Mapping[str, Any]) -> None:
    columns = sorted((c for c in outliers.get("columns", []) if c["outlier_count"]), key=lambda c: -c["outlier_pct"])[:CHART_MAX_BOXES]
    check = outliers.get("row_check", {})
    if not columns and not check.get("rows"):
        return
    nb.markdown(f"## Outliers\n\nBox-plot rule: outside Q1 - {OUTLIER_IQR_K:g} x IQR or Q3 + {OUTLIER_IQR_K:g} x IQR.")
    for c in columns:
        notes = []
        lines = [f"values = {_col(df, c['position'])}.replace([np.inf, -np.inf], np.nan).dropna().astype(float)"]
        if c["dominant"] is not None:
            lines.append(f"values = values[values != {c['dominant']['value']!r}]  # {c['dominant']['share']:.0%} of rows are this value; check the rest")
            notes.append(f"{c['dominant']['share']:.0%} of rows share one value, so the others are checked among themselves")
        if c["log_scale"]:
            lines += [
                "q1, q3 = np.percentile(np.log(values), [25, 75])  # long tail: measured on the log scale",
                f"low, high = np.exp(q1 - {OUTLIER_IQR_K!r} * (q3 - q1)), np.exp(q3 + {OUTLIER_IQR_K!r} * (q3 - q1))",
            ]
            notes.append("long-tailed, so measured on the log scale")
        else:
            lines += [
                "q1, q3 = np.percentile(values, [25, 75])",
                f"low, high = q1 - {OUTLIER_IQR_K!r} * (q3 - q1), q3 + {OUTLIER_IQR_K!r} * (q3 - q1)",
            ]
        outside = "(values < low) | (values > high)"
        if c["few_values"]:
            lines.append(f"share = values.map(values.value_counts(normalize=True))  # few different values: only rare ones count")
            outside = f"({outside}) & (share < {OUTLIER_RARE_PCT / 100!r})"
            notes.append(f"few different values, so only values under {OUTLIER_RARE_PCT:g}% of rows count")
        lines += [
            f"outside = values[{outside}]",
            "print(f\"normal range {low:.4f} to {high:.4f}; {len(outside)} outside\")",
            "plt.boxplot(values, whis=(0, 100)); plt.show()",
            "outside.sort_values()",
        ]
        nb.markdown(f"**{_md(c['name'])}**: {c['outlier_count']:,} outlier(s) ({format_pct(c['outlier_pct'])})" + (f"; {'; '.join(notes)}" if notes else ""))
        nb.code("\n".join(lines), finding=f"outliers in {c['name']}",
                expect=[f"normal range {c['lower']:.4f} to {c['upper']:.4f}; {c['outlier_count']} outside"])

    if check.get("status") == "done" and check.get("rows"):
        rows = [r["row"] - 1 for r in check["rows"][:ROWS_SHOWN]]
        nb.markdown(
            f"**Unusual combinations:** {check['unusual_count']:,} row(s) ({format_pct(check['unusual_pct'])}) are far from how "
            f"{', '.join(_md(c) for c in check['columns'])} usually go together (robust Mahalanobis distance, see the report)."
            + (" Caution: far more than expected; the columns may follow a curve or groups this check can't model." if check.get("warning") else "")
            + f" The {len(rows)} most unusual:"
        )
        nb.code(f"df.iloc[{rows!r}][{check['columns']!r}]", finding="most unusual rows")


def _insight_cells(nb: _Notebook, insights: Sequence[str]) -> None:
    cleaned = [_md(text).strip() for text in insights]
    cleaned = [text for text in cleaned if text]
    if cleaned:
        nb.markdown("## Insights\n\n" + "\n".join(f"- {text}" for text in cleaned))


# --- entry point ---

def generate_notebook(
    loaded: LoadedFile,
    dataset_name: str,
    report_id: str,
    tool_payloads: Optional[Mapping[str, Mapping[str, Any]]] = None,
    insights: Optional[Sequence[str]] = None,
) -> ToolResult:
    """Write reports/<report_id>/notebook.ipynb and return what it contains.

    The notebook reads the file with the settings DS Copilot detected, then re-checks each finding
    with a few lines of pandas whose printed numbers match the report. Text from the upload or the
    LLM is only placed in Markdown, cleaned of links, images and HTML; code cells are built only from
    tool results, with every value written as a quoted Python literal. Every code cell is
    compile-checked before the file is written.
    """
    try:
        if not REPORT_ID.match(report_id):
            raise ValueError("report_id may only contain letters, digits, '-' and '_'")
        payloads = tool_payloads or {}
        df = loaded.df

        nb = _Notebook()
        _load_cells(nb, loaded, dataset_name)
        nb.markdown("## Overview")
        nb.code("print(df.dtypes)\ndf.describe(include='all').T")
        _missing_cells(nb, df, payloads.get("missing_value_analyzer", {}))
        _relationship_cells(nb, df, payloads.get("correlation_analyzer", {}))
        _outlier_cells(nb, df, payloads.get("outlier_analyzer", {}))
        _insight_cells(nb, insights or [])

        notebook = nbformat.v4.new_notebook()
        notebook.cells = nb.cells
        notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nbformat.validate(notebook)

        output_dir = Path(REPORTS_DIR) / report_id
        output_dir.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, str(output_dir / NOTEBOOK_FILE))  # nbformat always writes UTF-8

        findings = len(nb.checks) - 1  # the first check is the file loading
        payload = {"file": NOTEBOOK_FILE, "cells": len(nb.cells), "findings_checked": findings, "checks": nb.checks}
        summary = f"Notebook ({NOTEBOOK_FILE}): loads the file as DS Copilot did and re-checks {findings} finding(s) with short pandas code"
        return ToolResult(tool_name=TOOL_NAME, success=True, payload=payload, summary=summary)
    except SyntaxError as exc:
        logger.exception("notebook_generator produced invalid code")
        return ToolResult(tool_name=TOOL_NAME, success=False, payload={}, summary="", error=f"The notebook failed its code check: {exc}")
    except Exception as exc:
        logger.exception("notebook_generator failed")
        return ToolResult(tool_name=TOOL_NAME, success=False, payload={}, summary="", error=str(exc))
