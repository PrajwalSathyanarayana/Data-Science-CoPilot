"""Measure how far our token estimate (characters / CHARS_PER_TOKEN) is from Claude's real token count.

Uses Anthropic's free token-counting endpoint: no model runs and nothing is billed, but the prompt text
(built from the sample files only) is sent to the API with the key in .env. Run from the project root:
    python -m scripts.measure_token_estimate

It counts the exact planner inputs for several sample uploads on both models, plus a near-empty message
to show any fixed overhead, then fits real = characters / k + overhead and suggests a CHARS_PER_TOKEN.
"""
import logging
from typing import List, Tuple

import anthropic
import numpy as np
import pandas as pd

from backend.agents.planner import create_plan, detect_interesting_columns
from backend.config import ANALYSIS_MODEL, CHARS_PER_TOKEN, CRITIC_MODEL
from backend.tools.dataset_inspector import inspect_dataset
from backend.utils.file_handler import LoadedFile, load_file
from backend.utils.token_manager import estimate_tokens
from tests.llm_fakes import ScriptedClient, make_response

SAMPLES = ["weather_ddmm.csv", "hospital.xlsx", "survey_with_gaps.csv", "shop_orders.csv"]
LIVE_WEATHER_INPUT_TOKENS = 1259  # usage.input_tokens from the first real planner call


def wide_file() -> LoadedFile:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({f"spend_{i:02d}": rng.lognormal(3, 1, 1000).round(2) for i in range(60)})
    df["churned"] = np.where(rng.random(1000) < 0.06, "yes", "no")
    return LoadedFile(df=df, file_type="csv", original_columns=list(df.columns))


def planner_input(loaded: LoadedFile) -> Tuple[str, str]:
    """The exact (system prompt, user message) create_plan would send, captured by a fake client."""
    client = ScriptedClient([make_response("", stop_reason="refusal")])  # one call, then the fallback
    create_plan(loaded, inspect_dataset(loaded.df), detect_interesting_columns(loaded.df), client=client)
    request = client.requests[0]
    return request["system"], request["messages"][0]["content"]


def real_count(client: anthropic.Anthropic, model: str, system: str, user: str) -> int:
    kwargs = {"model": model, "messages": [{"role": "user", "content": user}]}
    if system:
        kwargs["system"] = system
    return client.messages.count_tokens(**kwargs).input_tokens


def main() -> None:
    logging.disable(logging.WARNING)  # the capture run ends in a fallback on purpose
    client = anthropic.Anthropic()

    inputs: List[Tuple[str, str, str]] = [("near-empty message", "", "x")]
    for name in SAMPLES:
        with open(f"tests/sample_data/{name}", "rb") as handle:
            inputs.append((name, *planner_input(load_file(handle.read(), name))))
    inputs.append(("wide file (60 + churned)", *planner_input(wide_file())))

    print(f"{'input':<26}{'chars':>8}{'estimate':>10}{'opus real':>11}{'ratio':>7}{'haiku real':>12}{'ratio':>7}")
    rows = []
    for label, system, user in inputs:
        chars = len(system) + len(user)
        estimate = estimate_tokens(system) + estimate_tokens(user)
        opus = real_count(client, ANALYSIS_MODEL, system, user)
        haiku = real_count(client, CRITIC_MODEL, system, user)
        rows.append((label, chars, opus, haiku))
        print(f"{label:<26}{chars:>8,}{estimate:>10,}{opus:>11,}{opus / estimate:>7.2f}{haiku:>12,}{haiku / estimate:>7.2f}")

    # real = chars * slope + overhead, fitted on the planner inputs (the near-empty one shows the overhead)
    chars = np.array([r[1] for r in rows[1:]], dtype=float)
    for model_label, column in (("opus", 2), ("haiku", 3)):
        real = np.array([r[column] for r in rows[1:]], dtype=float)
        slope, overhead = np.polyfit(chars, real, 1)
        print(f"\n{model_label}: real = chars / {1 / slope:.2f} + {overhead:.0f}   "
              f"(near-empty message: {rows[0][column]} tokens; current CHARS_PER_TOKEN = {CHARS_PER_TOKEN})")

    weather_opus = next(r[2] for r in rows if r[0] == "weather_ddmm.csv")
    print(f"\nweather planner input: count_tokens {weather_opus:,} vs the live call's usage.input_tokens "
          f"{LIVE_WEATHER_INPUT_TOKENS:,} (difference = what the real request adds: thinking setup, fallback beta)")


if __name__ == "__main__":
    main()
