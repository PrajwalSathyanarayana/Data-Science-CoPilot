"""Shared helpers for smoke scripts — print EXPECTED vs ACTUAL for each check and report a final result."""
import sys
from typing import Any, List


class SmokeRunner:
    """Collects checks, prints each one as it runs, and exits non-zero if any failed."""

    def __init__(self, title: str) -> None:
        self.title = title
        self.failures: List[str] = []
        self.total = 0
        print(f"\n=== Smoke test: {title} ===")

    def section(self, name: str) -> None:
        print(f"\n--- {name} ---")

    def check(self, label: str, expected: Any, actual: Any) -> None:
        self.total += 1
        passed = expected == actual
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {label}")
        print(f"         expected: {expected}")
        print(f"         actual:   {actual}")
        if not passed:
            self.failures.append(label)

    def finish(self) -> None:
        passed = self.total - len(self.failures)
        print(f"\n=== {passed}/{self.total} checks passed ===")
        if self.failures:
            print("Failed checks:")
            for label in self.failures:
                print(f"  - {label}")
            sys.exit(1)
