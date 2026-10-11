"""Utility — builds a tool summary line by line without going over a token limit."""
from typing import Callable, List, Optional

from .token_manager import estimate_tokens


class SummaryBuilder:
    """Collects summary lines, most important first, and stops before the token limit.

    Usage:
        builder = SummaryBuilder(max_tokens=1000)
        builder.add("Dataset: 340 rows x 5 columns")
        builder.add_items(column_lines, more=lambda n: f"... and {n} more columns")
        summary = builder.text()
    """

    def __init__(self, max_tokens: int) -> None:
        self.max_tokens = max_tokens
        self.lines: List[str] = []
        self.used = 0

    def _cost(self, line: str) -> int:
        return estimate_tokens(line) + 1  # +1 for the newline

    def add(self, line: str) -> bool:
        """Add one line if it fits. Returns whether it was added."""
        cost = self._cost(line)
        if self.used + cost > self.max_tokens:
            return False
        self.lines.append(line)
        self.used += cost
        return True

    def add_items(
        self,
        items: List[str],
        more: Callable[[int], str],
        header: Optional[str] = None,
    ) -> int:
        """Add a list of item lines in order, leaving room to say how many didn't fit.

        `more(n)` writes the closing line for n left-out items. `header` is only added
        if at least one item fits. Returns how many items were added.
        """
        if not items:
            return 0

        costs = [self._cost(item) for item in items]
        header_cost = self._cost(header) if header else 0

        # Everything fits: no closing line needed
        if self.used + header_cost + sum(costs) <= self.max_tokens:
            self._append([header] if header else [], header_cost)
            self._append(items, sum(costs))
            return len(items)

        # Otherwise keep room for the closing line, sized for the largest possible count
        reserve = self._cost(more(len(items)))
        if self.used + header_cost + costs[0] + reserve > self.max_tokens:
            # Not even one item fits: just say how many there are
            self.add(more(len(items)))
            return 0
        self._append([header] if header else [], header_cost)

        added = 0
        for item, cost in zip(items, costs):
            if self.used + cost + reserve > self.max_tokens:
                break
            self._append([item], cost)
            added += 1
        self.add(more(len(items) - added))
        return added

    def _append(self, lines: List[str], cost: int) -> None:
        self.lines.extend(lines)
        self.used += cost

    def text(self) -> str:
        return "\n".join(self.lines)
