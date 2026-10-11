"""Tests for backend/utils/summary_builder.py."""
from backend.utils.summary_builder import SummaryBuilder
from backend.utils.token_manager import estimate_tokens


def more(n: int) -> str:
    return f"... and {n} more"


def test_everything_fits():
    builder = SummaryBuilder(max_tokens=1000)
    builder.add("Header")
    added = builder.add_items(["- a", "- b"], more=more, header="Items:")

    assert added == 2
    assert builder.text() == "Header\nItems:\n- a\n- b"


def test_stops_before_limit_and_says_how_many_were_left_out():
    builder = SummaryBuilder(max_tokens=40)
    items = [f"- item number {i} with some words" for i in range(20)]
    added = builder.add_items(items, more=more)

    lines = builder.text().splitlines()
    assert 0 < added < 20
    assert lines[-1] == f"... and {20 - added} more"
    assert estimate_tokens(builder.text()) <= 40


def test_header_is_skipped_when_no_item_fits():
    builder = SummaryBuilder(max_tokens=12)
    builder.add_items(["- a very long item line that cannot possibly fit"], more=more, header="Items:")

    assert builder.text() == "... and 1 more"


def test_last_item_may_use_the_room_kept_for_the_closing_line():
    # Exactly enough room for both items but not for two items + a closing line
    items = ["- a", "- b"]
    exact = sum(estimate_tokens(i) + 1 for i in items)
    builder = SummaryBuilder(max_tokens=exact)

    assert builder.add_items(items, more=more) == 2
    assert builder.text() == "- a\n- b"


def test_add_refuses_a_line_that_does_not_fit():
    builder = SummaryBuilder(max_tokens=3)
    assert builder.add("short") is True
    assert builder.add("this line is far too long to fit") is False
    assert builder.text() == "short"


def test_empty_item_list_adds_nothing():
    builder = SummaryBuilder(max_tokens=100)
    assert builder.add_items([], more=more, header="Items:") == 0
    assert builder.text() == ""
