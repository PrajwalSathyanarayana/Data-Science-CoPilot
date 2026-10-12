"""Tests for backend/utils/text_safety.py."""
import pytest

from backend.utils.text_safety import to_plain_markdown


@pytest.mark.parametrize(
    "text, expected",
    [
        ("North is highest ![](https://evil.example/t?d=116.8)", "North is highest "),
        ("see [the docs](https://evil.example)", "see the docs"),
        ("a <img src=x onerror=alert(1)> b", "a  b"),
        ("<https://evil.example>", ""),
        ("visit https://evil.example/x now", "visit `https://evil.example/x` now"),
        ("www.evil.example", "`www.evil.example`"),
        ("note [1]\n[1]: https://evil.example", "note [1]\n"),
        ("Plain text with 116.8 and 37.22", "Plain text with 116.8 and 37.22"),
    ],
    ids=["image", "link", "html", "autolink", "bare-url", "www", "reference", "plain"],
)
def test_outside_addresses_are_removed_or_made_inert(text, expected):
    assert to_plain_markdown(text) == expected
