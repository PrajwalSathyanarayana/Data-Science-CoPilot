"""Utility — makes text from the LLM or the upload safe to place in Markdown (notebooks, reports)."""
import re

IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")  # ![alt](address)
LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")  # [words](address)
REFERENCE_DEFINITION = re.compile(r"^\s*\[[^\]]+\]:\s*\S+.*$", re.MULTILINE)  # [1]: https://...
HTML_TAG = re.compile(r"<[^>]*>")  # <img ...>, <a ...>, <script>, <https://...>
BARE_URL = re.compile(r"\b(?:https?://|www\.)[^\s`]+", re.IGNORECASE)


def to_plain_markdown(text: str) -> str:
    """Remove anything that could load or point to an outside address when the Markdown is shown.

    Images are dropped, links keep their words, HTML tags are removed, and bare web addresses
    become inline code so they're shown as text, not as clickable links. Text from the LLM can be
    steered by the uploaded file (prompt injection), so it must never be able to send data anywhere.
    """
    text = IMAGE.sub("", text)
    text = LINK.sub(r"\1", text)
    text = REFERENCE_DEFINITION.sub("", text)
    text = HTML_TAG.sub("", text)
    return BARE_URL.sub(lambda match: f"`{match.group(0)}`", text)
