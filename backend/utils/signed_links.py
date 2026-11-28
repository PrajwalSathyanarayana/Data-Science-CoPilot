"""Utility — download links that only the server can make, and that stop working after a while.

A link looks like /api/download/<report_id>/<file>?expires=<time>&sig=<signature>. The signature is an
HMAC-SHA256 of the report ID, the file name and the expiry time, made with a secret only the server knows.
Changing any part of the link, or using it after it expires, makes the check fail.
"""
import base64
import hashlib
import hmac
import time
from typing import Literal, Optional
from urllib.parse import quote

from ..config import DOWNLOAD_LINK_SECONDS, DOWNLOAD_SIGNING_KEY

LinkCheck = Literal["ok", "expired", "invalid"]


def _signature(report_id: str, filename: str, expires: int) -> str:
    message = f"{report_id}/{filename}/{expires}".encode()
    digest = hmac.new(DOWNLOAD_SIGNING_KEY.encode(), message, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).decode().rstrip("=")


def signed_link(report_id: str, filename: str, now: Optional[float] = None) -> str:
    expires = int(now if now is not None else time.time()) + DOWNLOAD_LINK_SECONDS
    sig = _signature(report_id, filename, expires)
    return f"/api/download/{report_id}/{quote(filename)}?expires={expires}&sig={sig}"


def check_link(report_id: str, filename: str, expires: Optional[int], sig: Optional[str], now: Optional[float] = None) -> LinkCheck:
    if expires is None or not sig:
        return "invalid"
    # Compared in constant time, so the response time never hints at how much of a guess was right
    if not hmac.compare_digest(_signature(report_id, filename, expires), sig):
        return "invalid"
    if (now if now is not None else time.time()) > expires:
        return "expired"
    return "ok"
