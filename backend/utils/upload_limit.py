"""Utility — refuses an upload that says it is too big, before any of it is received.

Every browser states the size of what it sends (the Content-Length header). Checking it first means an
oversized upload is turned away straight away, instead of being received and stored on disk only to be
rejected. A client that leaves the size out is still stopped by the route, which counts while it reads.
"""
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


class UploadSizeLimit:
    def __init__(self, app: ASGIApp, path: str, max_body_bytes: int) -> None:
        self.app = app
        self.path = path
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["method"] == "POST" and scope["path"] == self.path:
            stated = dict(scope["headers"]).get(b"content-length")
            if stated is not None:
                if not stated.isdigit():
                    await JSONResponse({"detail": "The upload's size header is not valid."}, status_code=400)(scope, receive, send)
                    return
                if int(stated) > self.max_body_bytes:
                    response = JSONResponse({"detail": "The file is too large. The limit is 50 MB."}, status_code=413)
                    await response(scope, receive, send)
                    return
        await self.app(scope, receive, send)
