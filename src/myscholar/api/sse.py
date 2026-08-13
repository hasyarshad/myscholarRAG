"""Server-Sent Events framing.

The wire format is trivial — ``event:`` and ``data:`` lines separated by a blank
line — and the parts that matter are the details that go wrong in production:
proxies buffering the response, and idle connections being closed.
"""

from __future__ import annotations

from pydantic import BaseModel

#: Headers that stop an intermediary holding the response and delivering it in
#: one piece, which is exactly what streaming exists to avoid.
#:
#: X-Accel-Buffering is nginx-specific and harmless elsewhere; no-transform
#: stops a CDN rewriting the body.
SSE_HEADERS = {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

#: A comment line. Keeps the connection alive without the client seeing an
#: event; anything above about ten seconds of silence risks a timeout in some
#: proxy along the way.
HEARTBEAT = ": keep-alive\n\n"


def frame(event: str, payload: BaseModel | str) -> str:
    """Encode one event.

    Newlines inside the payload have to be split across ``data:`` lines, or the
    frame terminates early and the client sees a truncated event.
    """
    body = payload if isinstance(payload, str) else payload.model_dump_json()
    lines = "\n".join(f"data: {line}" for line in body.split("\n"))
    return f"event: {event}\n{lines}\n\n"
