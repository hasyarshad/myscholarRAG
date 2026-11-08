"""Server-Sent Events framing: ``event:`` and ``data:`` lines, then a blank line."""

from __future__ import annotations

from pydantic import BaseModel

#: Headers that stop intermediaries buffering the stream. X-Accel-Buffering is
#: nginx-specific and harmless elsewhere; no-transform stops a CDN rewriting the body.
SSE_HEADERS = {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def frame(event: str, payload: BaseModel | str) -> str:
    """Encode one event.

    Newlines in the payload are split across ``data:`` lines; a raw newline
    would end the frame early.
    """
    body = payload if isinstance(payload, str) else payload.model_dump_json()
    lines = "\n".join(f"data: {line}" for line in body.split("\n"))
    return f"event: {event}\n{lines}\n\n"
