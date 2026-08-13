"""Bounding the work a public endpoint will accept.

A GPU serves one question at a time. Without limits, one script exhausts the
machine and every other visitor gets nothing — so the queue is bounded and
callers over the limit are told to come back rather than left waiting.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass


class RateLimitedError(Exception):
    """The caller has exceeded their allowance."""

    def __init__(self, retry_after: int) -> None:
        super().__init__(f"rate limited; retry after {retry_after}s")
        self.retry_after = retry_after


class OverloadedError(Exception):
    """The queue is full. Better to say so than to accept work and stall."""


@dataclass(frozen=True, slots=True)
class Limits:
    per_minute: int = 8
    per_hour: int = 40
    max_concurrent: int = 1
    """One question at a time: generation is serialised on the GPU anyway, and
    pretending otherwise just moves the queue somewhere less visible."""

    max_queued: int = 6


class RateLimiter:
    """A sliding window per caller.

    In-memory, so it resets on restart and does not survive multiple processes.
    That is the right trade for a single-process portfolio service; anything
    larger wants the counter in Postgres or at the edge.
    """

    def __init__(self, limits: Limits | None = None) -> None:
        self._limits = limits or Limits()
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, caller: str) -> None:
        now = time.monotonic()
        hits = self._hits[caller]
        while hits and now - hits[0] > 3600:
            hits.popleft()

        recent = sum(1 for t in hits if now - t <= 60)
        if recent >= self._limits.per_minute:
            raise RateLimitedError(retry_after=60)
        if len(hits) >= self._limits.per_hour:
            raise RateLimitedError(retry_after=600)

        hits.append(now)

    def forget(self, caller: str) -> None:
        self._hits.pop(caller, None)


class Gate:
    """Admission control for the GPU.

    Rejecting at the door beats accepting and stalling: a visitor told to retry
    in a moment has a better experience than one watching a spinner behind five
    other questions.
    """

    def __init__(self, limits: Limits | None = None) -> None:
        self._limits = limits or Limits()
        self._semaphore = asyncio.Semaphore(self._limits.max_concurrent)
        self._waiting = 0

    @property
    def queue_depth(self) -> int:
        return self._waiting

    async def __aenter__(self) -> Gate:
        # Rejection applies only to callers that would have to *wait*. Checking
        # the queue unconditionally meant max_queued=0 rejected everything,
        # including the first caller with the slot free.
        if self._semaphore.locked() and self._waiting >= self._limits.max_queued:
            raise OverloadedError
        self._waiting += 1
        try:
            await self._semaphore.acquire()
        finally:
            self._waiting -= 1
        return self

    async def __aexit__(self, *_: object) -> None:
        # Released in __aexit__ rather than after the response is built, so a
        # client disconnecting mid-stream frees the slot immediately. Otherwise
        # abandoning requests would be a denial of service.
        self._semaphore.release()
