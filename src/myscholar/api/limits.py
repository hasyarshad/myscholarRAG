"""Bounds on the work the public endpoint accepts.

A per-caller rate limit and a bounded queue, so one client cannot exhaust the
service and callers over the limit are told to retry.
"""

from __future__ import annotations

import asyncio
import hmac
import time
from collections import defaultdict, deque
from dataclasses import dataclass


class RateLimitedError(Exception):
    """The caller has exceeded their allowance."""

    def __init__(self, retry_after: int) -> None:
        super().__init__(f"rate limited; retry after {retry_after}s")
        self.retry_after = retry_after


class OverloadedError(Exception):
    """The admission queue is full."""


#: Header carrying the secret the Next.js proxy injects.
SECRET_HEADER = "x-myscholar-secret"


def authorised(supplied: str | None, expected: str) -> bool:
    """Whether a request may proceed.

    An empty ``expected`` disables the check, as in development. ``compare_digest``
    runs in constant time: a plain ``==`` stops at the first differing byte, so
    response timing would leak how much of a guess was right.
    """
    if not expected:
        return True
    return bool(supplied) and hmac.compare_digest(supplied or "", expected)


@dataclass(frozen=True, slots=True)
class Limits:
    per_minute: int = 8
    per_hour: int = 40
    max_concurrent: int = 1
    """One question at a time; a local GPU model serves requests serially anyway."""

    max_queued: int = 6


class RateLimiter:
    """A sliding window per caller.

    In memory, so counts reset on restart and are not shared between processes.
    """

    #: Requests between sweeps of the caller table. The key comes from the
    #: client-controlled ``X-Forwarded-For``, so an unswept table could be grown
    #: without bound by varying it.
    _SWEEP_EVERY = 256

    def __init__(self, limits: Limits | None = None) -> None:
        self._limits = limits or Limits()
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._since_sweep = 0

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

        self._since_sweep += 1
        if self._since_sweep >= self._SWEEP_EVERY:
            self._since_sweep = 0
            self._sweep(now)

    def _sweep(self, now: float) -> None:
        """Drop callers with no hits in the last hour.

        Runs on the request path every ``_SWEEP_EVERY`` checks, so no background task is needed.
        """
        stale = [caller for caller, hits in self._hits.items() if not hits or now - hits[-1] > 3600]
        for caller in stale:
            del self._hits[caller]


class Gate:
    """Admission control: a bounded queue in front of the pipeline.

    A caller that would wait behind ``max_queued`` others is rejected at once
    and told to retry.
    """

    def __init__(self, limits: Limits | None = None) -> None:
        self._limits = limits or Limits()
        self._semaphore = asyncio.Semaphore(self._limits.max_concurrent)
        self._waiting = 0

    @property
    def queue_depth(self) -> int:
        return self._waiting

    async def __aenter__(self) -> Gate:
        # Only callers that would have to wait are rejected, so max_queued=0
        # still admits a caller when the slot is free.
        if self._semaphore.locked() and self._waiting >= self._limits.max_queued:
            raise OverloadedError
        self._waiting += 1
        try:
            await self._semaphore.acquire()
        finally:
            self._waiting -= 1
        return self

    async def __aexit__(self, *_: object) -> None:
        # Also runs when a client disconnects mid-stream, so an abandoned
        # request cannot hold the slot.
        self._semaphore.release()
