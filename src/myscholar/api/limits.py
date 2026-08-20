"""Bounding the work a public endpoint will accept.

A GPU serves one question at a time. Without limits, one script exhausts the
machine and every other visitor gets nothing — so the queue is bounded and
callers over the limit are told to come back rather than left waiting.
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
    """The queue is full. Better to say so than to accept work and stall."""


#: Header carrying the secret the Next.js proxy injects.
SECRET_HEADER = "x-myscholar-secret"


def authorised(supplied: str | None, expected: str) -> bool:
    """Whether a request may proceed.

    An empty ``expected`` disables the check, which is how development runs —
    the alternative is a secret in every local shell, and a check people turn
    off by pasting a placeholder is worse than one that is honestly absent.

    Compared with ``compare_digest``. A plain ``==`` returns as soon as two
    bytes differ, so the time it takes reveals how much of the prefix was
    right, and a few thousand requests recover the secret one character at a
    time. The comparison here is short and over a network, which makes the
    signal small but not absent; constant time costs nothing, so there is no
    reason to leave it.
    """
    if not expected:
        return True
    return bool(supplied) and hmac.compare_digest(supplied or "", expected)


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

    #: How often the caller table is swept for entries with nothing left in them.
    #:
    #: Each caller's own deque was already pruned on the way past, but only for
    #: the caller making the request — the *table* keyed by caller was never
    #: pruned at all, and `forget` was written and then never called from
    #: anywhere. So every distinct caller left a permanent entry behind.
    #:
    #: That matters more than a slow leak usually would, because the key is
    #: taken from `X-Forwarded-For`, which the header's own docstring notes is
    #: client-controlled. A caller sending a different value each time therefore
    #: allocated a new permanent entry per request, against a container with a
    #: few hundred megabytes to its name. Spoofing the header already defeated
    #: the rate limit; it should not also have been the cheapest way to exhaust
    #: the process.
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
        """Drop callers with no activity inside the window.

        Amortised rather than scheduled: it runs on the request path, so there
        is no background task to own, and the cost is one pass over a table that
        only reaches a size worth sweeping if it is being filled.
        """
        stale = [caller for caller, hits in self._hits.items() if not hits or now - hits[-1] > 3600]
        for caller in stale:
            del self._hits[caller]

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
