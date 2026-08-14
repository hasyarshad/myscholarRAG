"""Crossref — the registry that mints most of this corpus's DOIs.

Free, no key, no account. The one courtesy it asks is a contact address in the
User-Agent, which puts requests in the "polite pool" and gets them served
ahead of anonymous traffic. Omitting it is not an error, so it is easy to skip
and then wonder why lookups are slow.

Measured against this corpus: all 20 DOIs resolve, and the returned titles,
authors and venues agree with the reviewed file. Three of the years do not —
Crossref is right in all three cases, since the year on a PDF's first page is
frequently the conference year, the copyright year or the online-first year
rather than the year of record.
"""

from __future__ import annotations

import asyncio
import logging
import re
from html import unescape
from typing import Any
from urllib.parse import quote

import httpx

from myscholar.ports.metadata import (
    MetadataResolutionError,
    MetadataResolver,
    ResolvedMetadata,
)

logger = logging.getLogger(__name__)

#: Types that are not the version of record. A paper is routinely deposited
#: twice — once as a preprint, once by the publisher — and the preprint
#: carries an earlier year, which is exactly the drift being corrected here.
_PREPRINT_TYPES = frozenset({"posted-content"})


def _year_of(message: dict[str, Any]) -> int | None:
    """The year of record, preferring the print/online issue date.

    ``issued`` is the date the work was issued in its final form. ``created``
    is when Crossref first saw the record, which for an online-first article
    can be a year earlier and is the wrong answer to "when was this
    published".
    """
    for key in ("issued", "published-print", "published-online", "published"):
        parts = (message.get(key) or {}).get("date-parts") or []
        if parts and parts[0] and parts[0][0]:
            return int(parts[0][0])
    return None


def _authors_of(message: dict[str, Any]) -> tuple[str, ...]:
    """Author names as "F. Surname", matching how the corpus already stores them."""
    names: list[str] = []
    for author in message.get("author") or []:
        family = (author.get("family") or "").strip()
        given = (author.get("given") or "").strip()
        if family and given:
            initials = " ".join(f"{part[0]}." for part in given.split() if part)
            names.append(f"{initials} {family}")
        elif family:
            names.append(family)
        elif name := (author.get("name") or "").strip():
            # Consortium and group authors have no family name.
            names.append(name)
    return tuple(names)


#: Inline markup publishers deposit inside titles — <i>, <sub>, <scp> and the
#: JATS variants. Kept as text would put literal tags in a citation.
_TAGS = re.compile(r"</?[a-zA-Z][^>]{0,40}>")


def _clean(text: str) -> str:
    """Undo the markup a title is deposited with.

    Crossref stores titles as XML fragments, so "THD & EMI" arrives as
    "THD &amp; EMI" and italicised terms arrive wrapped in tags. Written
    straight into the corpus, the first correction this made would have
    replaced a correct title with an escaped one.
    """
    return unescape(_TAGS.sub("", text)).strip()


def _first(values: Any) -> str | None:
    """Crossref returns title and container-title as lists, occasionally empty."""
    if isinstance(values, list) and values:
        text = _clean(str(values[0]))
        return text or None
    return None


class CrossrefResolver(MetadataResolver):
    def __init__(
        self,
        *,
        mailto: str,
        base_url: str = "https://api.crossref.org",
        timeout_seconds: float = 20.0,
        max_attempts: int = 3,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._max_attempts = max_attempts
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=timeout_seconds,
            # The polite pool. The address is a contact for Crossref's
            # operators, not authentication.
            headers={"User-Agent": f"myscholar/0.1 (mailto:{mailto})"},
            follow_redirects=True,
        )

    async def resolve(self, doi: str) -> ResolvedMetadata | None:
        cleaned = doi.strip().rstrip(".,;").removeprefix("https://doi.org/")
        if not cleaned:
            return None

        url = f"{self._base_url}/works/{quote(cleaned, safe='')}"
        message = await self._get(url, cleaned)
        if message is None:
            return None

        if str(message.get("type", "")) in _PREPRINT_TYPES:
            logger.info("%s resolves to a preprint record, not the version of record", cleaned)

        title = _first(message.get("title"))
        if not title:
            # A record with no title is unusable as a correction, and
            # overwriting a good title with an empty one would be worse than
            # leaving the extracted metadata alone.
            return None

        return ResolvedMetadata(
            doi=str(message.get("DOI") or cleaned),
            title=title,
            authors=_authors_of(message),
            year=_year_of(message),
            venue=_first(message.get("container-title")),
            cited_by=message.get("is-referenced-by-count"),
            source="crossref",
        )

    async def _get(self, url: str, doi: str) -> dict[str, Any] | None:
        """One lookup, retried on transport and 5xx errors.

        404 means Crossref does not have the DOI, which is a normal outcome
        for a freshly minted one and must not be retried or raised — the
        caller falls back to extraction.
        """
        last: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = await self._client.get(url)
            except httpx.HTTPError as exc:
                last = exc
            else:
                if response.status_code == httpx.codes.NOT_FOUND:
                    return None
                if response.status_code < httpx.codes.INTERNAL_SERVER_ERROR:
                    response.raise_for_status()
                    payload = response.json()
                    message = payload.get("message")
                    return message if isinstance(message, dict) else None
                last = httpx.HTTPStatusError(
                    f"HTTP {response.status_code}", request=response.request, response=response
                )

            if attempt < self._max_attempts:
                await asyncio.sleep(0.5 * 2 ** (attempt - 1))

        msg = f"could not resolve {doi} via Crossref after {self._max_attempts} attempts: {last}"
        raise MetadataResolutionError(msg) from last

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
