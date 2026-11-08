"""Crossref, the registry that minted most of this corpus's DOIs.

Free and keyless. A contact address in the User-Agent puts requests in the faster
"polite pool". All 20 DOIs in the corpus resolve; Crossref corrected three years,
because a PDF's first page often shows the conference, copyright or online-first year.
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

#: Types that are not the version of record. A preprint deposit carries an
#: earlier year than the published paper.
_PREPRINT_TYPES = frozenset({"posted-content"})


def _year_of(message: dict[str, Any]) -> int | None:
    """The year of record, from the issue or publication dates.

    ``created`` (when Crossref first saw the record) is not used: for an
    online-first article it can be a year early.
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


#: Inline markup deposited inside titles: <i>, <sub>, <scp> and the JATS variants.
_TAGS = re.compile(r"</?[a-zA-Z][^>]{0,40}>")


def _clean(text: str) -> str:
    """Strip tags and unescape entities; Crossref titles are XML fragments ("THD &amp; EMI")."""
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
        self._max_attempts = max_attempts
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=timeout_seconds,
            # Crossref's polite pool. The address is a contact, not authentication.
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
            # Without a title the record cannot correct anything, so the
            # extracted metadata is kept.
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

        A 404 (a DOI Crossref does not have yet) returns None without retrying,
        and the caller falls back to extraction.
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
