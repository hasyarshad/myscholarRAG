"""The Crossref adapter against a stubbed transport.

httpx's MockTransport runs the real client code (retries, status handling, headers)
against scripted responses, with no network.
"""

from __future__ import annotations

import httpx
import pytest

from myscholar.adapters.metadata import CrossrefResolver
from myscholar.ports.metadata import MetadataResolutionError

WORK = {
    "DOI": "10.1109/tase.2023.3293049",
    "title": ["Implications of the Sensorless Predictive Control for LSPMSM"],
    "container-title": ["IEEE Transactions on Automation Science and Engineering"],
    "issued": {"date-parts": [[2024, 3]]},
    "created": {"date-parts": [[2023, 7, 5]]},
    "author": [
        {"given": "Muhammad Haseeb", "family": "Arshad"},
        {"given": "Mohamed A.", "family": "Abido"},
    ],
    "is-referenced-by-count": 17,
    "type": "journal-article",
}


def _resolver(handler: object, **kwargs: object) -> CrossrefResolver:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    client = httpx.AsyncClient(transport=transport, base_url="https://api.crossref.org")
    return CrossrefResolver(mailto="test@example.com", client=client, **kwargs)  # type: ignore[arg-type]


class TestParsing:
    async def test_the_record_is_mapped(self) -> None:
        resolver = _resolver(lambda _: httpx.Response(200, json={"message": WORK}))

        record = await resolver.resolve("10.1109/TASE.2023.3293049")

        assert record is not None
        assert record.year == 2024
        assert record.venue == "IEEE Transactions on Automation Science and Engineering"
        assert record.cited_by == 17
        assert record.source == "crossref"

    async def test_authors_become_initialled_surnames(self) -> None:
        # The corpus stores names this way, so a reformatted name is not reported as a change.
        resolver = _resolver(lambda _: httpx.Response(200, json={"message": WORK}))

        record = await resolver.resolve("10.1109/tase.2023.3293049")

        assert record is not None
        assert record.authors == ("M. H. Arshad", "M. A. Abido")

    async def test_the_issue_date_is_preferred_over_the_deposit_date(self) -> None:
        # ``created`` is when Crossref first saw the record, a year early for an
        # online-first article.
        resolver = _resolver(lambda _: httpx.Response(200, json={"message": WORK}))

        record = await resolver.resolve("10.1109/tase.2023.3293049")

        assert record is not None
        assert record.year == 2024

    async def test_a_group_author_without_a_surname_is_kept(self) -> None:
        work = {**WORK, "author": [{"name": "The IEEE PES Task Force"}]}
        resolver = _resolver(lambda _: httpx.Response(200, json={"message": work}))

        record = await resolver.resolve("10.1109/x.1")

        assert record is not None
        assert record.authors == ("The IEEE PES Task Force",)

    @pytest.mark.parametrize(
        ("deposited", "expected"),
        [
            ("THD &amp; EMI improvement", "THD & EMI improvement"),
            ("Effect of <i>k</i><sub>p</sub> on flow", "Effect of kp on flow"),
            ("H&#x2082;O cooling", "H₂O cooling"),
        ],
    )
    async def test_deposited_markup_is_undone(self, deposited: str, expected: str) -> None:
        # Titles are deposited as XML fragments.
        work = {**WORK, "title": [deposited]}
        resolver = _resolver(lambda _: httpx.Response(200, json={"message": work}))

        record = await resolver.resolve("10.1/x")

        assert record is not None
        assert record.title == expected

    async def test_a_record_without_a_title_is_unusable(self) -> None:
        resolver = _resolver(lambda _: httpx.Response(200, json={"message": {"DOI": "10.1/x"}}))

        assert await resolver.resolve("10.1/x") is None


class TestRequests:
    async def test_the_polite_pool_header_is_sent(self) -> None:
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.headers.get("user-agent", ""))
            return httpx.Response(200, json={"message": WORK})

        transport = httpx.MockTransport(handler)
        client = httpx.AsyncClient(
            transport=transport, headers={"User-Agent": "myscholar/0.1 (mailto:a@b.c)"}
        )
        await CrossrefResolver(mailto="a@b.c", client=client).resolve("10.1/x")

        assert "mailto:a@b.c" in seen[0]

    async def test_a_doi_with_a_slash_is_escaped_into_the_path(self) -> None:
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            return httpx.Response(200, json={"message": WORK})

        await _resolver(handler).resolve("10.1109/a/b")

        assert seen[0].endswith("/works/10.1109%2Fa%2Fb")


class TestFailures:
    async def test_a_missing_doi_returns_nothing_without_retrying(self) -> None:
        # 404 is a normal outcome for a newly minted DOI.
        calls = 0

        def handler(_: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(404)

        assert await _resolver(handler).resolve("10.1/unknown") is None
        assert calls == 1

    async def test_a_server_error_is_retried_then_raised(self) -> None:
        calls = 0

        def handler(_: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(503)

        with pytest.raises(MetadataResolutionError):
            await _resolver(handler, max_attempts=2).resolve("10.1/x")

        assert calls == 2

    async def test_a_transient_error_recovers(self) -> None:
        calls = 0

        def handler(_: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(503 if calls == 1 else 200, json={"message": WORK})

        record = await _resolver(handler, max_attempts=3).resolve("10.1/x")

        assert record is not None
        assert calls == 2

    async def test_an_empty_doi_is_not_a_request(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
            msg = "should not have been called"
            raise AssertionError(msg)

        assert await _resolver(handler).resolve("   ") is None
