"""Reconciling the corpus against publisher records, offline against a fake registry."""

from __future__ import annotations

import pytest

from myscholar.adapters.fakes import FakeMetadataResolver
from myscholar.ingest.sync import IndexedDocument, dois_in, normalise_doi, plan
from myscholar.ports.metadata import ResolvedMetadata


def _doc(**overrides: object) -> IndexedDocument:
    base = {
        "slug": "a-paper",
        "title": "A Paper",
        "authors": ("M. Arshad",),
        "year": 2020,
        "venue": "IEEE Access",
        "doi": "10.1109/example.2020.1",
        "cited_by": None,
    }
    return IndexedDocument(**{**base, **overrides})  # type: ignore[arg-type]


def _record(**overrides: object) -> ResolvedMetadata:
    base = {
        "doi": "10.1109/example.2020.1",
        "title": "A Paper",
        "authors": ("M. Arshad",),
        "year": 2020,
        "venue": "IEEE Access",
        "source": "crossref",
    }
    return ResolvedMetadata(**{**base, **overrides})  # type: ignore[arg-type]


class TestNormalisation:
    def test_case_is_ignored(self) -> None:
        # DOIs are case-insensitive but deposited in mixed case.
        assert normalise_doi("10.1109/TASE.2023.3293049") == normalise_doi(
            "10.1109/tase.2023.3293049"
        )

    @pytest.mark.parametrize(
        "raw",
        ["https://doi.org/10.1109/a.1", " 10.1109/a.1 ", "10.1109/a.1.", "10.1109/a.1;"],
    )
    def test_prefixes_and_trailing_punctuation_are_stripped(self, raw: str) -> None:
        assert normalise_doi(raw) == "10.1109/a.1"

    def test_dois_are_found_in_page_markup(self) -> None:
        html = '<a href="https://doi.org/10.1109/a.1">x</a><p>See 10.3390/A15120453.</p>'

        assert dois_in(html) == {"10.1109/a.1", "10.3390/a15120453"}


class TestCorrections:
    async def test_a_wrong_year_is_reported(self) -> None:
        resolver = FakeMetadataResolver({"10.1109/example.2020.1": _record(year=2022)})

        result = await plan([_doc(year=2015)], resolver=resolver)

        assert [str(c) for c in result.corrections[0].changes] == ["year: 2015 -> 2022"]

    async def test_an_agreeing_record_produces_nothing(self) -> None:
        resolver = FakeMetadataResolver({"10.1109/example.2020.1": _record()})

        result = await plan([_doc()], resolver=resolver)

        assert result.corrections == []
        assert result.is_empty

    async def test_punctuation_differences_are_not_corrections(self) -> None:
        # Publishers deposit "Two -Phase" where the PDF sets "Two-Phase".
        resolver = FakeMetadataResolver(
            {"10.1109/example.2020.1": _record(title="Advanced  Direct-Torque  CONTROL")}
        )

        result = await plan([_doc(title="Advanced Direct Torque Control")], resolver=resolver)

        assert result.corrections == []

    async def test_a_thin_record_never_blanks_a_known_field(self) -> None:
        resolver = FakeMetadataResolver(
            {"10.1109/example.2020.1": _record(venue=None, authors=(), year=None)}
        )

        result = await plan([_doc()], resolver=resolver)

        assert result.corrections == []

    async def test_citation_counts_are_picked_up(self) -> None:
        resolver = FakeMetadataResolver({"10.1109/example.2020.1": _record(cited_by=42)})

        result = await plan([_doc(cited_by=None)], resolver=resolver)

        assert result.corrections[0].fields == {"cited_by": 42}


class TestFailureIsolation:
    async def test_an_unreachable_registry_leaves_the_document_alone(self) -> None:
        resolver = FakeMetadataResolver(fail_on={"10.1109/example.2020.1"})

        result = await plan([_doc()], resolver=resolver)

        assert result.corrections == []
        assert result.unreachable[0][0] == "10.1109/example.2020.1"

    async def test_one_failure_does_not_stop_the_rest(self) -> None:
        resolver = FakeMetadataResolver(
            {"10.1109/b.2": _record(doi="10.1109/b.2", year=2024)},
            fail_on={"10.1109/example.2020.1"},
        )

        result = await plan(
            [_doc(), _doc(slug="b", doi="10.1109/b.2", year=2019)], resolver=resolver
        )

        assert len(result.unreachable) == 1
        assert [c.slug for c in result.corrections] == ["b"]

    async def test_an_unknown_doi_is_not_a_failure(self) -> None:
        result = await plan([_doc()], resolver=FakeMetadataResolver())

        assert result.unknown_to_registry == ["10.1109/example.2020.1"]
        assert result.unreachable == []

    async def test_a_document_without_a_doi_is_listed_not_looked_up(self) -> None:
        resolver = FakeMetadataResolver()

        result = await plan([_doc(doi=None)], resolver=resolver)

        assert result.without_doi == ["a-paper"]
        assert resolver.calls == []


class TestReconciliation:
    async def test_a_paper_on_the_site_with_no_pdf_is_reported(self) -> None:
        result = await plan(
            [_doc()],
            resolver=FakeMetadataResolver({"10.1109/example.2020.1": _record()}),
            published_dois=["10.1109/example.2020.1", "10.1109/NEW.2026.9"],
        )

        assert result.missing_from_index == ["10.1109/new.2026.9"]
        assert result.missing_from_site == []

    async def test_an_indexed_paper_absent_from_the_site_is_reported(self) -> None:
        result = await plan(
            [_doc()],
            resolver=FakeMetadataResolver({"10.1109/example.2020.1": _record()}),
            published_dois=["10.1109/other.1"],
        )

        assert result.missing_from_site == ["10.1109/example.2020.1"]

    async def test_no_published_list_means_no_reconciliation_claims(self) -> None:
        result = await plan(
            [_doc()], resolver=FakeMetadataResolver({"10.1109/example.2020.1": _record()})
        )

        assert result.missing_from_index == []
        assert result.missing_from_site == []
