"""Invariants a healthy corpus satisfies.

Ingestion is where this project has gone wrong most often, and every time it
went wrong quietly. These are not accuracy measurements — there are no
ground-truth section labels — but statements true of every well-formed document
here, so a violation means something upstream broke even where nobody can say
what the right answer was.

The check earned itself on its first run: re-ingesting the site pages replaced
their document rows and their summaries went with them, so the CV and every
page had silently dropped out of document selection.
"""

from __future__ import annotations

from myscholar.ingest.health import DocumentStats, Severity, check


def _paper(**overrides: object) -> DocumentStats:
    base = {
        "title": "A Paper",
        "kind": "paper",
        "pages": 10,
        "chunks": 120,
        "sections": 12,
        "unclassified_sections": 2,
        "excluded_sections": 0,
        "empty_chunks": 0,
        "has_summary": True,
        "verified": True,
        "doi": "10.1109/x.1",
    }
    return DocumentStats(**{**base, **overrides})  # type: ignore[arg-type]


def _messages(stats: list[DocumentStats], severity: Severity) -> list[str]:
    return [f.message for f in check(stats) if f.severity is severity]


class TestDensity:
    def test_a_repeating_text_layer_fails(self) -> None:
        """796 chunks from 6 pages, which sat in the index for four phases."""
        assert _messages([_paper(pages=6, chunks=796)], Severity.FAIL)

    def test_a_missing_text_layer_fails(self) -> None:
        """A scanned PDF extracts headers and nothing else, and indexes
        cleanly while answering nothing."""
        assert _messages([_paper(pages=16, chunks=6)], Severity.FAIL)

    def test_a_normal_paper_passes(self) -> None:
        assert not check([_paper()])


class TestStructure:
    def test_references_reaching_the_index_fail(self) -> None:
        """A quarter of a paper that answers nothing."""
        assert _messages([_paper(excluded_sections=2)], Severity.FAIL)

    def test_empty_chunks_fail(self) -> None:
        assert _messages([_paper(empty_chunks=3)], Severity.FAIL)

    def test_a_document_with_no_chunks_fails(self) -> None:
        assert _messages([_paper(chunks=0)], Severity.FAIL)

    def test_mostly_unclassified_sections_warn(self) -> None:
        """Section kind drives the boost branches, so this degrades routing
        without failing anything."""
        assert _messages([_paper(sections=10, unclassified_sections=9)], Severity.WARN)


class TestProvenance:
    def test_a_missing_summary_warns(self) -> None:
        """The finding that justified the whole check."""
        assert _messages([_paper(has_summary=False)], Severity.WARN)

    def test_unverified_metadata_warns(self) -> None:
        assert _messages([_paper(verified=False)], Severity.WARN)

    def test_a_missing_doi_warns(self) -> None:
        assert _messages([_paper(doi=None)], Severity.WARN)

    def test_a_duplicate_title_fails(self) -> None:
        """A re-ingest that did not replace, which produced cv/cv-2 and
        doubled that content in every search."""
        assert _messages([_paper(), _paper()], Severity.FAIL)


class TestEmptyCorpus:
    def test_nothing_indexed_fails(self) -> None:
        assert check([])[0].severity is Severity.FAIL
