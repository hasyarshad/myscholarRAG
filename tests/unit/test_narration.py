"""Describing what the pipeline did, for the interface.

Server-side because only the server knows. The widget's own mapping claimed
every retrieval searched the papers, which stopped being true the moment the CV
was indexed — and stayed on screen for months because nothing tested a label.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from myscholar.domain.enums import QueryClass
from myscholar.domain.narration import describe_class, describe_sources

PAPER = "paper"
CV = "cv"
PAGE = "page"


class TestSources:
    def test_papers_are_counted_by_document_not_by_passage(self) -> None:
        """Parent expansion routinely returns several spans from one paper.

        Counting passages would report "3 papers" from a single source — wrong
        in the direction that flatters, which is the direction to be most
        careful about.
        """
        one = uuid4()

        assert describe_sources([(PAPER, one), (PAPER, one), (PAPER, one)]) == "1 paper"

    def test_distinct_papers_are_counted(self) -> None:
        assert describe_sources([(PAPER, uuid4()), (PAPER, uuid4())]) == "2 papers"

    def test_the_cv_is_named_not_counted(self) -> None:
        """There is one CV, so "1 the CV" would read like a bug and would be
        one more number available to be wrong."""
        cv = uuid4()

        assert describe_sources([(CV, cv), (CV, cv)]) == "the CV"

    def test_mixed_sources_are_joined(self) -> None:
        assert (
            describe_sources([(PAPER, uuid4()), (PAPER, uuid4()), (CV, uuid4()), (PAGE, uuid4())])
            == "2 papers, the CV and the site"
        )

    def test_papers_come_first(self) -> None:
        """Ordering is fixed rather than following the result set, so the same
        mix of sources always reads the same way."""
        assert describe_sources([(PAGE, uuid4()), (PAPER, uuid4())]) == "1 paper and the site"

    def test_nothing_retrieved_says_nothing(self) -> None:
        """The caller appends this to a sentence, so an empty result must not
        produce a dangling "from"."""
        assert describe_sources([]) == ""

    def test_an_unknown_kind_is_described_rather_than_dropped(self) -> None:
        """A new document type should look vague, not invisible."""
        assert "dataset" in describe_sources([("dataset", uuid4())])


class TestClasses:
    @pytest.mark.parametrize("query_class", list(QueryClass))
    def test_every_class_has_a_phrase(self, query_class: QueryClass) -> None:
        phrase = describe_class(query_class, short_name="Haseeb")

        assert phrase
        assert "_" not in phrase, "the raw enum value leaked into the interface"

    def test_the_person_is_named(self) -> None:
        assert "Haseeb" in describe_class(QueryClass.BIOGRAPHICAL, short_name="Haseeb")
