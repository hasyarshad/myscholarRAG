"""Server-side descriptions of what the pipeline retrieved, shown in the interface."""

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
        # Parent expansion often returns several spans from one paper.
        one = uuid4()

        assert describe_sources([(PAPER, one), (PAPER, one), (PAPER, one)]) == "1 paper"

    def test_distinct_papers_are_counted(self) -> None:
        assert describe_sources([(PAPER, uuid4()), (PAPER, uuid4())]) == "2 papers"

    def test_the_cv_is_named_not_counted(self) -> None:
        cv = uuid4()

        assert describe_sources([(CV, cv), (CV, cv)]) == "the CV"

    def test_mixed_sources_are_joined(self) -> None:
        assert (
            describe_sources([(PAPER, uuid4()), (PAPER, uuid4()), (CV, uuid4()), (PAGE, uuid4())])
            == "2 papers, the CV and the site"
        )

    def test_papers_come_first(self) -> None:
        assert describe_sources([(PAGE, uuid4()), (PAPER, uuid4())]) == "1 paper and the site"

    def test_nothing_retrieved_says_nothing(self) -> None:
        assert describe_sources([]) == ""

    def test_an_unknown_kind_is_described_rather_than_dropped(self) -> None:
        assert "dataset" in describe_sources([("dataset", uuid4())])


class TestClasses:
    @pytest.mark.parametrize("query_class", list(QueryClass))
    def test_every_class_has_a_phrase(self, query_class: QueryClass) -> None:
        phrase = describe_class(query_class, short_name="Haseeb")

        assert phrase
        assert "_" not in phrase, "the raw enum value leaked into the interface"

    def test_the_person_is_named(self) -> None:
        assert "Haseeb" in describe_class(QueryClass.BIOGRAPHICAL, short_name="Haseeb")
