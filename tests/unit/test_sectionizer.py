"""Section classification, tested against headings taken from the real corpus."""

from __future__ import annotations

import pytest

from myscholar.adapters.fakes import FakeChatModel
from myscholar.domain.enums import DocumentKind, SectionKind
from myscholar.ingest.sectionizer import (
    HeadingClassification,
    Sectionizer,
    classify_by_pattern,
    strip_enumeration,
)


class TestStripEnumeration:
    @pytest.mark.parametrize(
        ("heading", "expected"),
        [
            ("I. INTRODUCTION", "INTRODUCTION"),
            ("VII. EXPERIMENTAL RESULTS", "EXPERIMENTAL RESULTS"),
            ("3.1 Results", "Results"),
            ("B. ESTIMATION", "ESTIMATION"),
            ("IV - Three Leg Six Switch Inverter", "Three Leg Six Switch Inverter"),
            ("Abstract", "Abstract"),
        ],
    )
    def test_leading_numbering_is_removed(self, heading: str, expected: str) -> None:
        assert strip_enumeration(heading) == expected


class TestPatternClassification:
    """Headings below are copied verbatim from the extraction spike."""

    @pytest.mark.parametrize(
        ("heading", "expected"),
        [
            ("I. INTRODUCTION", SectionKind.INTRODUCTION),
            ("II. LITERATURE REVIEW OF MPC", SectionKind.RELATED_WORK),
            ("II. RELATED WORK", SectionKind.RELATED_WORK),
            ("III. METHODOLOGY", SectionKind.METHODS),
            ("VII. EXPERIMENTAL RESULTS", SectionKind.RESULTS),
            ("VIII. DISCUSSION", SectionKind.DISCUSSION),
            ("IX. CONCLUSION", SectionKind.CONCLUSION),
            ("REFERENCES", SectionKind.REFERENCES),
            ("Abstract", SectionKind.ABSTRACT),
            ("V. EXPERIMENTAL SETUP", SectionKind.EXPERIMENTS),
            # Correctly caught by the "algorithm" keyword: pseudocode blocks
            # describe method, so METHODS is the right home for them.
            ("Algorithm 1 Pseudo Code for NSGA-II", SectionKind.METHODS),
        ],
    )
    def test_conventional_headings_match(self, heading: str, expected: SectionKind) -> None:
        assert classify_by_pattern(heading, DocumentKind.PAPER) is expected

    @pytest.mark.parametrize(
        "heading",
        [
            "III. DISCRETE MODEL OF THREE PHASE INDUCTION MOTOR",
            "IV. THREE LEG SIX SWITCH INVERTER",
            "B. TOPSIS FOR BEST COMPROMISED SOLUTION",
            "E. CONTROL LAW APPLICATION",
        ],
    )
    def test_domain_specific_headings_do_not_match(self, heading: str) -> None:
        """The finding that makes the model fallback necessary.

        These are all genuine headings from the corpus. Every one is a methods
        section, and none contains a word a generic matcher looks for. Without
        a fallback they would all be filed as OTHER, and section filtering
        would be useless on exactly the papers it matters most for.
        """
        assert classify_by_pattern(heading, DocumentKind.PAPER) is None

    def test_conclusion_wins_over_discussion_in_a_combined_heading(self) -> None:
        """Ordering in the pattern table is load-bearing, so pin it."""
        assert (
            classify_by_pattern("Discussion and Conclusion", DocumentKind.PAPER)
            is SectionKind.CONCLUSION
        )

    def test_related_work_is_not_swallowed_by_a_broader_pattern(self) -> None:
        assert classify_by_pattern("Related Work", DocumentKind.PAPER) is SectionKind.RELATED_WORK

    def test_cv_headings_use_the_cv_table(self) -> None:
        assert classify_by_pattern("Education", DocumentKind.CV) is SectionKind.CV_EDUCATION
        assert (
            classify_by_pattern("Journal Publications", DocumentKind.CV)
            is SectionKind.CV_PUBLICATIONS
        )

    def test_empty_heading_matches_nothing(self) -> None:
        assert classify_by_pattern("   ", DocumentKind.PAPER) is None


class TestSectionizer:
    async def test_pattern_match_avoids_calling_the_model(self) -> None:
        model = FakeChatModel()
        sectionizer = Sectionizer(model)

        result = await sectionizer.classify("I. INTRODUCTION")

        assert result.kind is SectionKind.INTRODUCTION
        assert result.matched_by == "pattern"
        assert model.calls == []

    async def test_unmatched_heading_goes_to_the_model(self) -> None:
        model = FakeChatModel(
            structured_responses=[HeadingClassification(kind=SectionKind.METHODS, confidence=0.9)]
        )
        sectionizer = Sectionizer(model)

        result = await sectionizer.classify(
            "III. DISCRETE MODEL OF THREE PHASE INDUCTION MOTOR",
            title="Weighting factors optimization of MPTC",
        )

        assert result.kind is SectionKind.METHODS
        assert result.matched_by == "model"

    async def test_model_failure_falls_back_rather_than_aborting(self) -> None:
        """One unclassifiable heading must not fail a whole ingestion run."""
        sectionizer = Sectionizer(FakeChatModel())  # no queued response, so it raises

        result = await sectionizer.classify("IV. THREE LEG SIX SWITCH INVERTER")

        assert result.kind is SectionKind.OTHER
        assert result.matched_by == "fallback"

    async def test_works_without_a_model_at_all(self) -> None:
        sectionizer = Sectionizer(None)

        assert (await sectionizer.classify("I. INTRODUCTION")).kind is SectionKind.INTRODUCTION
        assert (await sectionizer.classify("IV. SIX SWITCH INVERTER")).kind is SectionKind.OTHER

    async def test_stats_record_how_each_heading_was_resolved(self) -> None:
        """So a run can report how much work the fallback is doing."""
        model = FakeChatModel(
            structured_responses=[HeadingClassification(kind=SectionKind.METHODS, confidence=0.8)]
        )
        sectionizer = Sectionizer(model)

        await sectionizer.classify("I. INTRODUCTION")
        await sectionizer.classify("III. DISCRETE MODEL OF INDUCTION MOTOR")
        await sectionizer.classify("IV. THREE LEG SIX SWITCH INVERTER")

        assert sectionizer.stats == {"pattern": 1, "model": 1, "fallback": 1}
