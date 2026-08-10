"""Bibliographic metadata extraction and the review cycle."""

from __future__ import annotations

from pathlib import Path

from myscholar.adapters.fakes import FakeChatModel
from myscholar.ingest.metadata import (
    ExtractedMetadata,
    MetadataExtractor,
    find_doi,
    find_year,
    read_review_file,
    slugify,
    write_review_file,
)

FIRST_PAGE = """
Weighting Factors Optimization of Model Predictive Torque Control of
Induction Motor Using NSGA-II With TOPSIS Decision Making

H. ARSHAD, M. KHAN, AND S. AHMED
IEEE Access, vol. 9, 2021
Digital Object Identifier 10.1109/ACCESS.2021.3078245
"""


class TestFindDoi:
    def test_extracts_a_doi(self) -> None:
        assert find_doi(FIRST_PAGE) == "10.1109/ACCESS.2021.3078245"

    def test_trailing_punctuation_is_dropped(self) -> None:
        assert find_doi("see doi 10.1109/TIE.2020.1234567.") == "10.1109/TIE.2020.1234567"

    def test_returns_none_when_absent(self) -> None:
        assert find_doi("no identifier here") is None


class TestFindYear:
    def test_most_frequent_year_wins(self) -> None:
        """Publication years repeat in header and footer; citation years vary."""
        text = "IEEE Access 2021 ... published 2021 ... citing work from 2015 and 2018"

        assert find_year(text) == 2021

    def test_returns_none_when_absent(self) -> None:
        assert find_year("no dates at all") is None

    def test_ignores_implausible_numbers(self) -> None:
        assert find_year("equation 3456 and value 1234") is None


class TestSlugify:
    def test_produces_a_url_safe_slug(self) -> None:
        assert slugify("A Chaos Based SVPWM Technique!", "fallback") == (
            "a-chaos-based-svpwm-technique"
        )

    def test_falls_back_when_the_title_is_empty(self) -> None:
        """The fallback is slugified too — slugs go in URLs and unique keys."""
        assert slugify("", "Some File Name") == "some-file-name"


class TestMetadataExtractor:
    async def test_uses_the_model_result(self) -> None:
        model = FakeChatModel(
            structured_responses=[
                ExtractedMetadata(
                    title="Weighting Factors Optimization of MPTC",
                    authors=["H. Arshad", "M. Khan"],
                    year=2021,
                    venue="IEEE Access",
                )
            ]
        )

        proposal = await MetadataExtractor(model).propose(
            first_page=FIRST_PAGE, source_path=Path("paper.pdf")
        )

        assert proposal.metadata.title == "Weighting Factors Optimization of MPTC"
        assert proposal.metadata.authors == ("H. Arshad", "M. Khan")
        assert proposal.metadata.venue == "IEEE Access"

    async def test_nothing_is_verified_on_extraction(self) -> None:
        """Verification is a human act; extraction only ever proposes."""
        model = FakeChatModel(structured_responses=[ExtractedMetadata(title="A paper", year=2021)])

        proposal = await MetadataExtractor(model).propose(
            first_page=FIRST_PAGE, source_path=Path("paper.pdf")
        )

        assert proposal.metadata.verified is False

    async def test_regex_doi_beats_the_model(self) -> None:
        """A pattern match against literal page text cannot hallucinate."""
        model = FakeChatModel(
            structured_responses=[
                ExtractedMetadata(title="A paper", doi="10.9999/INVENTED.0000000")
            ]
        )

        proposal = await MetadataExtractor(model).propose(
            first_page=FIRST_PAGE, source_path=Path("paper.pdf")
        )

        assert proposal.metadata.doi == "10.1109/ACCESS.2021.3078245"

    async def test_year_disagreement_is_flagged_for_review(self) -> None:
        model = FakeChatModel(structured_responses=[ExtractedMetadata(title="A paper", year=2019)])

        proposal = await MetadataExtractor(model).propose(
            first_page=FIRST_PAGE, source_path=Path("paper.pdf")
        )

        assert any("disagreement" in note for note in proposal.confidence_notes)

    async def test_model_failure_falls_back_to_the_filename(self) -> None:
        """One unreadable first page must not abort a twenty-paper run."""
        proposal = await MetadataExtractor(FakeChatModel()).propose(
            first_page="unparseable", source_path=Path("Some Paper Title.pdf")
        )

        assert proposal.metadata.title == "Some Paper Title"
        assert any("failed" in note for note in proposal.confidence_notes)

    async def test_missing_fields_are_flagged(self) -> None:
        proposal = await MetadataExtractor(None).propose(
            first_page="no useful content", source_path=Path("x.pdf")
        )

        assert "no year found" in proposal.confidence_notes
        assert "no authors extracted" in proposal.confidence_notes


class TestReviewFile:
    async def test_round_trips_through_yaml(self, tmp_path: Path) -> None:
        model = FakeChatModel(
            structured_responses=[
                ExtractedMetadata(
                    title="A paper", authors=["H. Arshad"], year=2021, venue="IEEE Access"
                )
            ]
        )
        proposal = await MetadataExtractor(model).propose(
            first_page=FIRST_PAGE, source_path=Path("paper.pdf")
        )
        target = tmp_path / "metadata.yaml"

        write_review_file([proposal], target)
        loaded = read_review_file(target)

        assert loaded["paper.pdf"].title == "A paper"
        assert loaded["paper.pdf"].year == 2021
        assert loaded["paper.pdf"].verified is False

    async def test_flagged_entries_are_listed_first(self, tmp_path: Path) -> None:
        """The entries most likely to be wrong get read while attention is fresh."""
        clean = await MetadataExtractor(
            FakeChatModel(
                structured_responses=[
                    ExtractedMetadata(title="Zebra paper", authors=["A. B"], year=2021)
                ]
            )
        ).propose(first_page=FIRST_PAGE, source_path=Path("z.pdf"))
        flagged = await MetadataExtractor(None).propose(
            first_page="nothing", source_path=Path("a.pdf")
        )
        target = tmp_path / "metadata.yaml"

        write_review_file([clean, flagged], target)

        body = target.read_text()
        assert body.index("a.pdf") < body.index("z.pdf")

    def test_reading_a_verified_file_marks_entries_verified(self, tmp_path: Path) -> None:
        target = tmp_path / "metadata.yaml"
        target.write_text(
            "- file: paper.pdf\n"
            "  title: A corrected title\n"
            "  authors: [H. Arshad]\n"
            "  year: 2021\n"
            "  venue: IEEE Access\n"
            "  doi: null\n"
            "  verified: true\n"
        )

        loaded = read_review_file(target)

        assert loaded["paper.pdf"].verified is True
        assert loaded["paper.pdf"].title == "A corrected title"

    def test_malformed_entries_are_skipped_not_fatal(self, tmp_path: Path) -> None:
        target = tmp_path / "metadata.yaml"
        target.write_text("- not_a_mapping\n- file: ok.pdf\n  title: Fine\n  verified: true\n")

        loaded = read_review_file(target)

        assert list(loaded) == ["ok.pdf"]
