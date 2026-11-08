"""Docling block classification against a stubbed document.

Docling is not invoked: a stub stands in for the document it returns, so these tests
cover how its labels are interpreted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from myscholar.adapters.extraction.docling_extractor import (
    DoclingExtractor,
    collapse_repeated_lines,
    is_plausible_title,
    looks_like_section_heading,
    normalise,
)
from myscholar.domain.enums import ChunkKind
from myscholar.ports.extractor import ExtractionError


@dataclass
class _Prov:
    page_no: int = 1


@dataclass
class _Item:
    label: str
    text: str
    prov: list[_Prov] = field(default_factory=lambda: [_Prov()])


@dataclass
class _Doc:
    items: list[_Item]
    pages: dict[int, object] = field(default_factory=lambda: {1: object()})
    tables: list[object] = field(default_factory=list)

    def iterate_items(self) -> list[tuple[_Item, int]]:
        return [(item, 0) for item in self.items]


class _StubConverter:
    def __init__(self, doc: _Doc) -> None:
        self._doc = doc

    def convert(self, _path: str) -> Any:
        return type("Result", (), {"document": self._doc})()


def _extractor(items: list[_Item], **kwargs: Any) -> DoclingExtractor:
    extractor = DoclingExtractor(**kwargs)
    extractor._converter = _StubConverter(_Doc(items))  # noqa: SLF001
    return extractor


@pytest.fixture
def pdf(tmp_path: Path) -> Path:
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF-1.4 stub")
    return path


class TestTitleHandling:
    def test_the_title_does_not_become_a_section_heading(self, pdf: Path) -> None:
        extractor = _extractor(
            [
                _Item("title", "A Chaos Based SVPWM Technique for B4 Inverters"),
                _Item("text", "Abstract— Based on chaos theory, a hybrid SVPWM is proposed."),
                _Item("section_header", "I. INTRODUCTION"),
                _Item("text", "Motor drives are widely used."),
            ]
        )

        result = extractor.extract(pdf)

        assert result.headings == ("I. INTRODUCTION",)

    def test_the_title_is_captured_as_metadata(self, pdf: Path) -> None:
        extractor = _extractor(
            [
                _Item("title", "A Chaos Based SVPWM Technique for B4 Inverters"),
                _Item("text", "Some body text."),
            ]
        )

        assert extractor.extract(pdf).title == "A Chaos Based SVPWM Technique for B4 Inverters"

    def test_the_abstract_survives_as_content(self, pdf: Path) -> None:
        extractor = _extractor(
            [
                _Item("title", "Some Paper About Predictive Control Methods"),
                _Item("text", "Abstract— we propose a scheme."),
                _Item("section_header", "I. INTRODUCTION"),
            ]
        )

        texts = [b.text for b in extractor.extract(pdf).blocks]

        assert any("we propose a scheme" in t for t in texts)

    def test_a_later_title_item_becomes_content(self, pdf: Path) -> None:
        # Only the first title item is the paper's; later ones title figures or tables.
        extractor = _extractor(
            [
                _Item("title", "A Study of Predictive Control for Induction Motors"),
                _Item("text", "Body."),
                _Item("title", "TABLE I: Motor parameters"),
            ]
        )

        result = extractor.extract(pdf)

        assert result.title == "A Study of Predictive Control for Induction Motors"
        assert any("TABLE I" in b.text for b in result.blocks)
        assert "TABLE I: Motor parameters" not in result.headings

    def test_a_missing_title_is_warned_about(self, pdf: Path) -> None:
        extractor = _extractor([_Item("text", "Body with no title item.")])

        assert any("no title" in w for w in extractor.extract(pdf).warnings)


class TestBlockClassification:
    def test_formulas_become_equation_blocks(self, pdf: Path) -> None:
        extractor = _extractor(
            [_Item("text", "The cost function:"), _Item("formula", r"$$g = |T_{ref} - T|$$")]
        )

        kinds = [b.kind for b in extractor.extract(pdf).blocks]

        assert ChunkKind.EQUATION in kinds

    def test_captions_are_kept_separate(self, pdf: Path) -> None:
        extractor = _extractor([_Item("caption", "Fig. 3. Torque response.")])

        assert extractor.extract(pdf).blocks[0].kind is ChunkKind.CAPTION

    def test_running_heads_and_footers_are_dropped(self, pdf: Path) -> None:
        extractor = _extractor(
            [
                _Item("page_header", "IEEE ACCESS, VOL. 9, 2021"),
                _Item("page_footer", "1234"),
                _Item("text", "Real content."),
            ]
        )

        blocks = extractor.extract(pdf).blocks

        assert len(blocks) == 1
        assert blocks[0].text == "Real content."

    def test_page_numbers_are_carried_through(self, pdf: Path) -> None:
        extractor = _extractor([_Item("text", "On page seven.", prov=[_Prov(page_no=7)])])

        assert extractor.extract(pdf).blocks[0].page == 7

    def test_missing_provenance_does_not_fail(self, pdf: Path) -> None:
        extractor = _extractor([_Item("text", "No provenance.", prov=[])])

        assert extractor.extract(pdf).blocks[0].page == 0

    def test_a_document_with_no_content_is_an_error(self, pdf: Path) -> None:
        extractor = _extractor([_Item("page_footer", "1234")])

        with pytest.raises(ExtractionError, match="no content"):
            extractor.extract(pdf)

    def test_a_missing_file_is_an_error(self) -> None:
        with pytest.raises(ExtractionError, match="no such file"):
            DoclingExtractor().extract(Path("/nonexistent/paper.pdf"))


class TestNormalise:
    def test_ligature_codepoints_are_decomposed(self) -> None:
        assert normalise("stator ﬂux linkage") == "stator flux linkage"
        assert normalise("deﬁning the scheme") == "defining the scheme"

    def test_soft_hyphens_and_zero_width_spaces_are_removed(self) -> None:
        assert normalise("mod­ulation") == "modulation"
        assert normalise("wide​band") == "wideband"

    def test_runs_of_whitespace_collapse(self) -> None:
        assert normalise("too    many     spaces") == "too many spaces"

    def test_paragraph_breaks_survive(self) -> None:
        assert "\n\n" in normalise("First paragraph.\n\n\n\nSecond paragraph.")


class TestTitleLabelledAsSectionHeader:
    """Docling labels a paper's title as either a title item or a section_header."""

    def test_a_long_leading_section_header_is_treated_as_the_title(self, pdf: Path) -> None:
        long_title = (
            "Artificial Bee Colony Optimized Self-tuning PI Speed Controller for "
            "FCS-MPCC of Permanent Magnet Synchronous Machines"
        )
        extractor = _extractor(
            [
                _Item("section_header", long_title),
                _Item("text", "Abstract— this paper proposes a tuning scheme."),
                _Item("section_header", "I. INTRODUCTION"),
                _Item("text", "Motor drives are widely used."),
            ]
        )

        result = extractor.extract(pdf)

        assert result.title == long_title
        assert result.headings == ("I. INTRODUCTION",)

    def test_a_long_heading_later_in_the_document_stays_a_heading(self, pdf: Path) -> None:
        long_heading = (
            "V. FINITE CONTROL SET MODEL PREDICTIVE TORQUE CONTROL OF INDUCTION "
            "MOTOR FED WITH SSTPI AND ITS EXPERIMENTAL VALIDATION ON A TEST BENCH"
        )
        extractor = _extractor(
            [
                _Item("title", "A Study of Predictive Control for Induction Motors"),
                _Item("text", "Abstract."),
                _Item("section_header", long_heading),
                _Item("text", "Body."),
            ]
        )

        result = extractor.extract(pdf)

        assert result.title == "A Study of Predictive Control for Induction Motors"
        assert long_heading in result.headings

    def test_a_short_leading_heading_is_not_mistaken_for_a_title(self, pdf: Path) -> None:
        extractor = _extractor([_Item("section_header", "I. INTRODUCTION"), _Item("text", "Body.")])

        result = extractor.extract(pdf)

        assert result.headings == ("I. INTRODUCTION",)
        assert result.title == ""


class TestSectionHeadingShape:
    """Heading shape separates a title from a section heading.

    Corpus titles run 79-108 characters and some headings run longer, so a length
    threshold misclassified four papers in five.
    """

    @pytest.mark.parametrize(
        "heading",
        [
            "I. INTRODUCTION",
            "III. METHODOLOGY",
            "3.1 Results",
            "B. ESTIMATION",
            "Abstract",
            "Keywords",
            "IMPACT STATEMENT",
            "CCS CONCEPTS",
            "REFERENCES",
            "Q. Adaptive Forgetting Factor with Generalized Regularization Online"
            " Sequential Learning",
        ],
    )
    def test_section_headings_are_recognised(self, heading: str) -> None:
        assert looks_like_section_heading(heading)

    @pytest.mark.parametrize(
        "title",
        [
            "A Chaos Based SVPWM technique for B4 Inverter fed Two-Phase Symmetric Induction Motor",
            "Exploring Crowdsourced Content Moderation Through Lens of Reddit during COVID-19",
            "Robust Control of Grid-Tied Inverter with LC Filter via Adaptive FCS-MPC",
            "Leveraging Meta-Learner-based Stacking Framework for Stator Winding Fault Detection",
        ],
    )
    def test_paper_titles_are_not_mistaken_for_headings(self, title: str) -> None:
        assert not looks_like_section_heading(title)


class TestTitleBeforeAnySectionHeading:
    def test_a_title_after_a_banner_block_is_still_captured(self, pdf: Path) -> None:
        # Several papers open with a journal banner or author block.
        extractor = _extractor(
            [
                _Item("text", "IEEE TRANSACTIONS ON INDUSTRIAL ELECTRONICS, VOL. 70"),
                _Item(
                    "section_header", "An Overview of Sequential Learning Algorithms for Networks"
                ),
                _Item("section_header", "Abstract"),
                _Item("text", "We survey online learning."),
            ]
        )

        result = extractor.extract(pdf)

        assert result.title == "An Overview of Sequential Learning Algorithms for Networks"
        assert "Abstract" in result.headings

    def test_the_title_repeated_as_a_running_head_is_dropped(self, pdf: Path) -> None:
        extractor = _extractor(
            [
                _Item(
                    "section_header", "An Overview of Sequential Learning Algorithms for Networks"
                ),
                _Item("section_header", "Abstract"),
                _Item("text", "We survey online learning."),
                _Item(
                    "section_header", "An Overview of Sequential Learning Algorithms for Networks"
                ),
                _Item("text", "More content."),
            ]
        )

        result = extractor.extract(pdf)

        assert result.headings == ("Abstract",)


class TestPublisherFurniture:
    """Elsevier, Frontiers and IEEE templates print a banner above the title."""

    @pytest.mark.parametrize(
        "banner",
        [
            "ScienceDirect",
            "ScienceDirect ScienceDirect",
            "ARTICLE",
            "ORIGINAL RESEARCH",
            "Research paper",
            "Contents lists available at ScienceDirect",
            "Journal homepage: www.elsevier.com/locate/x",
        ],
    )
    def test_banners_are_not_plausible_titles(self, banner: str) -> None:
        assert not is_plausible_title(banner)

    def test_the_real_title_after_a_banner_is_captured(self, pdf: Path) -> None:
        # A banner counted as a section heading would block the title that follows it.
        extractor = _extractor(
            [
                _Item("section_header", "ScienceDirect"),
                _Item(
                    "section_header",
                    "Genetic algorithm tuned adaptive discrete-time sliding mode controller",
                ),
                _Item("section_header", "Abstract"),
                _Item("text", "We propose a controller."),
            ]
        )

        result = extractor.extract(pdf)

        assert result.title.startswith("Genetic algorithm tuned")
        assert "ScienceDirect" not in result.headings

    def test_a_very_short_heading_is_never_taken_as_a_title(self, pdf: Path) -> None:
        extractor = _extractor(
            [_Item("section_header", "ARTICLE"), _Item("text", "Body text here.")]
        )

        assert extractor.extract(pdf).title == ""


class TestCarriageReturns:
    def test_carriage_returns_become_newlines(self) -> None:
        assert "\r" not in normalise("First line\r\nSecond line")

    def test_ordinary_text_is_untouched(self) -> None:
        text = "The stator flux estimate converges within two sampling periods."
        assert normalise(text) == text

    def test_legitimate_word_doubling_survives(self) -> None:
        assert normalise("had had a different meaning") == "had had a different meaning"
        assert normalise("that that is fine") == "that that is fine"


class TestRepeatedText:
    """Two of twenty papers ship a stacked text layer; one produced 796 chunks from 6 pages."""

    def test_a_line_repeated_in_the_same_block_is_dropped(self) -> None:
        text = (
            "Abstract: This paper proposes a linear quadratic Gaussian control.\n"
            "Abstract: This paper proposes a linear quadratic Gaussian control.\n"
            "The quadrotor model is derived from first principles."
        )

        assert collapse_repeated_lines(text).count("Abstract:") == 1

    def test_punctuation_only_differences_count_as_repeats(self) -> None:
        # The stacked layer has its punctuation stripped.
        text = (
            "Surveillance, reconnaissance, infrastructure inspection.\n"
            "Surveillancereconnaissanceinfrastructure inspection"
        )

        assert len(collapse_repeated_lines(text).split("\n")) == 1

    @pytest.mark.parametrize(
        "text",
        [
            "The controller had had a stable response.",
            "We note that that assumption is reasonable.",
            "It is what it is, and the result is what the result is.",
        ],
    )
    def test_english_doubling_survives(self, text: str) -> None:
        # A repeat must be a whole line of at least twenty characters.
        assert collapse_repeated_lines(text) == text

    def test_no_line_is_ever_edited(self) -> None:
        # Citations are verified against this text, so surviving lines must be verbatim.
        text = "A unique first line here.\nA unique first line here.\nA different second line."

        for line in collapse_repeated_lines(text).split("\n"):
            assert line in text

    def test_repeats_are_found_across_blocks(self) -> None:
        # Consecutive blocks are growing prefixes of the same text, so copies are rarely adjacent.
        seen: set[str] = set()
        first = collapse_repeated_lines("A sufficiently long line of real prose here.", seen)
        second = collapse_repeated_lines("A sufficiently long line of real prose here.", seen)

        assert first
        assert not second.strip()
