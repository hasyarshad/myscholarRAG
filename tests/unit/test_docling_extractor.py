"""Docling block classification, exercised against a stubbed document.

Docling itself is not invoked: these tests stub the object it returns, so they
verify how *we* interpret its labels. That interpretation is where the real bug
was — everything Docling produced was correct.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from myscholar.adapters.extraction.docling_extractor import DoclingExtractor, normalise
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
    """Regression guards for a defect that required re-indexing the corpus."""

    def test_the_title_does_not_become_a_section_heading(self, pdf: Path) -> None:
        """Docling labels the paper's own title as a title item.

        Treating that as a heading produced a section headed by the entire
        paper title, spanning the abstract and author block, which the
        sectionizer then classified by content and filed as methods.
        """
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
                _Item("title", "A Chaos Based SVPWM Technique"),
                _Item("text", "Some body text."),
            ]
        )

        assert extractor.extract(pdf).title == "A Chaos Based SVPWM Technique"

    def test_the_abstract_survives_as_content(self, pdf: Path) -> None:
        """It sits before the first heading, and is among the best text there is."""
        extractor = _extractor(
            [
                _Item("title", "Some Paper"),
                _Item("text", "Abstract— we propose a scheme."),
                _Item("section_header", "I. INTRODUCTION"),
            ]
        )

        texts = [b.text for b in extractor.extract(pdf).blocks]

        assert any("we propose a scheme" in t for t in texts)

    def test_a_later_title_item_becomes_content(self, pdf: Path) -> None:
        """Only the first is the paper's; later ones are figure or table titles."""
        extractor = _extractor(
            [
                _Item("title", "The Paper Title"),
                _Item("text", "Body."),
                _Item("title", "TABLE I: Motor parameters"),
            ]
        )

        result = extractor.extract(pdf)

        assert result.title == "The Paper Title"
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
        """Citations resolve to a page."""
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
        """Backends disagree about emitting one codepoint or two; the index
        cannot contain both forms and still match consistently."""
        assert normalise("stator ﬂux linkage") == "stator flux linkage"
        assert normalise("deﬁning the scheme") == "defining the scheme"

    def test_soft_hyphens_and_zero_width_spaces_are_removed(self) -> None:
        assert normalise("mod­ulation") == "modulation"
        assert normalise("wide​band") == "wideband"

    def test_runs_of_whitespace_collapse(self) -> None:
        assert normalise("too    many     spaces") == "too many spaces"

    def test_paragraph_breaks_survive(self) -> None:
        assert "\n\n" in normalise("First paragraph.\n\n\n\nSecond paragraph.")
