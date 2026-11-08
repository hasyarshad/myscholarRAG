"""Grouping blocks into sections."""

from __future__ import annotations

from myscholar.adapters.fakes import FakeChatModel
from myscholar.domain.enums import ChunkKind, SectionKind
from myscholar.ingest.grouping import group_into_sections
from myscholar.ingest.sectionizer import Sectionizer
from myscholar.ports.extractor import ExtractedBlock


def heading(text: str, page: int = 1) -> ExtractedBlock:
    return ExtractedBlock(kind=ChunkKind.PROSE, text=text, page=page, heading=text)


def prose(text: str, page: int = 1) -> ExtractedBlock:
    return ExtractedBlock(kind=ChunkKind.PROSE, text=text, page=page)


def equation(text: str, page: int = 1) -> ExtractedBlock:
    return ExtractedBlock(kind=ChunkKind.EQUATION, text=text, page=page)


def _sectionizer() -> Sectionizer:
    return Sectionizer(FakeChatModel())


class TestGroupIntoSections:
    async def test_blocks_land_under_the_preceding_heading(self) -> None:
        blocks = [
            heading("I. INTRODUCTION"),
            prose("Motor drives are widely used."),
            heading("VII. EXPERIMENTAL RESULTS"),
            prose("Torque ripple fell by 18%."),
        ]

        sections = await group_into_sections(blocks, sectionizer=_sectionizer())

        assert [s.kind for s in sections] == [SectionKind.INTRODUCTION, SectionKind.RESULTS]
        assert "Torque ripple" in sections[1].blocks[0].text

    async def test_content_before_any_heading_is_kept(self) -> None:
        # In most IEEE papers this is the abstract.
        blocks = [
            prose("This paper proposes a weighting factor optimisation scheme."),
            heading("I. INTRODUCTION"),
            prose("Motor drives are widely used."),
        ]

        sections = await group_into_sections(blocks, sectionizer=_sectionizer())

        assert sections[0].kind is SectionKind.ABSTRACT
        assert "weighting factor optimisation" in sections[0].blocks[0].text

    async def test_a_heading_with_no_content_is_dropped(self) -> None:
        blocks = [
            heading("I. INTRODUCTION"),
            heading("II. RELATED WORK"),
            prose("Prior work is reviewed."),
        ]

        sections = await group_into_sections(blocks, sectionizer=_sectionizer())

        assert [s.kind for s in sections] == [SectionKind.RELATED_WORK]

    async def test_a_section_of_only_equations_is_dropped(self) -> None:
        blocks = [
            heading("III. DERIVATION"),
            equation(r"$$T = \frac{3}{2} p \psi i_q$$"),
            equation(r"$$\lambda_1 = 0$$"),
            heading("VII. RESULTS"),
            prose("The scheme was validated experimentally."),
        ]

        sections = await group_into_sections(blocks, sectionizer=_sectionizer())

        assert [s.kind for s in sections] == [SectionKind.RESULTS]

    async def test_equations_are_kept_when_prose_accompanies_them(self) -> None:
        blocks = [
            heading("III. METHODOLOGY"),
            prose("The cost function is defined as follows."),
            equation(r"$$g = |T_{ref} - T|$$"),
        ]

        sections = await group_into_sections(blocks, sectionizer=_sectionizer())

        assert len(sections) == 1
        assert any(b.kind is ChunkKind.EQUATION for b in sections[0].blocks)

    async def test_page_range_spans_the_section(self) -> None:
        blocks = [
            heading("VII. RESULTS", page=7),
            prose("First result.", page=7),
            prose("Second result.", page=9),
        ]

        sections = await group_into_sections(blocks, sectionizer=_sectionizer())

        assert sections[0].pages == (7, 9)

    async def test_missing_page_numbers_do_not_break_the_range(self) -> None:
        blocks = [heading("VII. RESULTS", page=0), prose("A result.", page=0)]

        sections = await group_into_sections(blocks, sectionizer=_sectionizer())

        assert sections[0].pages == (0, 0)

    async def test_empty_input_yields_nothing(self) -> None:
        assert await group_into_sections([], sectionizer=_sectionizer()) == []

    async def test_references_are_grouped_but_left_for_the_caller_to_exclude(self) -> None:
        blocks = [
            heading("VII. RESULTS"),
            prose("Findings."),
            heading("REFERENCES"),
            prose("[1] Some citation."),
        ]

        sections = await group_into_sections(blocks, sectionizer=_sectionizer())

        assert SectionKind.REFERENCES in {s.kind for s in sections}
