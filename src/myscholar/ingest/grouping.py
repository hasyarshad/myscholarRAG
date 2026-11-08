"""Grouping extracted blocks into sections.

Kept apart from the pipeline so it can be tested on its own: a mis-grouped block gets the
wrong heading and contextual header, which nothing downstream detects.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from myscholar.domain.enums import ChunkKind, DocumentKind, SectionKind
from myscholar.ingest.sectionizer import Sectionizer
from myscholar.ports.extractor import ExtractedBlock


@dataclass(frozen=True, slots=True)
class PendingSection:
    """A section and its content, before either acquires database identity."""

    kind: SectionKind
    heading: str
    blocks: list[ExtractedBlock] = field(default_factory=list)

    @property
    def pages(self) -> tuple[int, int]:
        pages = [b.page for b in self.blocks if b.page > 0]
        return (min(pages), max(pages)) if pages else (0, 0)

    @property
    def has_answerable_content(self) -> bool:
        """Whether anything here could answer a question.

        A section of equations alone has no prose to retrieve or cite.
        """
        return any(b.kind is not ChunkKind.EQUATION for b in self.blocks)


async def group_into_sections(
    blocks: Sequence[ExtractedBlock],
    *,
    sectionizer: Sectionizer,
    title: str = "",
    kind: DocumentKind = DocumentKind.PAPER,
    leading_kind: SectionKind = SectionKind.ABSTRACT,
) -> list[PendingSection]:
    """Assign each block to the section whose heading precedes it.

    Content before any heading is kept in a leading section; on most IEEE papers it is
    the abstract.
    """
    sections: list[PendingSection] = []
    current = PendingSection(kind=leading_kind, heading="Abstract")

    for block in blocks:
        if block.is_heading:
            if current.blocks:
                sections.append(current)
            assignment = await sectionizer.classify(block.heading or "", title=title, kind=kind)
            current = PendingSection(kind=assignment.kind, heading=block.heading or "")
            continue
        current.blocks.append(block)

    if current.blocks:
        sections.append(current)

    return [s for s in sections if s.has_answerable_content]
