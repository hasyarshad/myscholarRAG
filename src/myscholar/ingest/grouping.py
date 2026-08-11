"""Grouping extracted blocks into sections.

Separated from the pipeline because it is the step most likely to be wrong in
ways that are invisible downstream: a mis-grouped block ends up filed under the
wrong heading, gets the wrong contextual header, and is then filtered out of
searches that should have found it. Worth testing on its own.
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

        A section containing only equations has no prose to retrieve and no
        prose to cite, so indexing it would add a chunk that can never usefully
        match.
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

    Content appearing before any heading is kept in a leading section rather
    than dropped — on most IEEE papers that content is the abstract, which is
    among the most useful text in the document.
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
