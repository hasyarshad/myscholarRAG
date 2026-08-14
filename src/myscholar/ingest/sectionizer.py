"""Mapping document headings to canonical section kinds.

Sections do three jobs downstream: they bound chunks (a chunk never spans two),
they enable filtered retrieval ("search only the methods"), and they appear in
citations. So every heading has to end up as one of a fixed set of kinds.

Two paths, in order:

1. A pattern table, which handles conventional headings and IEEE numbering.
2. A small-model fallback for the rest.

The fallback is not decoration. Measured on this corpus, a keyword matcher
classified only 5 of 23 headings in one paper, because headings here are
domain-specific rather than conventional — ``III. DISCRETE MODEL OF THREE PHASE
INDUCTION MOTOR`` is a methods section and says none of the usual words. See
docs/learning/01b-extraction-findings.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import BaseModel, Field

from myscholar.domain.enums import DocumentKind, SectionKind
from myscholar.ports.llm import ChatModel, Message, Role

#: Leading enumeration to strip before matching: "III.", "3.1", "B)", "IV -".
_ENUMERATION = re.compile(r"^\s*(?:[IVXLC]+|\d+(?:\.\d+)*|[A-Z])\s*[.):\-]?\s+", re.IGNORECASE)

#: Ordered because the first match wins; more specific patterns come first.
#: "related work" must precede "work", "conclusion" must precede "discussion"
#: for headings like "Discussion and Conclusion".
_PATTERNS: tuple[tuple[re.Pattern[str], SectionKind], ...] = (
    (re.compile(r"^abstract"), SectionKind.ABSTRACT),
    (re.compile(r"^(index terms|keywords)"), SectionKind.ABSTRACT),
    (re.compile(r"^(introduction|motivation)"), SectionKind.INTRODUCTION),
    (
        re.compile(r"(related work|literature review|prior work|state of the art|background)"),
        SectionKind.RELATED_WORK,
    ),
    (
        re.compile(
            r"(methodolog|proposed (method|approach|scheme|technique|control)|"
            r"materials and methods|system model|problem formulation|"
            r"control (strategy|scheme|design)|algorithm)"
        ),
        SectionKind.METHODS,
    ),
    # RESULTS precedes EXPERIMENTS deliberately. "VII. EXPERIMENTAL RESULTS" is
    # ubiquitous in this corpus and is a results section; matching EXPERIMENTS
    # first would file every one of them wrongly. Nothing in the RESULTS
    # pattern matches "experimental setup" or "simulation setup", so the
    # reverse case still lands correctly.
    (
        re.compile(
            r"(results|findings|performance (analysis|evaluation)|"
            r"validation|comparison)"
        ),
        SectionKind.RESULTS,
    ),
    (
        re.compile(
            r"(experimental setup|experiment|simulation setup|"
            r"dataset|data collection|test bench)"
        ),
        SectionKind.EXPERIMENTS,
    ),
    (re.compile(r"(limitation|threats to validity|future work)"), SectionKind.LIMITATIONS),
    (re.compile(r"conclusion"), SectionKind.CONCLUSION),
    (re.compile(r"discussion"), SectionKind.DISCUSSION),
    (re.compile(r"(references|bibliography)"), SectionKind.REFERENCES),
    (re.compile(r"acknowledg"), SectionKind.ACKNOWLEDGEMENTS),
    (re.compile(r"appendix|supplementary|nomenclature"), SectionKind.APPENDIX),
)

_CV_PATTERNS: tuple[tuple[re.Pattern[str], SectionKind], ...] = (
    (re.compile(r"(summary|profile|objective|about)"), SectionKind.CV_SUMMARY),
    (re.compile(r"(education|qualification|academic)"), SectionKind.CV_EDUCATION),
    (re.compile(r"(experience|employment|appointment|position|career)"), SectionKind.CV_EMPLOYMENT),
    (re.compile(r"(publication|journal|conference|paper|patent)"), SectionKind.CV_PUBLICATIONS),
    (re.compile(r"(skill|competenc|technical|software|language)"), SectionKind.CV_SKILLS),
    (re.compile(r"(award|honou?r|grant|scholarship|prize|achievement)"), SectionKind.CV_AWARDS),
)


class HeadingClassification(BaseModel):
    """Schema handed to the model, so decoding is constrained to valid kinds."""

    kind: SectionKind
    confidence: float = Field(ge=0.0, le=1.0)


@dataclass(frozen=True, slots=True)
class SectionAssignment:
    kind: SectionKind
    heading: str
    matched_by: str
    """'pattern', 'model', or 'fallback' — recorded so the split is measurable."""


def strip_enumeration(heading: str) -> str:
    """Remove leading numbering: 'III. METHODOLOGY' -> 'METHODOLOGY'."""
    return _ENUMERATION.sub("", heading).strip()


def classify_by_pattern(heading: str, kind: DocumentKind) -> SectionKind | None:
    """Match a heading against the pattern table. ``None`` when nothing fits."""
    text = strip_enumeration(heading).lower().strip(" :.-")
    if not text:
        return None

    # Site pages use the same heading vocabulary as a CV — Education,
    # Experience, Skills — so they share the pattern table.
    patterns = _CV_PATTERNS if kind in (DocumentKind.CV, DocumentKind.PAGE) else _PATTERNS
    for pattern, section_kind in patterns:
        if pattern.search(text):
            return section_kind
    return None


_PROMPT = """You are labelling section headings from an academic paper.

Paper title: {title}

Assign this heading to the single most appropriate section kind.

Heading: "{heading}"

Guidance for this corpus, which is largely power electronics and control:
- Headings describing a system model, a converter or motor topology, a control
  law, or an estimation or prediction scheme are METHODS.
- Headings about test benches, simulation setup or datasets are EXPERIMENTS.
- Headings presenting measurements, waveforms or comparisons are RESULTS.
- Use OTHER only when nothing else plausibly fits."""


class Sectionizer:
    """Assigns a canonical kind to each heading."""

    def __init__(self, model: ChatModel | None = None) -> None:
        self._model = model
        self.stats: dict[str, int] = {"pattern": 0, "model": 0, "fallback": 0}

    async def classify(
        self,
        heading: str,
        *,
        title: str = "",
        kind: DocumentKind = DocumentKind.PAPER,
    ) -> SectionAssignment:
        if matched := classify_by_pattern(heading, kind):
            self.stats["pattern"] += 1
            return SectionAssignment(kind=matched, heading=heading, matched_by="pattern")

        if self._model is not None:
            try:
                result = await self._model.structured(
                    [
                        Message(
                            role=Role.USER,
                            content=_PROMPT.format(title=title or "unknown", heading=heading),
                        )
                    ],
                    HeadingClassification,
                )
            except Exception:  # noqa: BLE001 - a model failure must not abort ingestion
                pass
            else:
                self.stats["model"] += 1
                return SectionAssignment(kind=result.kind, heading=heading, matched_by="model")

        # Unclassifiable headings become OTHER rather than being dropped: the
        # text beneath them is still content, and losing it silently would be
        # worse than filing it imprecisely.
        self.stats["fallback"] += 1
        return SectionAssignment(kind=SectionKind.OTHER, heading=heading, matched_by="fallback")
