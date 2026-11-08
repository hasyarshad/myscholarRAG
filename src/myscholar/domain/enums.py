"""Vocabulary shared across the pipeline."""

from __future__ import annotations

from enum import StrEnum


class DocumentKind(StrEnum):
    """What sort of source document a record came from."""

    PAPER = "paper"
    CV = "cv"
    PAGE = "page"
    """A page from the portfolio site: projects, roles, research summaries."""


class SectionKind(StrEnum):
    """Canonical section, mapped from whatever heading the document used.

    Drives retrieval boosts and exclusions, chunk boundaries (a chunk never spans two
    sections) and citation display.
    """

    TITLE = "title"
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    RELATED_WORK = "related_work"
    METHODS = "methods"
    EXPERIMENTS = "experiments"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    LIMITATIONS = "limitations"
    ACKNOWLEDGEMENTS = "acknowledgements"
    REFERENCES = "references"
    APPENDIX = "appendix"
    OTHER = "other"

    # CV-specific
    CV_SUMMARY = "cv_summary"
    CV_EDUCATION = "cv_education"
    CV_EMPLOYMENT = "cv_employment"
    CV_PUBLICATIONS = "cv_publications"
    CV_SKILLS = "cv_skills"
    CV_AWARDS = "cv_awards"


#: Sections excluded from the retrieval index. References are 20-30% of a
#: typical paper and are pure noise for question answering.
NON_INDEXED_SECTIONS: frozenset[SectionKind] = frozenset(
    {SectionKind.REFERENCES, SectionKind.ACKNOWLEDGEMENTS}
)


class ChunkKind(StrEnum):
    """Chunks are not all prose; tables and captions are never split."""

    PROSE = "prose"
    TABLE = "table"
    CAPTION = "caption"
    EQUATION = "equation"


class QueryClass(StrEnum):
    """What kind of question this is, which decides how it is answered."""

    BIBLIOGRAPHIC = "bibliographic"
    """Answered by SQL over verified metadata, without vector search."""

    SINGLE_PAPER_FACTUAL = "single_paper_factual"
    CROSS_PAPER_SYNTHESIS = "cross_paper_synthesis"
    METHODOLOGICAL = "methodological"
    BIOGRAPHICAL = "biographical"
    OUT_OF_SCOPE = "out_of_scope"
    """Answered with a fixed reply before any retrieval."""

    AMBIGUOUS = "ambiguous"
    """Answered with a clarifying question."""


class Verdict(StrEnum):
    RELEVANT = "relevant"
    PARTIAL = "partial"
    IRRELEVANT = "irrelevant"


class RefusalReason(StrEnum):
    """Why the loop declined to answer. The corpus is closed, with no web fallback."""

    NO_RELEVANT_CONTENT = "no_relevant_content"
    OUT_OF_CORPUS_SCOPE = "out_of_corpus_scope"
    UNGROUNDED_AFTER_RETRY = "ungrounded_after_retry"
    BUDGET_EXHAUSTED = "budget_exhausted"


class Stage(StrEnum):
    """Pipeline stages streamed to the interface."""

    CLASSIFYING = "classifying"
    RETRIEVING = "retrieving"
    GRADING = "grading"
    GENERATING = "generating"
    VERIFYING = "verifying"
    DONE = "done"
