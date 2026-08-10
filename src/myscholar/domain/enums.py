"""Vocabulary shared across the pipeline."""

from __future__ import annotations

from enum import StrEnum


class DocumentKind(StrEnum):
    """What sort of source document a record came from."""

    PAPER = "paper"
    CV = "cv"


class SectionKind(StrEnum):
    """Canonical section, mapped from whatever heading the document actually used.

    Section identity drives retrieval filtering ("search only the methods"),
    chunk boundaries (a chunk never spans two sections) and citation display.
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
    """What kind of question this is, which decides how it gets answered.

    Routing on this is the point: HyDE actively harms a bibliographic lookup,
    and decomposition wastes seconds on a single-fact question.
    """

    BIBLIOGRAPHIC = "bibliographic"
    """Answered by SQL over curated metadata. No vector search, no invention."""

    SINGLE_PAPER_FACTUAL = "single_paper_factual"
    CROSS_PAPER_SYNTHESIS = "cross_paper_synthesis"
    METHODOLOGICAL = "methodological"
    BIOGRAPHICAL = "biographical"
    OUT_OF_SCOPE = "out_of_scope"
    """Short-circuited before any retrieval happens."""

    AMBIGUOUS = "ambiguous"
    """Answered with a clarifying question, which is a valid terminal state."""


class Verdict(StrEnum):
    RELEVANT = "relevant"
    PARTIAL = "partial"
    IRRELEVANT = "irrelevant"


class RefusalReason(StrEnum):
    """Refusal is a first-class outcome, not an error.

    The corpus is deliberately closed and there is no web fallback, so
    admitting ignorance is the correct behaviour rather than a failure.
    """

    NO_RELEVANT_CONTENT = "no_relevant_content"
    OUT_OF_CORPUS_SCOPE = "out_of_corpus_scope"
    UNGROUNDED_AFTER_RETRY = "ungrounded_after_retry"
    BUDGET_EXHAUSTED = "budget_exhausted"


class Stage(StrEnum):
    """Pipeline stages surfaced to the UI so the adaptive loop is visible."""

    CLASSIFYING = "classifying"
    RETRIEVING = "retrieving"
    GRADING = "grading"
    GENERATING = "generating"
    VERIFYING = "verifying"
    DONE = "done"
