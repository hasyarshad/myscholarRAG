"""Phrases describing pipeline state for the interface.

Built server-side because only the server knows which sources were searched.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from myscholar.domain.enums import DocumentKind, QueryClass

#: How each class is described in the interface.
_CLASS_PHRASE = {
    QueryClass.BIBLIOGRAPHIC: "a question about the publication list",
    QueryClass.SINGLE_PAPER_FACTUAL: "a fact from one paper",
    QueryClass.CROSS_PAPER_SYNTHESIS: "spans several papers",
    QueryClass.METHODOLOGICAL: "about methods",
    QueryClass.BIOGRAPHICAL: "about {short_name} rather than the research",
    QueryClass.AMBIGUOUS: "not specific enough to answer yet",
    QueryClass.OUT_OF_SCOPE: "outside what this covers",
}


def describe_class(query_class: QueryClass, *, short_name: str = "him") -> str:
    return _CLASS_PHRASE.get(query_class, query_class.value).format(short_name=short_name)


#: Singular and plural nouns for each source kind.
_SOURCE_NOUN = {
    DocumentKind.PAPER: ("paper", "papers"),
    DocumentKind.CV: ("the CV", "the CV"),
    DocumentKind.PAGE: ("the site", "the site"),
}


def _join(parts: Sequence[str]) -> str:
    if len(parts) <= 1:
        return parts[0] if parts else ""
    return f"{', '.join(parts[:-1])} and {parts[-1]}"


def describe_sources(sources: Iterable[tuple[str, object]]) -> str:
    """Name where a set of passages came from.

    Takes ``(kind, document_id)`` pairs and counts distinct documents, since one paper
    often yields several spans. Papers are counted; the CV and the site are named.
    """
    by_kind: dict[str, set[object]] = {}
    for kind, document_id in sources:
        by_kind.setdefault(kind, set()).add(document_id)
    if not by_kind:
        return ""

    parts: list[str] = []
    for kind in (DocumentKind.PAPER, DocumentKind.CV, DocumentKind.PAGE):
        if not (documents := by_kind.get(kind.value)):
            continue
        singular, plural = _SOURCE_NOUN[kind]
        if kind is DocumentKind.PAPER:
            parts.append(f"{len(documents)} {plural if len(documents) > 1 else singular}")
        else:
            parts.append(singular)

    # An unrecognised kind is listed by name instead of dropped.
    known = {d.value for d in DocumentKind}
    parts.extend(sorted(k for k in by_kind if k not in known))

    return _join(parts)
