"""Turning pipeline state into phrases a visitor can read.

Kept server-side rather than in the widget. The widget had its own mapping from
stage name to English, which meant it was describing what it *assumed* each
stage did — and it assumed retrieval always searched the papers. Once the CV
and the site were indexed that label became a plain untruth, on exactly the
questions where a visitor is most likely to be testing whether the thing works.

Only the server knows what was actually searched, so only the server can say.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from myscholar.domain.enums import DocumentKind, QueryClass

#: How each class is described in the interface. Naming the route makes the
#: adaptive part of the system visible: a visitor watching a question about
#: publication counts get answered "from the metadata" without a search is
#: seeing the design work.
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


#: Plural nouns for what a passage can have come from.
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

    Takes ``(kind, document_id)`` pairs and counts *distinct documents*, not
    passages. Parent expansion routinely returns three spans from one paper,
    and counting those would report "3 papers" from a single source — a number
    that is wrong in the direction that flatters.

    Papers are counted because which and how many is the interesting part. The
    CV and the site are not: there is one of each, and "1 the site" reads like
    a bug.
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

    # An unrecognised kind is described generically rather than dropped, so a
    # new document type shows up as vague rather than as missing.
    known = {d.value for d in DocumentKind}
    parts.extend(sorted(k for k in by_kind if k not in known))

    return _join(parts)
