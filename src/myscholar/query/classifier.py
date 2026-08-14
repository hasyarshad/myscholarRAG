"""Classifying a question so it can be routed.

One small-model call, schema-constrained, deciding what kind of question this is
and which paper (if any) it refers to. Everything downstream branches on the
result, so this is the single call whose reliability matters most: a wrong class
sends the question down a pipeline tuned for a different question type, and that
failure is systematic rather than random.

The corpus titles go into the prompt. Twenty titles is a few hundred tokens and
turns entity linking from a recall problem into a matching one.

See docs/learning/04-query-translation-and-routing.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from myscholar.domain.enums import QueryClass
from myscholar.ports.llm import ChatModel, Message, Role

#: Recognised without a model call. These are unambiguous, they are a large
#: share of what a public bot receives, and spending 300ms of GPU on "hi" is
#: waste.
_GREETING = re.compile(
    r"^\s*(hi|hey|hello|yo|sup|good\s+(morning|afternoon|evening)|"
    r"thanks?|thank\s+you|cheers|bye|goodbye|ok|okay)\b"
    r"(\s+(there|everyone|all|mate|again))?[\s!.?,]*$",
    re.IGNORECASE,
)

#: A bare reference with no antecedent: "the paper", "it", "that study". These
#: cannot be resolved without asking, and a model given one will confidently
#: pick a paper rather than admit the ambiguity — which is the wrong behaviour
#: even when the guess happens to be reasonable.
_BARE_REFERENCE = re.compile(
    r"^\s*(what|tell\s+me|describe|summar[iy][sz]e|explain|how|why|when|who)?\s*"
    r"(about|did|does|is|was|were)?\s*"
    r"(it|the\s+(paper|study|work|article|research)|that\s+(paper|study|work|one))"
    r"\b[^?.!]{0,24}[?.!]?\s*$",
    re.IGNORECASE,
)

#: Counting and listing questions about the collection. Matched before the model
#: sees them because this is the class where a wrong route causes confident
#: fabrication, and a pattern match cannot drift.
_BIBLIOGRAPHIC = re.compile(
    r"\b(how\s+many|how\s+much)\b.*\b(papers?|publications?|articles?|works?|stud(y|ies))\b"
    r"|\b(list|show|name)\b.*\b(all|every|your|his|the)\b.*"
    r"\b(paper|publication|article|work)s?\b"
    r"|\b(paper|publication|article|work)s?\b.*\bin\s+(19|20)\d{2}\b"
    # Recency and precedence: "his most recent paper", "his latest publication",
    # "what was his first paper".
    #
    # Only "most recent" used to be matched, so "what is his recent paper?"
    # went to retrieval and came back naming a 2021 chaos paper — a confident,
    # cited, wrong answer, which is a worse failure than the refusals the noun
    # gap produced. A bare "recent" is now included.
    #
    # The adjective and the noun must be adjacent. The earlier form allowed
    # ".*" between them, which quietly made "how does the latest controller
    # work" a bibliographic question. "work" additionally gets no intervening
    # words at all, since it is a verb as often as a noun; the unambiguous
    # nouns keep a wider gap so "his latest journal article" still matches.
    #
    # A topical tail disqualifies the match: "his recent work on fault
    # detection" is a content question, and answering it with the latest paper
    # overall would ignore what was asked.
    r"|\b(most\s+(recent|cited|latest)|recent|latest|newest|earliest|oldest|first)\s+"
    r"((\w+\s+){0,2}(paper|publication|article)s?|works?)\b"
    r"(?!\s+(on|about|regarding|concerning|involving|related\s+to)\b)"
    # Venue questions *about the collection*: "which journals has he published
    # in", "where has he published". Deliberately not "which paper appeared in
    # an IET journal", which asks to identify a paper and belongs in retrieval —
    # requiring a possessive or a bare "has ... published" keeps them apart.
    r"|\b(what|which)\b\s+(venue|journal|conference|proceedings)s?\b"
    r"|\b(his|her|their|your)\s+(venue|journal|conference|proceedings)s?\b"
    r"|\bwhere\b.*\b(has|have)\b.*\bpublish"
    r"|\bwhat\s+year\b.*\b(publish|appear)"
    # The verb on its own, with the noun left implicit: "how many did he
    # publish in 2025", "has he published anything since 2023". Requiring the
    # noun missed every one of these, and the model then classified them
    # out_of_scope — a counting question stripped of the word "papers" reads
    # like chitchat to it. Measured: 5 of 12 realistic phrasings fell through.
    # "why" is excluded: "why did he publish this as open access" asks about a
    # decision, and routing it to SQL would answer a count question nobody
    # asked.
    r"|(?<!why\s)\b(how\s+many|what|which|when|how\s+often|has|have|had|did|does|do)\b"
    r"[^?.!]{0,40}\bpublish",
    re.IGNORECASE,
)

#: A count with its subject left in the previous turn: "how many in 2025",
#: "and how many in 2024?".
#:
#: This is deliberately narrow — the whole question must be nothing but a
#: count and a year. Within a bot scoped to one person's publications there is
#: exactly one thing that phrasing can mean, so resolving it deterministically
#: beats both refusing and asking a model to guess. Anything longer ("how many
#: layers in the 2025 model") falls through to normal classification.
_ELLIPTICAL_COUNT = re.compile(
    r"^\s*(and\s+|ok\s+|so\s+)?(what\s+about\s+|how\s+(many|about))\s*"
    r"(in|during|for|about)?\s*(19[89]\d|20[0-4]\d)\s*[?.!]?\s*$",
    re.IGNORECASE,
)

#: Things that live on a CV and in no paper: honours, funding, teaching,
#: service, credentials, affiliation.
#:
#: Matched by pattern rather than left to the model, which classified "how many
#: awards has Haseeb won?" as out_of_scope — the prompt's biographical class
#: listed only education, employment and roles, so awards read as off-topic and
#: the question was refused without ever reaching the CV that answers it.
#:
#: Being generous here is close to free. The plan for a biographical question
#: *boosts* CV sections in a branch that runs alongside the unrestricted
#: search, so a content question misrouted into it is still searched normally.
#: A CV question misrouted to out_of_scope, by contrast, is refused outright.
#: The costs are not symmetric, so neither is the threshold.
_PERSONAL = re.compile(
    r"\b(award|prize|scholarship|fellowship|honou?rs?|honored|honoured|grant|funding|"
    r"medal|distinction|certificat|credential|qualification|"
    r"supervisor|advisor|adviser|mentor|"
    r"teach|taught|teaching|lectured|"
    r"cv|r[ée]sum[ée]|biograph|"
    r"degree|phd|ph\.d|doctorate|masters?|msc|m\.sc|bachelors?|bsc|b\.sc|"
    r"affiliat|employer|university|institution|nationality|"
    # Spoken languages, keyed on the verb rather than the noun: "language"
    # alone appears throughout the NLP and media-literacy papers.
    r"speaks?\b|spoken|fluent|bilingual|multilingual)",
    re.IGNORECASE,
)

#: Attempts to reach past the corpus and address the system itself.
#:
#: Pattern-matched because this is now the *only* way a question becomes
#: out_of_scope apart from a greeting — see ``_PROMPT``.
_INJECTION = re.compile(
    r"\b(ignore|disregard|forget)\b[^.?!]{0,30}\b(previous|prior|above|earlier|all)?\s*"
    r"(instruction|prompt|rule|direction|context)"
    r"|\byou\s+are\s+now\b|\bact\s+as\b[^.?!]{0,20}\b(dan|jailbreak|unrestricted)"
    r"|\b(system|developer)\s+prompt\b"
    r"|\brepeat\b[^.?!]{0,20}\byour\s+(instruction|prompt|system)",
    re.IGNORECASE,
)

#: Everyday chitchat, on topics this corpus could not plausibly cover.
#:
#: Narrow on purpose. Since the model can no longer choose out_of_scope, an
#: unanswerable question otherwise takes the content path and is refused by the
#: grader — which is correct, but a weather question then gets "which paper do
#: you mean?", because the model reaches for `ambiguous` instead. Naming the
#: handful of genuinely off-topic subjects gets those a straight answer without
#: handing the judgement back to a model that was bad at it.
_CHITCHAT = re.compile(
    r"\b(weather|forecast|temperature\s+(today|outside)|"
    r"what\s+time\s+is\s+it|what\s+day\s+is|today'?s\s+date|"
    r"tell\s+me\s+a\s+joke|football|cricket|soccer|movie|recipe|"
    r"stock\s+price|bitcoin|who\s+won\s+the\s+(match|game|election))\b",
    re.IGNORECASE,
)

_YEAR = re.compile(r"\b(19[89]\d|20[0-4]\d)\b")


class QueryEntities(BaseModel):
    """What the question refers to, if anything."""

    paper_titles: list[str] = Field(
        default_factory=list,
        description="Titles from the provided list that the question refers to. Empty if none.",
    )
    years: list[int] = Field(default_factory=list)
    methods: list[str] = Field(
        default_factory=list,
        description="Technical methods or acronyms named in the question, e.g. NSGA-II.",
    )


class QueryClassification(BaseModel):
    """The classifier's output. Schema-constrained, so it cannot be malformed.

    ``out_of_scope`` is deliberately absent from the enumeration. Decoding is
    constrained to this schema, so leaving it out does not merely discourage
    the class — it makes it unreachable.

    That is the point. Out-of-scope was the model's most attractive wrong
    answer, and each mistake was a hard refusal: "how many awards has Haseeb
    won?" and "does he know PyTorch?" were both declined outright, with the
    answers sitting in the indexed CV. The model was being asked to predict
    whether an answer exists, which is a retrieval question, and it was
    guessing.

    Nothing is lost. A genuinely unanswerable question now takes the content
    path, retrieves nothing relevant, and is refused by the grader with
    ``NO_RELEVANT_CONTENT`` and a list of nearest topics. That refusal is
    reached by evidence rather than by prediction, and it costs one retrieval
    round. Greetings and injection attempts never get that far — they are
    matched by pattern, which cannot drift.
    """

    query_class: Literal[
        QueryClass.BIBLIOGRAPHIC,
        QueryClass.SINGLE_PAPER_FACTUAL,
        QueryClass.CROSS_PAPER_SYNTHESIS,
        QueryClass.METHODOLOGICAL,
        QueryClass.BIOGRAPHICAL,
        QueryClass.AMBIGUOUS,
    ]
    entities: QueryEntities = Field(default_factory=QueryEntities)
    reasoning: str = Field(default="", max_length=200)


@dataclass(frozen=True, slots=True)
class Classification:
    """A classification plus how it was reached."""

    query_class: QueryClass
    entities: QueryEntities = field(default_factory=QueryEntities)
    reasoning: str = ""
    decided_by: str = "model"
    """'pattern', 'model' or 'fallback' — recorded so the split is measurable."""


_PROMPT = """Classify a question asked of a search assistant covering {person}'s
own research, publications and background. Nothing else is in scope.

The available papers are:
{titles}

Question: "{question}"

Classes:
- bibliographic: about the collection itself — counting papers, listing them,
  which year or venue something appeared in. The answer is not inside any paper.
- single_paper_factual: a specific fact from one identifiable paper.
- cross_paper_synthesis: requires combining several papers, or asks how
  something developed across them.
- methodological: about techniques, algorithms or how something was done,
  phrased about the method rather than about a particular paper. Choose this
  over single_paper_factual when no paper is identified, even if you can guess
  which one it came from. If the question does name or clearly point at one
  paper ("the weighting factors paper", "the Reddit study"), it is
  single_paper_factual instead.
- biographical: about {person} personally — anything a CV records. Education,
  employment, affiliation and roles, and equally: awards, prizes,
  scholarships, fellowships, grants and other funding; teaching and
  supervision; peer review and service; certificates, skills and languages;
  availability and contact. The list is not exhaustive. If a question is about
  the person rather than about the research, it belongs here.
- ambiguous: contains no distinguishing content at all — "tell me about the
  paper", "what did it find?". A question naming a method, a topic or a
  finding is never ambiguous, however it is phrased: "which work uses Lyapunov
  exponents?" is answerable and is not this class.

Every question gets one of these classes. There is no "out of scope" option
here, and you should not try to judge whether an answer exists — pick the class
that describes what is being asked and let the search find out.

Also list any titles from above that the question refers to. Use the exact title
text. Leave the list empty if the question names no particular paper."""


class QueryClassifier:
    def __init__(
        self,
        model: ChatModel | None = None,
        *,
        titles: list[str] | None = None,
        person_name: str = "the researcher",
    ) -> None:
        self._model = model
        self._titles = titles or []
        self._person = person_name
        self.stats: dict[str, int] = {"pattern": 0, "model": 0, "fallback": 0}

    def set_titles(self, titles: list[str]) -> None:
        """Refresh the corpus titles used for entity linking."""
        self._titles = titles

    def classify_by_pattern(self, question: str) -> Classification | None:
        """Recognise the two classes that must never be got wrong.

        Bibliographic questions cause confident fabrication when misrouted, and
        greetings are pure waste. Both are cheap to match exactly, and a pattern
        does not drift the way a model can.
        """
        if _GREETING.match(question):
            return Classification(
                query_class=QueryClass.OUT_OF_SCOPE,
                reasoning="greeting",
                decided_by="pattern",
            )
        if _INJECTION.search(question):
            return Classification(
                query_class=QueryClass.OUT_OF_SCOPE,
                reasoning="asks the assistant to disregard its instructions",
                decided_by="pattern",
            )
        if _CHITCHAT.search(question):
            return Classification(
                query_class=QueryClass.OUT_OF_SCOPE,
                reasoning="everyday topic the corpus does not cover",
                decided_by="pattern",
            )
        if _BARE_REFERENCE.match(question):
            return Classification(
                query_class=QueryClass.AMBIGUOUS,
                reasoning="refers to a paper with no way to tell which",
                decided_by="pattern",
            )
        if _BIBLIOGRAPHIC.search(question) or _ELLIPTICAL_COUNT.match(question):
            return Classification(
                query_class=QueryClass.BIBLIOGRAPHIC,
                entities=QueryEntities(years=[int(y) for y in _YEAR.findall(question)]),
                reasoning="counting or listing question about the collection",
                decided_by="pattern",
            )
        # Checked after the bibliographic patterns, so "how many papers did he
        # publish while at that university" stays a publication count rather
        # than becoming a CV lookup.
        if _PERSONAL.search(question):
            return Classification(
                query_class=QueryClass.BIOGRAPHICAL,
                entities=QueryEntities(years=[int(y) for y in _YEAR.findall(question)]),
                reasoning="asks about something recorded on the CV",
                decided_by="pattern",
            )
        return None

    async def classify(self, question: str) -> Classification:
        if matched := self.classify_by_pattern(question):
            self.stats[matched.decided_by] += 1
            return matched

        if self._model is not None:
            try:
                result = await self._model.structured(
                    [
                        Message(
                            role=Role.USER,
                            content=_PROMPT.format(
                                person=self._person,
                                titles="\n".join(f"- {t}" for t in self._titles) or "(none)",
                                question=question[:1000],
                            ),
                        )
                    ],
                    QueryClassification,
                )
            except Exception:  # noqa: BLE001 - a classifier failure must not fail the request
                pass
            else:
                self.stats["model"] += 1
                return Classification(
                    query_class=result.query_class,
                    entities=self._ground_entities(result.entities),
                    reasoning=result.reasoning,
                    decided_by="model",
                )

        # Falling back to the broadest content class rather than refusing: an
        # unclassifiable question is better served by a general search than by
        # a canned "I don't know" it might not deserve.
        self.stats["fallback"] += 1
        return Classification(
            query_class=QueryClass.CROSS_PAPER_SYNTHESIS,
            reasoning="classification unavailable; using the general content path",
            decided_by="fallback",
        )

    def _ground_entities(self, entities: QueryEntities) -> QueryEntities:
        """Keep only titles that actually exist in the corpus.

        A model asked to pick from a list will occasionally paraphrase one.
        Filtering against the real titles means a hallucinated reference becomes
        no filter rather than a filter matching nothing.
        """
        if not self._titles:
            return entities

        lowered = {t.casefold(): t for t in self._titles}
        grounded: list[str] = []
        for candidate in entities.paper_titles:
            key = candidate.casefold().strip()
            if key in lowered:
                grounded.append(lowered[key])
                continue
            # Accept a clear prefix or containment match, which is how a model
            # usually truncates a long title.
            for title_key, title in lowered.items():
                if len(key) > 20 and (key in title_key or title_key.startswith(key)):
                    grounded.append(title)
                    break

        return QueryEntities(
            paper_titles=list(dict.fromkeys(grounded)),
            years=entities.years,
            methods=entities.methods,
        )
