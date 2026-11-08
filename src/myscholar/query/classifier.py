"""Classifying a question so it can be routed.

Patterns catch the unambiguous cases; the rest go to one schema-constrained call to the
utility model. The corpus titles are in the prompt, so entity linking is a matching task.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import Field

from myscholar.domain.enums import QueryClass
from myscholar.ports.llm import ChatModel, Message, ModelOutput, Role

#: Greetings, recognised without a model call.
_GREETING = re.compile(
    r"^\s*(hi|hey|hello|yo|sup|(good\s+)?(morning|afternoon|evening)|"
    r"thanks?|thank\s+you|cheers|bye|goodbye|ok|okay)\b"
    r"(\s+(there|everyone|all|mate|again))?[\s!.?,]*$",
    re.IGNORECASE,
)

#: A bare reference with no antecedent: "the paper", "it", "that study". Given one, the
#: model picks a paper instead of reporting the ambiguity.
_BARE_REFERENCE = re.compile(
    r"^\s*(what|tell\s+me|describe|summar[iy][sz]e|explain|how|why|when|who)?\s*"
    r"(about|did|does|is|was|were)?\s*"
    r"(it|the\s+(paper|study|work|article|research)|that\s+(paper|study|work|one))"
    r"\b[^?.!]{0,24}[?.!]?\s*$",
    re.IGNORECASE,
)

#: Counting and listing questions about the collection. Matched by pattern because a
#: misrouted one gets a fabricated count from retrieval.
_BIBLIOGRAPHIC = re.compile(
    r"\b(how\s+many|how\s+much)\b.*\b(papers?|publications?|articles?|works?|stud(y|ies))\b"
    r"|\b(list|show|name)\b.*\b(all|every|your|his|the)\b.*"
    r"\b(paper|publication|article|work)s?\b"
    r"|\b(paper|publication|article|work)s?\b.*\bin\s+(19|20)\d{2}\b"
    # Recency and precedence: "his recent paper", "what was his first paper". The
    # adjective must be next to the noun, so "how does the latest controller work"
    # does not match; "work" allows no words between, since it is often a verb. A
    # topical tail ("his recent work on fault detection") makes it a content question.
    r"|\b(most\s+(recent|cited|latest)|recent|latest|newest|earliest|oldest|first)\s+"
    r"((\w+\s+){0,2}(paper|publication|article)s?|works?)\b"
    r"(?!\s+(on|about|regarding|concerning|involving|related\s+to)\b)"
    # Venue questions about the collection: "which journals has he published in".
    # "Which paper appeared in an IET journal" identifies a paper and is left to
    # retrieval.
    r"|\b(what|which)\b\s+(venue|journal|conference|proceedings)s?\b"
    r"|\b(his|her|their|your)\s+(venue|journal|conference|proceedings)s?\b"
    r"|\bwhere\b.*\b(has|have)\b.*\bpublish"
    r"|\bwhat\s+year\b.*\b(publish|appear)"
    # The verb without the noun: "how many did he publish in 2025". Requiring the
    # noun missed 5 of 12 realistic phrasings. "why" is excluded: "why did he
    # publish this as open access" is not a count.
    r"|(?<!why\s)\b(how\s+many|what|which|when|how\s+often|has|have|had|did|does|do)\b"
    r"[^?.!]{0,40}\bpublish",
    re.IGNORECASE,
)

#: A count with its subject left in the previous turn: "how many in 2025". Only a bare
#: count and year match; for one person's publications that can only mean papers.
_ELLIPTICAL_COUNT = re.compile(
    r"^\s*(and\s+|ok\s+|so\s+)?(what\s+about\s+|how\s+(many|about))\s*"
    r"(in|during|for|about)?\s*(19[89]\d|20[0-4]\d)\s*[?.!]?\s*$",
    re.IGNORECASE,
)

#: Topics found on a CV and in no paper: honours, funding, teaching, credentials,
#: affiliation. Matched broadly because a biographical plan only boosts CV sections
#: alongside an unrestricted search, so a misrouted content question is still searched.
_PERSONAL = re.compile(
    r"\b(award|prize|scholarship|fellowship|honou?rs?|honored|honoured|grant|funding|"
    r"medal|distinction|certificat|credential|qualification|"
    r"supervisor|advisor|adviser|mentor|"
    r"teach|taught|teaching|lectured|"
    r"cv|r[ée]sum[ée]|biograph|"
    r"degree|phd|ph\.d|doctorate|masters?|msc|m\.sc|bachelors?|bsc|b\.sc|"
    r"affiliat|employer|university|institution|nationality|"
    # Spoken languages, keyed on the verb: "language" alone is common in the NLP papers.
    r"speaks?\b|spoken|fluent|bilingual|multilingual)",
    re.IGNORECASE,
)

#: Attempts to address the system itself instead of asking about the corpus.
_INJECTION = re.compile(
    r"\b(ignore|disregard|forget)\b[^.?!]{0,30}\b(previous|prior|above|earlier|all)?\s*"
    r"(instruction|prompt|rule|direction|context)"
    r"|\byou\s+are\s+now\b|\bact\s+as\b[^.?!]{0,20}\b(dan|jailbreak|unrestricted)"
    r"|\b(system|developer)\s+prompt\b"
    r"|\brepeat\b[^.?!]{0,20}\byour\s+(instruction|prompt|system)"
    # "Forget the papers and act as a travel agent" names no instruction or persona.
    r"|\b(forget|ignore|disregard)\b[^.?!]{0,20}\b(the\s+)?(papers|corpus|research|documents)\b"
    r"|\bact\s+as\s+(a|an|my)\b"
    r"|\bpretend\s+(you|to\s+be)\b",
    re.IGNORECASE,
)

#: Requests to do something (write a poem, debug code) instead of asking about the research.
_TASK_REQUEST = re.compile(
    r"^\s*(can|could|will|would|please)?\s*(you\s+)?"
    r"(write|compose|draft|generate|create|translate|summari[sz]e\s+this|"
    r"help\s+me\s+(write|with|build|code|debug)|code|program|debug|fix)\b"
    r"[^?.!]{0,60}\b(poem|story|essay|song|joke|code|script|program|email|letter|"
    r"python|javascript|sql|homework|assignment)\b",
    re.IGNORECASE,
)

#: Everyday topics the corpus cannot cover, given the out-of-scope reply without a search.
#: Kept narrow: anything else takes the content path and is refused on the evidence.
_CHITCHAT = re.compile(
    r"\b(weather|forecast|temperature\s+(today|outside)|"
    r"what\s+time\s+is\s+it|what\s+day\s+is|today'?s\s+date|"
    r"tell\s+me\s+a\s+joke|football|cricket|soccer|movie|recipe|"
    r"stock\s+price|bitcoin"
    r"|who\s+won\s+the\s+(match|game|election|world\s+cup|cup|series|final|title))\b",
    re.IGNORECASE,
)

_YEAR = re.compile(r"\b(19[89]\d|20[0-4]\d)\b")

#: Cap on the classifier's own explanation of its choice.
_MAX_REASONING = 200


class QueryEntities(ModelOutput):
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


class QueryClassification(ModelOutput):
    """The model's output schema.

    ``out_of_scope`` and ``ambiguous`` are left out of the enum, so constrained
    decoding cannot produce them. Both end the turn without retrieving, and the
    model chose them for answerable questions. Patterns catch the real cases, and an
    unanswerable question is refused after retrieval.
    """

    query_class: Literal[
        QueryClass.BIBLIOGRAPHIC,
        QueryClass.SINGLE_PAPER_FACTUAL,
        QueryClass.CROSS_PAPER_SYNTHESIS,
        QueryClass.METHODOLOGICAL,
        QueryClass.BIOGRAPHICAL,
    ]
    entities: QueryEntities = Field(default_factory=QueryEntities)
    reasoning: str = Field(default="", max_length=_MAX_REASONING)


@dataclass(frozen=True, slots=True)
class Classification:
    """A classification plus how it was reached."""

    query_class: QueryClass
    entities: QueryEntities = field(default_factory=QueryEntities)
    reasoning: str = ""
    decided_by: str = "model"
    """'pattern', 'model' or 'fallback'."""


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
  something developed across them. Also choose this for questions about the
  person rather than about a technique — their strengths, contributions,
  experience or track record in some area. Those span everything they have
  written, however the question is phrased.
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
Every question gets one of these classes. There is no "out of scope" or
"ambiguous" option here, and you should not try to judge whether an answer
exists — pick the class that describes what is being asked and let the search
find out.

- bibliographic is about the collection *as a collection*. "Which of his papers
  covers wind turbine pitch control?" names a topic and needs the papers read,
  so it is single_paper_factual however it is phrased.

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

    def set_titles(self, titles: list[str]) -> None:
        """Refresh the corpus titles used for entity linking."""
        self._titles = titles

    def classify_by_pattern(self, question: str) -> Classification | None:
        """Classify by pattern alone, or return None to leave it to the model."""
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
        if _CHITCHAT.search(question) or _TASK_REQUEST.match(question):
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
                return Classification(
                    query_class=result.query_class,
                    entities=self._ground_entities(result.entities),
                    reasoning=result.reasoning,
                    decided_by="model",
                )

        # The broadest content class: a general search can still answer or refuse.
        return Classification(
            query_class=QueryClass.CROSS_PAPER_SYNTHESIS,
            reasoning="classification unavailable; using the general content path",
            decided_by="fallback",
        )

    def _ground_entities(self, entities: QueryEntities) -> QueryEntities:
        """Keep only titles that exist in the corpus.

        The model sometimes paraphrases a title; one that matches nothing is dropped.
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
