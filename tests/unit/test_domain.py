"""Domain types carry the behaviour the pipeline relies on."""

from __future__ import annotations

from uuid import uuid4

from myscholar.domain.answers import Budget, Refusal
from myscholar.domain.documents import Chunk, ContextBlock, PaperMetadata
from myscholar.domain.enums import NON_INDEXED_SECTIONS, ChunkKind, RefusalReason, SectionKind


class TestPaperMetadata:
    def test_single_author_label(self) -> None:
        meta = PaperMetadata(title="A paper", authors=("H. Arshad",), year=2023)

        assert meta.citation_label == "H. Arshad, 2023"

    def test_multiple_authors_are_abbreviated(self) -> None:
        meta = PaperMetadata(title="A paper", authors=("H. Arshad", "J. Doe"), year=2024)

        assert meta.citation_label == "H. Arshad et al., 2024"

    def test_falls_back_to_title_without_authors(self) -> None:
        meta = PaperMetadata(title="An untitled work with no authors listed")

        assert "An untitled work" in meta.citation_label

    def test_metadata_is_unverified_until_a_human_confirms_it(self) -> None:
        assert PaperMetadata(title="x").verified is False


class TestChunk:
    def _chunk(self, header: str) -> Chunk:
        return Chunk(
            id=uuid4(),
            document_id=uuid4(),
            section_id=uuid4(),
            ordinal_in_section=0,
            kind=ChunkKind.PROSE,
            text="The proposed method reduces THD by 3.1%.",
            context_header=header,
            token_count=12,
            page=4,
        )

    def test_embed_text_prepends_the_context_header(self) -> None:
        chunk = self._chunk("Paper: Chaos-based SVPWM (2021). Section: Results.")

        assert chunk.embed_text.startswith("Paper: Chaos-based SVPWM")
        assert chunk.text in chunk.embed_text

    def test_embed_text_is_bare_text_when_there_is_no_header(self) -> None:
        assert self._chunk("").embed_text == "The proposed method reduces THD by 3.1%."


class TestContextBlock:
    def test_render_includes_label_source_and_pages(self) -> None:
        block = ContextBlock(
            label="C1",
            document_id=uuid4(),
            section_id=uuid4(),
            document_title="Advanced Direct Torque Control",
            section_heading="4.2 Results",
            text="Torque ripple fell by 18%.",
            page_start=5,
            page_end=6,
        )

        rendered = block.render()

        assert rendered.startswith("[C1]")
        assert "Advanced Direct Torque Control" in rendered
        assert "p. 5-6" in rendered


class TestSectionKinds:
    def test_references_are_excluded_from_the_index(self) -> None:
        assert SectionKind.REFERENCES in NON_INDEXED_SECTIONS
        assert SectionKind.METHODS not in NON_INDEXED_SECTIONS


class TestBudget:
    def test_fresh_budget_is_not_exhausted(self) -> None:
        assert Budget().exhausted(elapsed_seconds=0.0) is False

    def test_wall_clock_deadline_exhausts_the_budget(self) -> None:
        assert Budget(deadline_seconds=10.0).exhausted(elapsed_seconds=11.0) is True

    def test_retrieval_rounds_exhaust_the_budget(self) -> None:
        budget = Budget(max_retrieval_rounds=3)
        budget.retrieval_rounds_used = 3

        assert budget.exhausted(elapsed_seconds=0.0) is True

    def test_step_limit_exhausts_the_budget(self) -> None:
        budget = Budget(max_steps=24)
        budget.steps_used = 24

        assert budget.exhausted(elapsed_seconds=0.0) is True


class TestRefusal:
    def test_refusal_reports_itself_as_refused(self) -> None:
        refusal = Refusal(
            reason=RefusalReason.NO_RELEVANT_CONTENT,
            text="I don't have anything on that.",
            nearest_topics=("grid-tied inverter control", "induction motor drives"),
        )

        assert refusal.refused is True
        assert len(refusal.nearest_topics) == 2
