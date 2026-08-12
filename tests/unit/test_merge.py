"""Parent-span expansion.

The property tests here are the point. Merging is where a subtle error is
invisible downstream: a wrong span reads perfectly well and simply is not what
was cited.
"""

from __future__ import annotations

from itertools import pairwise
from uuid import UUID, uuid4

from hypothesis import given, settings
from hypothesis import strategies as st

from myscholar.retrieval.merge import SeedHit, merge_windows

SECTION_A = UUID("11111111-1111-1111-1111-111111111111")
SECTION_B = UUID("22222222-2222-2222-2222-222222222222")
DOC_A = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
DOC_B = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")


def hit(
    ordinal: int,
    *,
    section: UUID = SECTION_A,
    document: UUID = DOC_A,
    score: float = 1.0,
) -> SeedHit:
    return SeedHit(
        chunk_id=uuid4(),
        document_id=document,
        section_id=section,
        ordinal=ordinal,
        score=score,
    )


class TestMergeWindows:
    def test_a_single_hit_expands_by_the_window(self) -> None:
        [span] = merge_windows([hit(5)], window=2)

        assert (span.start, span.end) == (3, 7)

    def test_the_window_is_clamped_at_zero(self) -> None:
        [span] = merge_windows([hit(1)], window=3)

        assert span.start == 0

    def test_the_window_is_clamped_to_the_section_length(self) -> None:
        [span] = merge_windows([hit(8)], window=3, section_lengths={SECTION_A: 10})

        assert span.end == 9

    def test_overlapping_windows_merge(self) -> None:
        spans = merge_windows([hit(3), hit(5)], window=2)

        assert len(spans) == 1
        assert (spans[0].start, spans[0].end) == (1, 7)

    def test_touching_windows_merge(self) -> None:
        """Chunks 0-2 and 3-5 are contiguous text; two blocks would misrepresent
        one passage as two independent pieces of evidence."""
        spans = merge_windows([hit(1), hit(4)], window=1)

        assert len(spans) == 1

    def test_distant_hits_stay_separate(self) -> None:
        spans = merge_windows([hit(0), hit(20)], window=2)

        assert len(spans) == 2

    def test_hits_in_different_sections_never_merge(self) -> None:
        """The defect in the previous implementation: it expanded by global row
        id with no filter, so a span could contain two documents and cite them
        as one source."""
        spans = merge_windows([hit(5, section=SECTION_A), hit(5, section=SECTION_B)], window=3)

        assert len(spans) == 2
        assert {s.section_id for s in spans} == {SECTION_A, SECTION_B}

    def test_hits_in_different_documents_never_merge(self) -> None:
        spans = merge_windows(
            [
                hit(5, section=SECTION_A, document=DOC_A),
                hit(6, section=SECTION_B, document=DOC_B),
            ],
            window=5,
        )

        assert len(spans) == 2

    def test_spans_are_ordered_by_score(self) -> None:
        """The generator should see the strongest evidence first."""
        spans = merge_windows([hit(0, score=0.2), hit(20, score=0.9)], window=1)

        assert spans[0].max_score == 0.9

    def test_seed_provenance_is_kept(self) -> None:
        a, b = hit(3), hit(5)

        [span] = merge_windows([a, b], window=2)

        assert span.seed_ids == frozenset({a.chunk_id, b.chunk_id})

    def test_empty_input_yields_nothing(self) -> None:
        assert merge_windows([]) == []

    def test_duplicate_hits_do_not_produce_duplicate_spans(self) -> None:
        spans = merge_windows([hit(4), hit(4), hit(4)], window=1)

        assert len(spans) == 1


class TestPromotion:
    def test_a_span_covering_most_of_a_section_takes_all_of_it(self) -> None:
        """Half a Methods section reads worse than all of it, at similar cost."""
        [span] = merge_windows(
            [hit(4)], window=2, section_lengths={SECTION_A: 8}, promote_ratio=0.5, max_chunks=20
        )

        assert (span.start, span.end) == (0, 7)

    def test_a_small_span_in_a_long_section_is_not_promoted(self) -> None:
        [span] = merge_windows(
            [hit(50)], window=2, section_lengths={SECTION_A: 200}, promote_ratio=0.5
        )

        assert (span.start, span.end) == (48, 52)

    def test_promotion_is_skipped_without_section_lengths(self) -> None:
        [span] = merge_windows([hit(4)], window=2)

        assert (span.start, span.end) == (2, 6)


class TestCapping:
    def test_an_over_long_span_is_truncated(self) -> None:
        hits = [hit(i) for i in range(0, 40, 3)]

        spans = merge_windows(hits, window=2, max_chunks=8)

        assert all(s.length <= 8 for s in spans)

    def test_truncation_centres_on_the_best_seed(self) -> None:
        hits = [hit(i, score=0.1) for i in range(0, 30, 2)]
        hits.append(hit(20, score=0.99))

        spans = merge_windows(hits, window=2, max_chunks=6)
        containing = [s for s in spans if s.contains(20)]

        assert containing, "the highest-scoring chunk must survive truncation"

    def test_a_truncated_span_keeps_at_least_one_seed(self) -> None:
        hits = [hit(i) for i in range(0, 40, 3)]

        spans = merge_windows(hits, window=2, max_chunks=4)

        assert all(s.seed_ids for s in spans)


class TestMergeProperties:
    @staticmethod
    def _hits(ordinals: list[int], sections: int) -> list[SeedHit]:
        section_ids = [SECTION_A, SECTION_B][:sections] or [SECTION_A]
        return [
            SeedHit(
                chunk_id=uuid4(),
                document_id=DOC_A,
                section_id=section_ids[i % len(section_ids)],
                ordinal=o,
                score=1.0 / (i + 1),
            )
            for i, o in enumerate(ordinals)
        ]

    @settings(max_examples=150, deadline=None)
    @given(
        st.lists(st.integers(0, 100), min_size=1, max_size=25),
        st.integers(0, 4),
        st.integers(1, 2),
    )
    def test_spans_within_a_section_never_overlap(
        self, ordinals: list[int], window: int, sections: int
    ) -> None:
        """Overlapping spans would show the generator the same text twice and
        let it cite one passage as two corroborating sources."""
        spans = merge_windows(self._hits(ordinals, sections), window=window, max_chunks=1000)

        by_section: dict[UUID, list[tuple[int, int]]] = {}
        for span in spans:
            by_section.setdefault(span.section_id, []).append((span.start, span.end))

        for ranges in by_section.values():
            ranges.sort()
            for (_, previous_end), (next_start, _) in pairwise(ranges):
                assert next_start > previous_end + 1

    @settings(max_examples=150, deadline=None)
    @given(st.lists(st.integers(0, 100), min_size=1, max_size=25), st.integers(0, 4))
    def test_every_hit_is_covered_by_some_span(self, ordinals: list[int], window: int) -> None:
        hits = self._hits(ordinals, sections=1)

        spans = merge_windows(hits, window=window, max_chunks=1000)

        for h in hits:
            assert any(s.section_id == h.section_id and s.contains(h.ordinal) for s in spans)

    @settings(max_examples=150, deadline=None)
    @given(
        st.lists(st.integers(0, 100), min_size=1, max_size=25),
        st.integers(0, 4),
        st.integers(2, 12),
    )
    def test_the_cap_is_never_exceeded(self, ordinals: list[int], window: int, cap: int) -> None:
        spans = merge_windows(self._hits(ordinals, sections=2), window=window, max_chunks=cap)

        assert all(s.length <= cap for s in spans)

    @settings(max_examples=150, deadline=None)
    @given(st.lists(st.integers(0, 100), min_size=1, max_size=25), st.integers(0, 4))
    def test_a_span_never_contains_more_than_one_section(
        self, ordinals: list[int], window: int
    ) -> None:
        hits = self._hits(ordinals, sections=2)

        spans = merge_windows(hits, window=window, max_chunks=1000)

        for span in spans:
            covered = {
                h.section_id
                for h in hits
                if span.contains(h.ordinal) and h.chunk_id in span.seed_ids
            }
            assert len(covered) <= 1

    @settings(max_examples=100, deadline=None)
    @given(st.lists(st.integers(0, 60), min_size=1, max_size=20), st.integers(0, 3))
    def test_start_never_exceeds_end(self, ordinals: list[int], window: int) -> None:
        spans = merge_windows(self._hits(ordinals, sections=1), window=window)

        assert all(s.start <= s.end for s in spans)

    @settings(max_examples=100, deadline=None)
    @given(st.lists(st.integers(0, 60), min_size=1, max_size=20))
    def test_input_order_does_not_change_the_result(self, ordinals: list[int]) -> None:
        hits = self._hits(ordinals, sections=1)

        forward = merge_windows(hits, window=2, max_chunks=1000)
        backward = merge_windows(list(reversed(hits)), window=2, max_chunks=1000)

        assert {(s.start, s.end) for s in forward} == {(s.start, s.end) for s in backward}
