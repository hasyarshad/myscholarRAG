"""The ingest guard that flags a defective document by its chunk density.

A PDF with a stacked text layer produced 796 chunks from 6 pages and ingested without error.
"""

from __future__ import annotations

import pytest

from myscholar.ingest.pipeline import _SUSPICIOUS_CHUNKS_PER_PAGE


class TestDensityThreshold:
    def test_it_sits_above_a_normal_paper(self) -> None:
        # Healthy papers in the corpus produce 8-20 chunks per page.
        assert _SUSPICIOUS_CHUNKS_PER_PAGE > 20

    def test_it_sits_below_the_defect_it_was_written_for(self) -> None:
        # The UAV paper produced 796 chunks over 6 pages, 132 per page.
        assert _SUSPICIOUS_CHUNKS_PER_PAGE < 796 / 6

    @pytest.mark.parametrize(
        ("chunks", "pages", "flagged"),
        [
            (155, 15, False),
            (133, 16, False),
            (140, 6, False),
            (796, 6, True),
            (272, 6, True),
        ],
    )
    def test_it_separates_the_real_documents(self, chunks: int, pages: int, flagged: bool) -> None:
        # Every pair is a real document from this corpus.
        assert (chunks / pages > _SUSPICIOUS_CHUNKS_PER_PAGE) is flagged
