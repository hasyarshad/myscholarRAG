"""Guards that catch a corpus defect at ingest rather than four phases later.

A paper with a stacked PDF text layer produced 796 chunks from 6 pages and sat
in the index unnoticed while its garbled fragments surfaced in unrelated
searches. It ingested cleanly, reported nothing unusual, and passed every
check, because nothing compared the output to the input.
"""

from __future__ import annotations

import pytest

from myscholar.ingest.pipeline import _SUSPICIOUS_CHUNKS_PER_PAGE


class TestDensityThreshold:
    def test_it_sits_above_a_normal_paper(self) -> None:
        """Measured across the corpus, healthy papers produce 8-20 chunks per
        page. The threshold has to clear that comfortably or every dense paper
        cries wolf."""
        assert _SUSPICIOUS_CHUNKS_PER_PAGE > 20

    def test_it_sits_below_the_defect_it_was_written_for(self) -> None:
        """The UAV paper produced 796 chunks over 6 pages — 132 per page."""
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
        """Every pair is a real document from this corpus, before or after the
        deduplication fix."""
        assert (chunks / pages > _SUSPICIOUS_CHUNKS_PER_PAGE) is flagged
