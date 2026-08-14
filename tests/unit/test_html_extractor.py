"""Parsing the portfolio site into blocks.

The CV is the document these rules were got wrong on, and it is the one where
being wrong is hardest to see: a page with half its content missing still
indexes cleanly, still answers biographical questions plausibly, and reports
nothing unusual.
"""

from __future__ import annotations

import pytest

from myscholar.adapters.extraction.html_extractor import _Reader, is_content


class TestWhatCountsAsContent:
    @pytest.mark.parametrize(
        "text",
        [
            "Jun 2025",
            "2023–2025",  # noqa: RUF001 - en dashes are what the CV actually uses
            "$10,000.",
            "R",
            "Git",
            "PyTorch",
            "NSGA-II",
            "University of Alberta",
            "2022–23, 2025–26",  # noqa: RUF001 - as above
        ],
    )
    def test_short_strings_on_a_cv_are_content(self, text: str) -> None:
        """A 24-character minimum discarded 99 of 200 blocks on /cv — every
        date range, every funder, every award amount, and the whole skills
        list. Short is not the same as empty; on a CV the short strings are
        the data.
        """
        assert is_content(text)

    @pytest.mark.parametrize("text", ["01", "02", "17", "doi", "DOI", "pdf", "iv", "—", "  "])
    def test_list_ordinals_and_bare_labels_are_not(self, text: str) -> None:
        """What the length rule was really aimed at: the numbers a styled list
        prints beside each entry, and bare link labels."""
        assert not is_content(text)

    @pytest.mark.parametrize("text", ["Skip to main content", "Menu", "© 2026", "Back to top"])
    def test_boilerplate_is_not(self, text: str) -> None:
        assert not is_content(text)


class TestHeadingHierarchy:
    def test_only_h1_and_h2_open_a_section(self) -> None:
        """h3 names an item within a section — a degree, a role, an award.

        Treating it as a boundary put "Education" and "PhD, University of
        Alberta" at the same level, and the MSc entry was then classified by
        content and filed under methods.
        """
        reader = _Reader()
        reader.feed(
            "<html><body><h1>CV</h1><h2>Education</h2>"
            "<h3>PhD, University of Alberta</h3><p>Fault diagnosis for electric drives.</p>"
            "</body></html>"
        )

        headings = [b.text for b in reader.blocks if b.heading]

        assert headings == ["Education"]

    def test_the_first_h1_is_the_page_title_not_a_section(self) -> None:
        reader = _Reader()
        reader.feed(
            "<html><body><h1>CV</h1><h2>Awards</h2><p>Something long enough.</p></body></html>"
        )

        assert reader.title == "CV"


class TestAwardsShapedMarkup:
    def test_a_date_beside_an_award_survives(self) -> None:
        """The shape the CV actually uses. Losing the date leaves the award
        with no year, so "what did he win in 2025" cannot be answered from a
        section that plainly contains the answer.
        """
        reader = _Reader()
        reader.feed(
            "<html><body><h2>Awards</h2>"
            "<li><span>Jun 2025</span></li>"
            "<li>Best Paper Presentation Award</li>"
            "<li>IEEE International Symposium on Industrial Electronics (ISIE)</li>"
            "</body></html>"
        )

        texts = [b.text for b in reader.blocks]

        assert "Jun 2025" in texts
        assert "Best Paper Presentation Award" in texts
