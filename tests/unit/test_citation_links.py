"""Where a citation sends a reader.

A citation that cannot be followed is only half a citation, and the rule for
choosing the destination is small enough to state exactly: a DOI if the document
has one, the page it came from otherwise, and nothing at all rather than a link
that fails.

The last branch is the one worth a test. ``source_path`` holds a URL for pages
fetched from the site and a *local filesystem path* for PDFs ingested from disk,
so a version of this that returned it unconditionally would put
``/home/haseeb/papers/foo.pdf`` in front of a visitor — a link that resolves on
exactly one machine in the world.
"""

from __future__ import annotations

from myscholar.adapters.persistence.repository import _source_url


class TestSourceUrl:
    def test_a_doi_becomes_a_resolver_link(self) -> None:
        assert _source_url("10.1016/j.egyr.2022.09.179", None) == (
            "https://doi.org/10.1016/j.egyr.2022.09.179"
        )

    def test_the_doi_wins_over_the_file_it_was_read_from(self) -> None:
        """Both are present for every ingested paper; only one is citable."""
        url = _source_url("10.1234/abc", "/home/haseeb/papers/motor.pdf")

        assert url == "https://doi.org/10.1234/abc"

    def test_a_page_links_to_itself(self) -> None:
        assert _source_url(None, "https://haseeb-arshad.me/projects") == (
            "https://haseeb-arshad.me/projects"
        )
        assert _source_url("", "http://example.com/x") == "http://example.com/x"

    def test_a_local_path_is_not_a_link(self) -> None:
        """The failure this exists to prevent: a link only the author can open."""
        assert _source_url(None, "/home/haseeb/myScholar-rag/papers/mmc.pdf") == ""
        assert _source_url(None, "papers/mmc.pdf") == ""

    def test_nothing_to_link_to_is_empty_not_a_guess(self) -> None:
        assert _source_url(None, None) == ""
        assert _source_url("", "") == ""
