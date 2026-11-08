"""Where a citation links: the DOI, else the source page, else nothing.

``source_path`` holds a URL for fetched pages and a local filesystem path for PDFs
ingested from disk, and a local path must never become a link.
"""

from __future__ import annotations

from myscholar.adapters.persistence.repository import _source_url


class TestSourceUrl:
    def test_a_doi_becomes_a_resolver_link(self) -> None:
        assert _source_url("10.1016/j.egyr.2022.09.179", None) == (
            "https://doi.org/10.1016/j.egyr.2022.09.179"
        )

    def test_the_doi_wins_over_the_file_it_was_read_from(self) -> None:
        url = _source_url("10.1234/abc", "/home/user/papers/motor.pdf")

        assert url == "https://doi.org/10.1234/abc"

    def test_a_page_links_to_itself(self) -> None:
        assert _source_url(None, "https://haseeb-arshad.me/projects") == (
            "https://haseeb-arshad.me/projects"
        )
        assert _source_url("", "http://example.com/x") == "http://example.com/x"

    def test_a_local_path_is_not_a_link(self) -> None:
        assert _source_url(None, "/home/user/myScholar-rag/papers/mmc.pdf") == ""
        assert _source_url(None, "papers/mmc.pdf") == ""

    def test_nothing_to_link_to_is_empty_not_a_guess(self) -> None:
        assert _source_url(None, None) == ""
        assert _source_url("", "") == ""
