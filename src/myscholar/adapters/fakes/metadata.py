"""A registry with a fixed set of records."""

from __future__ import annotations

from myscholar.ports.metadata import MetadataResolutionError, MetadataResolver, ResolvedMetadata


class FakeMetadataResolver(MetadataResolver):
    """Answers from a dict, so sync can be tested without a network.

    ``fail_on`` makes a DOI raise rather than return nothing, since the two
    outcomes must be handled differently: an unknown DOI falls back to
    extraction, while an unreachable registry must not silently rewrite a
    corpus with whatever it did manage to fetch.
    """

    def __init__(
        self,
        records: dict[str, ResolvedMetadata] | None = None,
        *,
        fail_on: set[str] | None = None,
    ) -> None:
        self.records = records or {}
        self.fail_on = fail_on or set()
        self.calls: list[str] = []

    async def resolve(self, doi: str) -> ResolvedMetadata | None:
        cleaned = doi.strip().rstrip(".,;").removeprefix("https://doi.org/")
        self.calls.append(cleaned)
        if cleaned in self.fail_on:
            msg = f"registry unreachable for {cleaned}"
            raise MetadataResolutionError(msg)
        return self.records.get(cleaned)
