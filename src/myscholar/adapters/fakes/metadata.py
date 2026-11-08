"""A registry with a fixed set of records."""

from __future__ import annotations

from myscholar.ports.metadata import MetadataResolutionError, MetadataResolver, ResolvedMetadata


class FakeMetadataResolver(MetadataResolver):
    """Answers from a dict, so sync can be tested without a network.

    A DOI in ``fail_on`` raises instead of returning None. An unknown DOI falls
    back to extraction; an unreachable registry must not partially rewrite the corpus.
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
