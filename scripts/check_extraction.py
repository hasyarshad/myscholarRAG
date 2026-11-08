"""Check title and heading extraction across the corpus before a full ingestion.

    uv run python scripts/check_extraction.py            # a sample
    uv run python scripts/check_extraction.py --all      # every paper

Formula enrichment is off (headings do not depend on it), so each paper takes seconds.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.disable(logging.INFO)

from myscholar.adapters.extraction.docling_extractor import (  # noqa: E402
    DoclingExtractor,
    is_plausible_title,
)

PAPERS = sorted(Path("papers").glob("*.pdf"))
SAMPLE_INDICES = (0, 3, 8, 12, 17)


def select(argv: list[str]) -> list[Path]:
    if "--all" in argv:
        return PAPERS
    named = [a for a in argv[1:] if not a.startswith("--")]
    if named:
        return [p for p in PAPERS if any(n.lower() in p.name.lower() for n in named)]
    return [PAPERS[i] for i in SAMPLE_INDICES if i < len(PAPERS)]


def main(argv: list[str]) -> int:
    targets = select(argv)
    if not targets:
        print("no matching papers", file=sys.stderr)
        return 1

    extractor = DoclingExtractor(enrich_formulas=False)
    problems = 0

    for path in targets:
        doc = extractor.extract(path)
        heads = doc.headings
        print(f"\n=== {path.name[:66]}")
        print(f"  TITLE   : {doc.title[:88] or '(none captured)'}")
        print(f"  HEADINGS: {len(heads)} -> {[h[:38] for h in heads[:5]]}")

        if not doc.title:
            print("  !! no title captured")
            problems += 1
        elif not is_plausible_title(doc.title):
            print("  !! title looks like publisher furniture, not a title")
            problems += 1

        repeated = [h for h in heads if doc.title and h.casefold() == doc.title.casefold()]
        if repeated:
            print(f"  !! title repeated as a heading: {repeated[0][:66]}")
            problems += 1

    print(f"\n{len(targets)} paper(s) checked")
    print("OK: no title/heading confusion" if not problems else f"{problems} PROBLEM(S)")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
