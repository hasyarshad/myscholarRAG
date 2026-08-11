"""Check title/heading separation across several papers, quickly.

Formula enrichment is off: headings do not depend on it, and leaving it on makes
each paper take two minutes instead of a few seconds. Run before committing to a
full ingestion, because a title misfiled as a section is only visible here.
"""

import logging
from pathlib import Path

logging.disable(logging.INFO)

from myscholar.adapters.extraction.docling_extractor import DoclingExtractor  # noqa: E402

PAPERS = sorted(Path("papers").glob("*.pdf"))
SAMPLE = [PAPERS[i] for i in (0, 3, 8, 12, 17) if i < len(PAPERS)]

extractor = DoclingExtractor(enrich_formulas=False)
problems = 0

for path in SAMPLE:
    doc = extractor.extract(path)
    heads = doc.headings
    print(f"\n=== {path.name[:64]}")
    print(f"  TITLE   : {doc.title[:88] or '(none captured)'}")
    print(f"  HEADINGS: {len(heads)} -> {[h[:40] for h in heads[:6]]}")

    if not doc.title:
        print("  !! no title captured")
        problems += 1
    repeated = [h for h in heads if doc.title and h.casefold() == doc.title.casefold()]
    if repeated:
        print(f"  !! title repeated as a heading: {repeated[0][:70]}")
        problems += 1

print(f"\n{'OK — no title/heading confusion' if not problems else f'{problems} PROBLEM(S)'}")
