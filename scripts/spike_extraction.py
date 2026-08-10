"""Measure what Docling actually produces for this corpus.

Run before committing to a chunking strategy:

    uv run python scripts/spike_extraction.py

These are two-column IEEE papers dense with equations and figures, which is a
harder case than the single-column arXiv preprints these pipelines are usually
demonstrated on. The questions worth answering with evidence rather than
assumption:

1. Are section headings detected, and can they be mapped to canonical kinds?
2. What happens to inline and display equations?
3. Do tables survive as structure, or collapse into noise?
4. Is the two-column reading order correct, or is text interleaved across
   columns?
5. How much of each paper is references — that is, how much we gain by
   dropping them?

Writes a report to data/spike/extraction-report.md plus the raw markdown for
each paper, so the output can be read rather than taken on trust.
"""

from __future__ import annotations

import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

PAPERS = Path("papers")
OUT = Path("data/spike")

#: Chosen to span the corpus rather than to flatter the extractor: one dense
#: control-theory paper, one with substantial tables, and one social-science
#: outlier with an entirely different structure.
SAMPLE = [
    "Weighting factors optimization of model predictive torque control of "
    "induction motor using NSGA-II with TOPSIS decision making.pdf",
    "Leveraging Meta-Learner-Based Stacking Framework for Stator Winding "
    "Inter-Turn Fault Detection of LSPMSM.pdf",
    "Exploring crowdsourced content moderation through lens of Reddit during COVID-19.pdf",
]

#: Headings we expect to find in an IEEE paper, in the forms they actually
#: appear: "II. METHODOLOGY", "3.1 Results", "Abstract".
HEADING_HINTS = (
    "abstract",
    "introduction",
    "related",
    "background",
    "method",
    "methodolog",
    "propos",
    "experiment",
    "result",
    "discussion",
    "conclusion",
    "reference",
    "acknowledg",
    "appendix",
    "simulation",
    "implementation",
)

#: Symbols that only appear in mathematics. Their presence in body text after
#: extraction tells us whether equations became readable text or debris.
MATH_SIGNALS = re.compile(
    r"[\u2211\u222b\u2202\u221a\u2264\u2265\u2248\u2260\u2208\u221e\u03bb\u03bc\u03c3\u03c9\u03b8\u03b1\u03b2\u03b3\u03b4\u03a6\u03a8\u22c5\u00d7\u00b1]|\$.*?\$|\\[a-zA-Z]+\{"
)


@dataclass
class PaperReport:
    name: str
    ok: bool = True
    error: str = ""
    seconds: float = 0.0
    pages: int = 0
    chars: int = 0
    headings: list[str] = field(default_factory=list)
    matched_headings: list[str] = field(default_factory=list)
    tables: int = 0
    table_rows: list[int] = field(default_factory=list)
    math_lines: int = 0
    math_examples: list[str] = field(default_factory=list)
    reference_chars: int = 0
    short_line_ratio: float = 0.0
    body_sample: str = ""


def _looks_like_heading(text: str) -> bool:
    stripped = text.strip().lower()
    stripped = re.sub(r"^[ivxlc]+[.)]\s*|^\d+([.)]\d*)*\s*", "", stripped)
    return any(hint in stripped for hint in HEADING_HINTS)


def _converter() -> object:
    """A converter tuned for born-digital publisher PDFs.

    OCR is off: these papers have a real text layer, so running OCR would cost
    minutes per document and introduce recognition errors into text that was
    already correct. Table structure is on, because several papers carry their
    headline results in tables.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions()
    options.do_ocr = False
    options.do_table_structure = True
    options.generate_page_images = False

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def analyse(path: Path) -> PaperReport:
    report = PaperReport(name=path.name)
    started = time.monotonic()
    try:
        result = _converter().convert(str(path))  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001 - a spike reports failures, not raises
        report.ok = False
        report.error = f"{type(exc).__name__}: {exc}"
        report.seconds = time.monotonic() - started
        return report

    report.seconds = time.monotonic() - started
    doc = result.document
    markdown = doc.export_to_markdown()

    report.pages = len(getattr(doc, "pages", {}) or {})
    report.chars = len(markdown)

    # Headings, as Docling labelled them.
    for item, _level in doc.iterate_items():
        label = str(getattr(item, "label", ""))
        if "title" in label.lower() or "header" in label.lower():
            text = (getattr(item, "text", "") or "").strip()
            if text:
                report.headings.append(text)
    report.matched_headings = [h for h in report.headings if _looks_like_heading(h)]

    # Tables: did structure survive?
    tables = getattr(doc, "tables", []) or []
    report.tables = len(tables)
    for table in tables:
        data = getattr(table, "data", None)
        rows = getattr(data, "num_rows", 0) if data else 0
        report.table_rows.append(int(rows))

    # Mathematics: how much survived, and in what state?
    for line in markdown.splitlines():
        if MATH_SIGNALS.search(line):
            report.math_lines += 1
            if len(report.math_examples) < 5 and len(line.strip()) > 10:
                report.math_examples.append(line.strip()[:160])

    # References: how much of the document would dropping them remove?
    lowered = markdown.lower()
    for marker in ("\n## references", "\n# references", "\nreferences\n", "\n## bibliography"):
        idx = lowered.find(marker)
        if idx != -1:
            report.reference_chars = len(markdown) - idx
            break

    # Two-column reading order: badly interleaved columns produce a high
    # proportion of very short lines, because each visual line is emitted
    # separately instead of being joined into paragraphs.
    lines = [ln for ln in markdown.splitlines() if ln.strip()]
    if lines:
        short = sum(1 for ln in lines if len(ln.strip()) < 40)
        report.short_line_ratio = short / len(lines)

    # A body sample from the middle, past the title block.
    midpoint = len(markdown) // 2
    report.body_sample = markdown[midpoint : midpoint + 700]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{path.stem[:60]}.md").write_text(markdown, encoding="utf-8")
    return report


def render(reports: list[PaperReport]) -> str:
    lines = [
        "# Docling extraction spike",
        "",
        "Measured before committing to a chunking strategy. Raw markdown for each",
        "paper is alongside this file.",
        "",
        "## Summary",
        "",
        "| Paper | ok | pages | chars | headings (matched) | tables |"
        " math lines | refs % | short-line ratio | secs |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        if not r.ok:
            lines.append(f"| {r.name[:40]} | FAIL | | | | | | | | {r.seconds:.0f} |")
            continue
        refs_pct = (r.reference_chars / r.chars * 100) if r.chars else 0.0
        lines.append(
            f"| {r.name[:40]} | yes | {r.pages} | {r.chars:,} | "
            f"{len(r.headings)} ({len(r.matched_headings)}) | {r.tables} | "
            f"{r.math_lines} | {refs_pct:.0f}% | {r.short_line_ratio:.2f} | {r.seconds:.0f} |"
        )

    ok = [r for r in reports if r.ok]
    if ok:
        ref_pcts = [(r.reference_chars / r.chars * 100) for r in ok if r.chars]
        lines += [
            "",
            f"Mean extraction time: {statistics.mean(r.seconds for r in ok):.1f}s per paper.",
            f"Mean references share: {statistics.mean(ref_pcts):.0f}% of extracted text."
            if ref_pcts
            else "References section not detected in any sample.",
            "",
        ]

    for r in reports:
        lines += ["", "---", "", f"## {r.name}", ""]
        if not r.ok:
            lines += [f"**Extraction failed.** `{r.error}`", ""]
            continue
        lines += [
            f"- pages: {r.pages}, characters: {r.chars:,}, time: {r.seconds:.1f}s",
            f"- tables: {r.tables}" + (f" (rows: {r.table_rows})" if r.table_rows else ""),
            f"- short-line ratio: {r.short_line_ratio:.2f}"
            " (high values suggest interleaved columns)",
            "",
            "### Headings detected",
            "",
        ]
        lines += [f"- `{h}`" for h in r.headings[:30]] or ["- none"]
        lines += ["", "### Lines containing mathematics", ""]
        if r.math_examples:
            lines += [f"```\n{ex}\n```" for ex in r.math_examples]
        else:
            lines.append("None found — equations were probably dropped entirely.")
        lines += [
            "",
            "### Body sample (from the middle of the document)",
            "",
            "```",
            r.body_sample,
            "```",
        ]

    return "\n".join(lines)


def main() -> int:
    if not PAPERS.is_dir():
        print(f"no papers directory at {PAPERS.resolve()}", file=sys.stderr)
        return 1

    reports = []
    for name in SAMPLE:
        path = PAPERS / name
        if not path.exists():
            print(f"skipping (not found): {name}", file=sys.stderr)
            continue
        print(f"extracting: {name[:70]}...", flush=True)
        report = analyse(path)
        status = "ok" if report.ok else f"FAILED ({report.error[:60]})"
        print(f"  {status} in {report.seconds:.1f}s", flush=True)
        reports.append(report)

    if not reports:
        print("nothing extracted", file=sys.stderr)
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "extraction-report.md"
    out.write_text(render(reports), encoding="utf-8")
    print(f"\nreport: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
