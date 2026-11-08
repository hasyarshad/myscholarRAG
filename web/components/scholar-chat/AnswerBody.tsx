"use client";

/**
 * Renders an answer with its citation markers turned into chips that show the
 * paper, section and pages each claim came from.
 */

import { useEffect, useRef, useState } from "react";
import type { AnswerEvent, Citation } from "./types";

const MARKER = /\[\s*(C\d+(?:\s*,\s*C\d+)*)\s*\]/gi;

function CitationChip({
  label,
  citation,
}: {
  label: string;
  citation: Citation | undefined;
}) {
  const [open, setOpen] = useState(false);
  const wrap = useRef<HTMLSpanElement>(null);

  // Close on Escape or a click outside. Listeners exist only while open, so a
  // long answer does not carry one per marker.
  useEffect(() => {
    if (!open) return;

    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    const onPointer = (event: PointerEvent) => {
      if (!wrap.current?.contains(event.target as Node)) setOpen(false);
    };

    document.addEventListener("keydown", onKey);
    document.addEventListener("pointerdown", onPointer);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("pointerdown", onPointer);
    };
  }, [open]);

  if (!citation) {
    // Unreachable if the verifier works; shown so a gap in it stays visible.
    return <sup className="chip chip-unknown">[{label}?]</sup>;
  }

  return (
    <span className="chip-wrap" ref={wrap}>
      <button
        type="button"
        className="chip"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        title={citation.document_title}
      >
        {label}
      </button>
      {open && (
        <span className="chip-popover" role="note">
          {/* Linked to a DOI or the page itself; plain text when neither exists. */}
          {citation.source_url ? (
            <a
              className="chip-link"
              href={citation.source_url}
              target="_blank"
              rel="noreferrer noopener"
              // The link opens a new tab, so close the popover on the way out.
              onClick={() => setOpen(false)}
            >
              {citation.document_title}
              <span aria-hidden="true"> ↗</span>
            </a>
          ) : (
            <strong>{citation.document_title}</strong>
          )}
          <span className="chip-meta">
            {citation.section_heading} · pages {citation.page_start}–
            {citation.page_end}
          </span>
          {citation.source_url.startsWith("https://doi.org/") && (
            <span className="chip-meta chip-doi">
              doi:{citation.source_url.slice("https://doi.org/".length)}
            </span>
          )}
        </span>
      )}
    </span>
  );
}

export function AnswerBody({ answer }: { answer: AnswerEvent }) {
  const byLabel = new Map(answer.citations.map((c) => [c.label.toUpperCase(), c]));

  const parts: React.ReactNode[] = [];
  let cursor = 0;
  let key = 0;

  for (const match of answer.text.matchAll(MARKER)) {
    const at = match.index ?? 0;
    if (at > cursor) parts.push(answer.text.slice(cursor, at));

    for (const label of match[1].split(",").map((l) => l.trim().toUpperCase())) {
      parts.push(
        <CitationChip key={`c-${key++}`} label={label} citation={byLabel.get(label)} />,
      );
    }
    cursor = at + match[0].length;
  }
  if (cursor < answer.text.length) parts.push(answer.text.slice(cursor));

  return (
    <div className={`answer${answer.refused ? " answer-refused" : ""}`}>
      <p>{parts}</p>

      {answer.refused && answer.nearest_topics.length > 0 && (
        <p className="nearest">
          Closest work here:{" "}
          {answer.nearest_topics.map((topic, i) => (
            <span key={topic}>
              {i > 0 && ", "}
              <em>{topic}</em>
            </span>
          ))}
        </p>
      )}

      {/* Set when generation failed or grounding was inconclusive on a verified answer. */}
      {answer.degraded && (
        <p className="degraded">
          Answered with lower confidence: not every check completed. The
          citations shown were still verified against their passages.
        </p>
      )}
    </div>
  );
}
