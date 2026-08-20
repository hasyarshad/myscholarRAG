"use client";

/**
 * Renders an answer with its citation markers turned into chips.
 *
 * The markers are the point: every factual sentence carries one, each resolves
 * to a real retrieved passage, and clicking it shows which paper, section and
 * pages the claim came from. Stripping them to make the prose "cleaner" would
 * discard the verification.
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

  // An open popover sits over the sentence underneath it, so it has to be
  // dismissable by the two gestures a reader will actually try — pressing
  // Escape and clicking away — rather than only by finding the same small chip
  // again. Listeners are attached only while one is open, so a long answer
  // full of citations does not carry a listener per marker.
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
    // Should be unreachable: the API only emits citations that resolved. Shown
    // rather than hidden, because silently dropping one would conceal a hole
    // in the verifier.
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
          {/* Linked only when the document resolved to somewhere real: a DOI
              for a paper, the page itself for the CV and site content. A
              citation with neither stays plain text, because a link that goes
              nowhere is worse than no link — the reader only discovers it
              after following it. */}
          {citation.source_url ? (
            <a
              className="chip-link"
              href={citation.source_url}
              target="_blank"
              rel="noreferrer noopener"
              // Closed on the way out. The link opens in a new tab, so this
              // page never navigates and nothing else would ever reset the
              // state — the reader comes back to the popover still covering
              // the sentence they were reading.
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

      {/* What `degraded` means, stated as it is rather than as it once was.
          This said "the reranking model was unavailable" long after the
          reranker had been measured harmful and removed from the system, so it
          reported the failure of a component that no longer existed. The flag
          now has two causes — generation itself failing, and a grounding check
          that would have deleted an otherwise verified answer — and what they
          share is that a check did not complete, not that anything was
          unavailable. */}
      {answer.degraded && (
        <p className="degraded">
          Answered with lower confidence: not every check completed. The
          citations shown were still verified against their passages.
        </p>
      )}
    </div>
  );
}
