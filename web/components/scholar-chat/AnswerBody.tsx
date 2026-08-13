"use client";

/**
 * Renders an answer with its citation markers turned into chips.
 *
 * The markers are the point: every factual sentence carries one, each resolves
 * to a real retrieved passage, and clicking it shows which paper, section and
 * pages the claim came from. Stripping them to make the prose "cleaner" would
 * discard the verification.
 */

import { useState } from "react";
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

  if (!citation) {
    // Should be unreachable: the API only emits citations that resolved. Shown
    // rather than hidden, because silently dropping one would conceal a hole
    // in the verifier.
    return <sup className="chip chip-unknown">[{label}?]</sup>;
  }

  return (
    <span className="chip-wrap">
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
          <strong>{citation.document_title}</strong>
          <span className="chip-meta">
            {citation.section_heading} · pages {citation.page_start}–
            {citation.page_end}
          </span>
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

      {answer.degraded && (
        <p className="degraded">
          Answered in reduced-precision mode — the reranking model was
          unavailable.
        </p>
      )}
    </div>
  );
}
