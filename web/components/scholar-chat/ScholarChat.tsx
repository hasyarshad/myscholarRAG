"use client";

import { useState } from "react";

import "./scholar-chat.css";
import { AnswerBody } from "./AnswerBody";
import { StageIndicator } from "./StageIndicator";
import { useScholarChat } from "./useScholarChat";
import type { Citation } from "./types";

const MAX_QUESTION = 1200;

export function ScholarChat({
  // Same-origin by default, through the route handler in app/api/scholar. The
  // backend URL and the shared secret live there, on the server; a client
  // component cannot read either, and `process.env` here would only ever return
  // undefined for anything not prefixed NEXT_PUBLIC_ — which is exactly what a
  // secret must never be.
  apiBase = "/api/scholar",
}: {
  apiBase?: string;
}) {
  const { turns, corpus, busy, ask, stop } = useScholarChat(apiBase);
  const [draft, setDraft] = useState("");

  // Every source cited so far, deduplicated — a reader wants the reading list
  // as much as the answers.
  const sources = new Map<string, Citation>();
  for (const turn of turns) {
    for (const citation of turn.answer?.citations ?? []) {
      sources.set(citation.document_title, citation);
    }
  }

  const submit = (question: string) => {
    setDraft("");
    void ask(question);
  };

  return (
    <section className="scholar">
      <header className="scholar-header">
        <h2>Ask about the research</h2>
        {/* Deliberately no counts and no year range. They were accurate and
            they aged badly: every published paper made the sentence wrong until
            someone remembered to re-ingest, and a number that is sometimes stale
            undermines the one claim this widget makes for itself — that what it
            says can be checked. The scope is described instead of measured. */}
        <p className="scholar-scope">
          Research papers, CV and project write-ups. Answers cite the passage
          they came from; if the corpus does not cover something, it says so.
        </p>
      </header>

      {turns.length === 0 && corpus && (
        <ul className="starters">
          {corpus.suggestions.map((suggestion) => (
            <li key={suggestion}>
              <button type="button" onClick={() => submit(suggestion)}>
                {suggestion}
              </button>
            </li>
          ))}
        </ul>
      )}

      <ol className="turns">
        {turns.map((turn) => (
          <li key={turn.id} className="turn">
            <p className="question">{turn.question}</p>

            <StageIndicator stages={turn.stages} pending={turn.pending} />

            {turn.answer && <AnswerBody answer={turn.answer} />}

            {turn.error && (
              <p className="error" role="alert">
                {turn.error.message}
                {turn.error.retryable && (
                  <button type="button" onClick={() => submit(turn.question)}>
                    Try again
                  </button>
                )}
              </p>
            )}
          </li>
        ))}
      </ol>

      {sources.size > 0 && (
        <details className="sources">
          <summary>{sources.size} paper(s) cited</summary>
          <ul>
            {[...sources.values()].map((citation) => (
              <li key={citation.document_title}>{citation.document_title}</li>
            ))}
          </ul>
        </details>
      )}

      <form
        className="composer"
        onSubmit={(event) => {
          event.preventDefault();
          submit(draft);
        }}
      >
        <label className="visually-hidden" htmlFor="scholar-input">
          Your question
        </label>
        <input
          id="scholar-input"
          value={draft}
          maxLength={MAX_QUESTION}
          disabled={busy}
          placeholder="What method is used for fault detection?"
          onChange={(event) => setDraft(event.target.value)}
        />
        {busy ? (
          <button type="button" onClick={stop}>
            Stop
          </button>
        ) : (
          <button type="submit" disabled={!draft.trim()}>
            Ask
          </button>
        )}
      </form>

      {/* Live region announces completion rather than every stage: stage-level
          updates make a screen reader unusable. */}
      <p className="visually-hidden" aria-live="polite">
        {busy ? "Working" : turns.at(-1)?.answer ? "Answer ready" : ""}
      </p>
    </section>
  );
}
