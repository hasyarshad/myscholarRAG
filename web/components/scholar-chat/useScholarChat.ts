"use client";

/**
 * The SSE client. Frames are parsed from a `fetch` body because `EventSource` is
 * GET-only and cannot send the question and session id as a POST body.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import type {
  AnswerEvent,
  CorpusInfo,
  ErrorEvent,
  StageEvent,
  Turn,
} from "./types";

interface Frame {
  event: string;
  data: string;
}

/**
 * Split buffered response text into complete SSE frames.
 *
 * Returns the frames and the unparsed remainder, since a read can end mid-frame.
 */
function parseFrames(buffer: string): { frames: Frame[]; rest: string } {
  const frames: Frame[] = [];
  const blocks = buffer.split("\n\n");
  const rest = blocks.pop() ?? "";

  for (const block of blocks) {
    let event = "";
    const data: string[] = [];
    for (const line of block.split("\n")) {
      if (line.startsWith("event: ")) event = line.slice(7);
      else if (line.startsWith("data: ")) data.push(line.slice(6));
      // Lines starting with ":" are heartbeat comments and are ignored.
    }
    if (event && data.length > 0) frames.push({ event, data: data.join("\n") });
  }

  return { frames, rest };
}

export interface UseScholarChat {
  turns: Turn[];
  corpus: CorpusInfo | null;
  busy: boolean;
  ask: (question: string) => Promise<void>;
  stop: () => void;
  reset: () => void;
}

export function useScholarChat(apiBase: string): UseScholarChat {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [corpus, setCorpus] = useState<CorpusInfo | null>(null);
  const [busy, setBusy] = useState(false);
  const abort = useRef<AbortController | null>(null);

  // Issued by the server on the first turn and echoed on later ones, so a
  // follow-up such as "and 2024?" resolves. Not persisted across page loads.
  const session = useRef<string | null>(null);

  useEffect(() => {
    fetch(`${apiBase}/v1/corpus`)
      .then((r) => (r.ok ? r.json() : null))
      .then(setCorpus)
      .catch(() => setCorpus(null));
  }, [apiBase]);

  // Abort in-flight work on unmount, so the server can stop answering.
  useEffect(() => () => abort.current?.abort(), []);

  const update = useCallback((id: string, patch: Partial<Turn>) => {
    setTurns((current) =>
      current.map((turn) => (turn.id === id ? { ...turn, ...patch } : turn)),
    );
  }, []);

  const ask = useCallback(
    async (question: string) => {
      const trimmed = question.trim();
      if (!trimmed || busy) return;

      const id = crypto.randomUUID();
      setTurns((current) => [
        ...current,
        { id, question: trimmed, stages: [], pending: true },
      ]);
      setBusy(true);

      const controller = new AbortController();
      abort.current = controller;

      try {
        const response = await fetch(`${apiBase}/v1/ask`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          // A ref, so sending it does not depend on a re-render.
          body: JSON.stringify({
            question: trimmed,
            session_id: session.current,
          }),
          signal: controller.signal,
        });

        // Failures knowable before the stream opens arrive as a real status.
        // Only mid-stream failures come through as an error event.
        if (!response.ok || !response.body) {
          const body = await response.json().catch(() => ({}));
          update(id, {
            pending: false,
            error: {
              code: body.code ?? String(response.status),
              message:
                body.message ??
                (response.status === 429
                  ? "Too many questions just now. Try again in a moment."
                  : "The assistant is unavailable."),
              retryable: response.status === 429 || response.status >= 500,
            },
          });
          return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        const stages: StageEvent[] = [];

        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const { frames, rest } = parseFrames(buffer);
          buffer = rest;

          for (const frame of frames) {
            if (frame.event === "session") {
              session.current = (
                JSON.parse(frame.data) as { session_id: string }
              ).session_id;
            } else if (frame.event === "stage") {
              stages.push(JSON.parse(frame.data) as StageEvent);
              update(id, { stages: [...stages] });
            } else if (frame.event === "answer") {
              update(id, {
                answer: JSON.parse(frame.data) as AnswerEvent,
                pending: false,
              });
            } else if (frame.event === "error") {
              update(id, {
                error: JSON.parse(frame.data) as ErrorEvent,
                pending: false,
              });
            }
          }
        }

        // The stream ended without an answer: a dropped connection, not a refusal.
        setTurns((current) =>
          current.map((turn) =>
            turn.id === id && turn.pending
              ? {
                  ...turn,
                  pending: false,
                  error: {
                    code: "incomplete",
                    message: "The connection ended before an answer arrived.",
                    retryable: true,
                  },
                }
              : turn,
          ),
        );
      } catch (caught) {
        if ((caught as Error).name !== "AbortError") {
          update(id, {
            pending: false,
            error: {
              code: "network",
              message: "Could not reach the assistant.",
              retryable: true,
            },
          });
        } else {
          update(id, { pending: false });
        }
      } finally {
        setBusy(false);
        abort.current = null;
      }
    },
    [apiBase, busy, update],
  );

  const stop = useCallback(() => abort.current?.abort(), []);
  const reset = useCallback(() => {
    abort.current?.abort();
    setTurns([]);
    // A new session, so the next question does not resolve against cleared turns.
    session.current = null;
  }, []);

  return { turns, corpus, busy, ask, stop, reset };
}
