"use client";

/**
 * The pipeline's progress, live.
 *
 * This is the most informative part of the interface. A visitor watching
 * "verifying — 2 citation issues" followed by another "generating" is seeing
 * the system reject its own draft, which is the thing that makes it
 * trustworthy and is invisible in a finished answer.
 */

import type { StageEvent } from "./types";

const LABELS: Record<string, string> = {
  classifying: "Reading the question",
  retrieving: "Searching the papers",
  grading: "Checking what was found",
  generating: "Writing the answer",
  verifying: "Verifying every citation",
  done: "Done",
};

function seconds(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
}

export function StageIndicator({
  stages,
  pending,
}: {
  stages: StageEvent[];
  pending: boolean;
}) {
  if (stages.length === 0 && !pending) return null;

  // Count regenerations so the retry is legible rather than looking like a
  // stutter — this is the adaptive loop's whole visible behaviour.
  const attempts = stages.filter((s) => s.stage === "generating").length;

  return (
    <ol className="stages" aria-label="Progress">
      {stages.map((stage, index) => {
        const last = index === stages.length - 1;
        return (
          <li
            key={`${stage.stage}-${index}`}
            className={`stage${last && pending ? " active" : ""}`}
          >
            <span className="stage-dot" aria-hidden="true" />
            <span className="stage-label">
              {LABELS[stage.stage] ?? stage.stage}
              {stage.stage === "generating" && attempts > 1 && (
                <em className="stage-retry">
                  {" "}
                  attempt {stages.slice(0, index + 1).filter((s) => s.stage === "generating").length}
                </em>
              )}
            </span>
            {stage.detail && <span className="stage-detail">{stage.detail}</span>}
            <span className="stage-time">{seconds(stage.elapsed_ms)}</span>
          </li>
        );
      })}
      {pending && stages.length === 0 && (
        <li className="stage active">
          <span className="stage-dot" aria-hidden="true" />
          <span className="stage-label">Starting…</span>
        </li>
      )}
    </ol>
  );
}
