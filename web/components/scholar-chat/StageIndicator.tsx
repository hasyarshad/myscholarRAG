"use client";

/** The pipeline's stages as they arrive, with the server's detail and timing. */

import type { StageEvent } from "./types";

/**
 * Labels name the stage; the server's detail says what it found (only the server
 * knows whether it searched papers, the CV or the site).
 */
const LABELS: Record<string, string> = {
  classifying: "Reading the question",
  retrieving: "Searching",
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
            <span className="stage-label">{LABELS[stage.stage] ?? stage.stage}</span>
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
