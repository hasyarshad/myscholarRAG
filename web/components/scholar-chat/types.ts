/**
 * The wire contract, mirroring src/myscholar/api/schemas.py.
 *
 * Hand-written here rather than generated, because the project has one
 * consumer and a generation step would be more machinery than it saves. If a
 * second consumer appears, generate these from the OpenAPI schema instead —
 * two hand-maintained copies of a contract drift.
 */

export type StageName =
  | "classifying"
  | "retrieving"
  | "grading"
  | "generating"
  | "verifying"
  | "done";

export interface StageEvent {
  stage: StageName | string;
  detail: string;
  elapsed_ms: number;
}

export interface Citation {
  label: string;
  document_title: string;
  section_heading: string;
  page_start: number;
  page_end: number;
  slug: string;
}

export interface AnswerEvent {
  text: string;
  citations: Citation[];
  refused: boolean;
  refusal_reason: string | null;
  nearest_topics: string[];
  confidence: number;
  /** A component was unavailable — currently reranking. Shown, not hidden. */
  degraded: boolean;
}

export interface ErrorEvent {
  code: string;
  message: string;
  retryable: boolean;
}

export interface CorpusInfo {
  documents: number;
  chunks: number;
  years: number[];
  suggestions: string[];
}

export interface Turn {
  id: string;
  question: string;
  stages: StageEvent[];
  answer?: AnswerEvent;
  error?: ErrorEvent;
  /** True while the pipeline is still working on this turn. */
  pending: boolean;
}
