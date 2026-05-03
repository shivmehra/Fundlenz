export type Role = "user" | "assistant";

export type MatchKind = "exact" | "field" | "semantic";

export interface Source {
  filename: string;
  page: number | null;
  score: number;          // 0-1 confidence (1.00 = exact id, 0.95 = field, else raw cosine)
  match?: MatchKind;
  type?: string;          // chunk_type — row, entity, enumeration, ...
  canonical_id?: string | null;
  rank_score?: number;    // internal ranking score, kept for debugging
}

export interface ChartPayload {
  text: string;
  chart_spec: { data: unknown[]; layout: Record<string, unknown> };
}

export type ConfidenceTier = "deterministic" | "grounded" | "semantic";

export interface Confidence {
  tier: ConfidenceTier;
  value: number;       // 0-1
  reason: string;      // human-readable explanation
}

export interface Message {
  role: Role;
  content: string;
  sources?: Source[];
  chart?: ChartPayload;
  confidence?: Confidence;
}
