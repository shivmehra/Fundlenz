export type Role = "user" | "assistant";

export interface Source {
  filename: string;
  page: number | null;
  score: number;
}

export interface ChartPayload {
  text: string;
  chart_spec: { data: unknown[]; layout: Record<string, unknown> };
}

export interface Message {
  role: Role;
  content: string;
  sources?: Source[];
  chart?: ChartPayload;
}
