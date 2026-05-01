import type { ChartPayload, Source } from "../types";

export interface StreamHandlers {
  onSources: (s: Source[]) => void;
  onToken: (t: string) => void;
  onChart: (c: ChartPayload) => void;
  onDone: () => void;
  onError: (err: unknown) => void;
}

export interface DocumentInfo {
  filename: string;
  chunks?: number;
}

export interface IngestResult {
  filename: string;
  chunks: number;
  summary: string;
}

export type ChatMode = "auto" | "chat" | "aggregate" | "query";

export async function uploadFile(file: File): Promise<IngestResult> {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch("/api/ingest", { method: "POST", body: fd });
  if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
  return res.json();
}

export async function getDocuments(): Promise<DocumentInfo[]> {
  const res = await fetch("/api/documents");
  if (!res.ok) throw new Error("Failed to fetch documents");
  return res.json();
}

export async function deleteDocument(filename: string): Promise<void> {
  const res = await fetch(`/api/documents/${encodeURIComponent(filename)}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`Delete failed: ${res.status}`);
}

export async function streamChat(
  sessionId: string,
  message: string,
  h: StreamHandlers,
  signal?: AbortSignal,
  mode: ChatMode = "auto",
): Promise<void> {
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({ session_id: sessionId, message, mode }),
      signal,
    });
    if (!res.body) throw new Error("No stream body");

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    // SSE frame separator can be \r\n\r\n (HTTP-style, what sse-starlette emits)
    // or \n\n (Unix-style). Match either.
    const frameSep = /\r?\n\r?\n/;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      while (true) {
        const m = frameSep.exec(buffer);
        if (!m) break;
        const raw = buffer.slice(0, m.index);
        buffer = buffer.slice(m.index + m[0].length);
        const ev = parseEvent(raw);
        if (!ev) continue;
        switch (ev.event) {
          case "sources": h.onSources(JSON.parse(ev.data)); break;
          case "token": h.onToken(ev.data); break;
          case "chart": h.onChart(JSON.parse(ev.data)); break;
          case "done": h.onDone(); return;
        }
      }
    }
    h.onDone();
  } catch (err) {
    // User-initiated abort: end cleanly, don't surface as an error.
    if ((err as Error).name === "AbortError") {
      h.onDone();
      return;
    }
    h.onError(err);
  }
}

function parseEvent(raw: string): { event: string; data: string } | null {
  let event = "message";
  const dataLines: string[] = [];
  for (const line of raw.split(/\r?\n/)) {
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      // SSE spec: strip exactly one leading space if present — preserves
      // leading-space token separators that some local LLMs emit (e.g. " The").
      let val = line.slice(5);
      if (val.startsWith(" ")) val = val.slice(1);
      dataLines.push(val);
    }
  }
  if (dataLines.length === 0) return null;
  return { event, data: dataLines.join("\n") };
}
