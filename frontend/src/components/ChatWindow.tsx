import { useEffect, useRef, useState } from "react";
import { streamChat, getLLMConfig, type ChatMode, type IngestResult } from "../api/client";
import type { Message } from "../types";
import MessageBubble from "./MessageBubble";
import FileUpload from "./FileUpload";

interface Props {
  sessionId: string;
  onUploaded: () => void;
}

const MODE_OPTIONS: { value: ChatMode; label: string; tooltip: string }[] = [
  { value: "auto", label: "Auto", tooltip: "Let the chatbot decide which tool to use." },
  { value: "chat", label: "Chat", tooltip: "Plain answer from documents — no tools, no calculations." },
  { value: "aggregate", label: "Calculate", tooltip: "Force compute_metric: average, sum, max, top-N, trend." },
  { value: "query", label: "Filter", tooltip: "Force query_table: filter, sort, or list rows from a CSV/Excel file." },
];

export default function ChatWindow({ sessionId, onUploaded }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [mode, setMode] = useState<ChatMode>("auto");
  const endRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function send() {
    const text = input.trim();
    if (!text || busy) return;
    setInput("");
    setBusy(true);

    const userMsg: Message = { role: "user", content: text };
    const assistantMsg: Message = { role: "assistant", content: "" };
    setMessages((m) => [...m, userMsg, assistantMsg]);

    const updateLast = (patch: (m: Message) => Message) =>
      setMessages((curr) => {
        const out = curr.slice();
        out[out.length - 1] = patch(out[out.length - 1]);
        return out;
      });

    const controller = new AbortController();
    abortRef.current = controller;

    await streamChat(
      sessionId,
      text,
      {
        onSources: (sources) => updateLast((m) => ({ ...m, sources })),
        onToken: (tok) => updateLast((m) => ({ ...m, content: m.content + tok })),
        onChart: (chart) => updateLast((m) => ({ ...m, chart })),
        onConfidence: (confidence) => updateLast((m) => ({ ...m, confidence })),
        onDone: () => setBusy(false),
        onError: (err) => {
          updateLast((m) => ({ ...m, content: m.content + `\n[error: ${String(err)}]` }));
          setBusy(false);
        },
      },
      controller.signal,
      mode,
      getLLMConfig(),
    );
    abortRef.current = null;
  }

  function stop() {
    abortRef.current?.abort();
  }

  function handleIngested(result: IngestResult) {
    if (result.summary) {
      const header = `**Ingested ${result.filename}** (${result.chunks} chunks)`;
      const summaryMsg: Message = {
        role: "assistant",
        content: `${header}\n\n${result.summary}`,
      };
      // If the assistant is mid-stream, splice the summary in BEFORE the
      // in-flight bubble so streamChat's updateLast still targets the right one.
      setMessages((curr) => {
        if (
          busy &&
          curr.length > 0 &&
          curr[curr.length - 1].role === "assistant"
        ) {
          return [...curr.slice(0, -1), summaryMsg, curr[curr.length - 1]];
        }
        return [...curr, summaryMsg];
      });
    }
    onUploaded();
  }

  return (
    <div className="chat-area">
      <div className="chat">
        {messages.map((m, i) => (
          <MessageBubble key={i} msg={m} />
        ))}
        <div ref={endRef} />
      </div>
      <div className="mode-pills" role="radiogroup" aria-label="Tool mode">
        {MODE_OPTIONS.map((opt) => (
          <button
            key={opt.value}
            type="button"
            role="radio"
            aria-checked={mode === opt.value}
            className={`mode-pill ${mode === opt.value ? "active" : ""}`}
            title={opt.tooltip}
            onClick={() => setMode(opt.value)}
            disabled={busy}
          >
            {opt.label}
          </button>
        ))}
      </div>
      <div className="composer">
        <FileUpload onIngested={handleIngested} />
        <input
          type="text"
          value={input}
          placeholder="Ask about the uploaded fund documents…"
          disabled={busy}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
        />
        {busy ? (
          <button onClick={stop} className="stop-btn">Stop</button>
        ) : (
          <button onClick={send}>Send</button>
        )}
      </div>
    </div>
  );
}
