import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { ConfidenceTier, MatchKind, Message } from "../types";
import PlotlyChart from "./PlotlyChart";

const MATCH_LABELS: Record<MatchKind, string> = {
  exact: "exact match",
  field: "field match",
  semantic: "semantic",
};

const TIER_LABELS: Record<ConfidenceTier, string> = {
  deterministic: "Verified",
  grounded: "Grounded",
  semantic: "Generated",
};

export default function MessageBubble({ msg }: { msg: Message }) {
  const empty = !msg.content;
  return (
    <div className={`bubble ${msg.role}`}>
      {msg.confidence && (
        <div
          className={`answer-confidence answer-confidence-${msg.confidence.tier}`}
          title={msg.confidence.reason}
        >
          <span className="answer-confidence-label">
            {TIER_LABELS[msg.confidence.tier]}
          </span>
          <span className="answer-confidence-pct">
            {Math.round(Math.max(0, Math.min(1, msg.confidence.value)) * 100)}%
          </span>
          <span className="answer-confidence-reason">{msg.confidence.reason}</span>
        </div>
      )}
      <div className="bubble-content">
        {empty ? (
          msg.role === "assistant" ? "…" : ""
        ) : msg.role === "assistant" ? (
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
        ) : (
          msg.content
        )}
      </div>
      {msg.chart && (
        <div style={{ marginTop: 10 }}>
          <PlotlyChart payload={msg.chart} />
        </div>
      )}
      {msg.sources && msg.sources.length > 0 && (
        <details className="sources">
          <summary>Sources ({msg.sources.length})</summary>
          <ul>
            {msg.sources.map((s, i) => {
              const pct = Math.round(Math.max(0, Math.min(1, s.score)) * 100);
              const match = s.match ?? "semantic";
              return (
                <li key={i}>
                  <span className={`confidence confidence-${match}`} title={MATCH_LABELS[match]}>
                    {pct}%
                  </span>
                  <span className="source-file">{s.filename}</span>
                  {s.type && <span className="source-meta"> · {s.type}</span>}
                  {s.canonical_id && <span className="source-meta"> · {s.canonical_id}</span>}
                  {s.page != null && <span className="source-meta"> · p.{s.page}</span>}
                </li>
              );
            })}
          </ul>
        </details>
      )}
    </div>
  );
}
