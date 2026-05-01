import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Message } from "../types";
import PlotlyChart from "./PlotlyChart";

export default function MessageBubble({ msg }: { msg: Message }) {
  const empty = !msg.content;
  return (
    <div className={`bubble ${msg.role}`}>
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
            {msg.sources.map((s, i) => (
              <li key={i}>
                {s.filename}
                {s.page != null ? ` — p.${s.page}` : ""} (score {s.score.toFixed(3)})
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}
