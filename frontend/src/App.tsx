import { useEffect, useMemo, useState } from "react";
import ChatWindow from "./components/ChatWindow";
import DocumentList from "./components/DocumentList";

export default function App() {
  const sessionId = useMemo(() => crypto.randomUUID(), []);
  const [refreshTick, setRefreshTick] = useState(0);
  const [drawerOpen, setDrawerOpen] = useState(false);

  // Close drawer on Escape.
  useEffect(() => {
    if (!drawerOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setDrawerOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [drawerOpen]);

  return (
    <div className="app">
      <div className="header">
        <button
          className="hamburger"
          aria-label="Toggle documents drawer"
          onClick={() => setDrawerOpen((o) => !o)}
        >
          ☰
        </button>
        <h1>Fundlenz</h1>
        <span className="meta">session {sessionId.slice(0, 6)}</span>
      </div>

      <aside className={`sidebar ${drawerOpen ? "open" : ""}`}>
        <p className="sidebar-title">Indexed documents</p>
        <DocumentList refreshTrigger={refreshTick} />
      </aside>

      {drawerOpen && (
        <div
          className="sidebar-backdrop"
          onClick={() => setDrawerOpen(false)}
        />
      )}

      <ChatWindow
        sessionId={sessionId}
        onUploaded={() => setRefreshTick((t) => t + 1)}
      />

      <div className="footer-meta">local RAG · FAISS · Ollama</div>
    </div>
  );
}
