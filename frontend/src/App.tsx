import { useEffect, useMemo, useState } from "react";
import ChatWindow from "./components/ChatWindow";
import DocumentList from "./components/DocumentList";
import { getSettings, updateSettings } from "./api/client";

export default function App() {
  const sessionId = useMemo(() => crypto.randomUUID(), []);
  const [refreshTick, setRefreshTick] = useState(0);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [numericAnn, setNumericAnn] = useState(false);
  const [crossEncoder, setCrossEncoder] = useState(true);
  const [openTooltip, setOpenTooltip] = useState<string | null>(null);

  useEffect(() => {
    getSettings()
      .then((s) => {
        setNumericAnn(s.enable_numeric_ann);
        setCrossEncoder(s.enable_cross_encoder);
      })
      .catch(() => {});
  }, []);

  // Close drawer on Escape.
  useEffect(() => {
    if (!drawerOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setDrawerOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [drawerOpen]);

  // Pinned tooltip: any click outside the open anchor/content closes it.
  // Anchor + tooltip click handlers stop propagation, so this only fires
  // for genuinely-outside clicks. Escape also dismisses for keyboard users.
  useEffect(() => {
    if (!openTooltip) return;
    const onDocClick = () => setOpenTooltip(null);
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpenTooltip(null);
    };
    document.addEventListener("click", onDocClick);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("click", onDocClick);
      document.removeEventListener("keydown", onKey);
    };
  }, [openTooltip]);

  function toggleTooltip(key: string, e: React.MouseEvent) {
    e.stopPropagation();
    setOpenTooltip((cur) => (cur === key ? null : key));
  }

  async function handleNumericAnnToggle(e: React.ChangeEvent<HTMLInputElement>) {
    const next = e.target.checked;
    try {
      const updated = await updateSettings({ enable_numeric_ann: next });
      setNumericAnn(updated.enable_numeric_ann);
    } catch {}
  }

  async function handleCrossEncoderToggle(e: React.ChangeEvent<HTMLInputElement>) {
    const next = e.target.checked;
    try {
      const updated = await updateSettings({ enable_cross_encoder: next });
      setCrossEncoder(updated.enable_cross_encoder);
    } catch {}
  }

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

        <div className="settings-section">
          <p className="settings-title">Settings</p>
          <div className="toggle-row">
            <span className="toggle-label-text">Numeric ANN</span>
            <span
              className={`tooltip-anchor ${openTooltip === "ann" ? "open" : ""}`}
              role="button"
              tabIndex={0}
              aria-label="Help: Numeric ANN"
              onClick={(e) => toggleTooltip("ann", e)}
            >
              ℹ
            </span>
            <label className="toggle-switch">
              <input
                type="checkbox"
                checked={numericAnn}
                onChange={handleNumericAnnToggle}
              />
              <span className="toggle-slider" />
            </label>
            {openTooltip === "ann" && (
              <div className="tooltip-text" onClick={(e) => e.stopPropagation()}>
                <button
                  type="button"
                  className="tooltip-close"
                  aria-label="Close"
                  onClick={() => setOpenTooltip(null)}
                >
                  ×
                </button>
                Enable when you have 50,000+ rows and pandas filtering becomes
                slow. At typical scale pandas is faster and exact — numeric ANN
                approximates and adds overhead. Re-upload files after enabling
                to build the index. Resets on backend restart.
              </div>
            )}
          </div>
          <div className="toggle-row">
            <span className="toggle-label-text">Cross-encoder</span>
            <span
              className={`tooltip-anchor ${openTooltip === "ce" ? "open" : ""}`}
              role="button"
              tabIndex={0}
              aria-label="Help: Cross-encoder"
              onClick={(e) => toggleTooltip("ce", e)}
            >
              ℹ
            </span>
            <label className="toggle-switch">
              <input
                type="checkbox"
                checked={crossEncoder}
                onChange={handleCrossEncoderToggle}
              />
              <span className="toggle-slider" />
            </label>
            {openTooltip === "ce" && (
              <div className="tooltip-text" onClick={(e) => e.stopPropagation()}>
                <button
                  type="button"
                  className="tooltip-close"
                  aria-label="Close"
                  onClick={() => setOpenTooltip(null)}
                >
                  ×
                </button>
                Second-stage rerank that scores (query, chunk) jointly for
                better quality. Adds ~80ms per query and loads a second model
                into RAM. Turn off if responses feel sluggish or memory is
                tight; quality drops slightly on qualitative questions.
                Resets on backend restart.
              </div>
            )}
          </div>
        </div>
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
