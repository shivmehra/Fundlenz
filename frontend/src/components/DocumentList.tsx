import { useState, useEffect, useCallback } from "react";
import { getDocuments, deleteDocument, type DocumentInfo } from "../api/client";

interface Props {
  refreshTrigger: number;
}

export default function DocumentList({ refreshTrigger }: Props) {
  const [docs, setDocs] = useState<DocumentInfo[]>([]);
  const [deleting, setDeleting] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setDocs(await getDocuments());
    } catch {}
  }, []);

  useEffect(() => { load(); }, [load, refreshTrigger]);

  async function handleDelete(filename: string) {
    setDeleting(filename);
    try {
      await deleteDocument(filename);
      await load();
    } catch (e) {
      console.error(e);
    } finally {
      setDeleting(null);
    }
  }

  if (docs.length === 0) {
    return <p className="doc-empty">No documents indexed.</p>;
  }

  return (
    <ul className="doc-list">
      {docs.map((d) => (
        <li key={d.filename} className="doc-item">
          <span className="doc-name" title={d.filename}>{d.filename}</span>
          <span className="doc-meta">{d.chunks ?? "?"} chunks</span>
          <button
            className="doc-delete"
            disabled={deleting === d.filename}
            onClick={() => handleDelete(d.filename)}
            title="Remove from index"
          >
            {deleting === d.filename ? "…" : "×"}
          </button>
        </li>
      ))}
    </ul>
  );
}
