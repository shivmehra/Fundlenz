import { useEffect, useState } from "react";
import { uploadFile, type IngestResult } from "../api/client";

interface Props {
  onIngested: (result: IngestResult) => void;
}

export default function FileUpload({ onIngested }: Props) {
  const [status, setStatus] = useState<string>("");

  // Auto-dismiss success / failure messages after a few seconds.
  useEffect(() => {
    if (!status || status.startsWith("Uploading")) return;
    const t = setTimeout(() => setStatus(""), 4000);
    return () => clearTimeout(t);
  }, [status]);

  async function handle(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setStatus(`Uploading ${file.name}…`);
    try {
      const res = await uploadFile(file);
      const extracted = res.extracted_tables ?? [];
      const ok = extracted.filter((t) => t.chunks > 0).length;
      const failed = extracted.filter((t) => t.error || t.chunks === 0).length;
      let msg = `Indexed ${res.chunks} chunks`;
      if (extracted.length > 0) {
        msg += ` (+${ok} table${ok === 1 ? "" : "s"} from PDF`;
        if (failed > 0) msg += `, ${failed} failed`;
        msg += ")";
      }
      setStatus(msg);
      onIngested(res);
    } catch (err) {
      setStatus(`Failed: ${(err as Error).message}`);
    } finally {
      e.target.value = "";
    }
  }

  return (
    <div className="upload-wrap">
      <label className="upload" title="Upload document">
        +
        <input type="file" accept=".pdf,.docx,.csv,.xlsx,.xls" onChange={handle} />
      </label>
      {status && <p className="upload-status">{status}</p>}
    </div>
  );
}
