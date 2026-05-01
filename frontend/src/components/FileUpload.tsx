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
      setStatus(`Indexed ${res.chunks} chunks`);
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
