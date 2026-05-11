import { useEffect, useState, useCallback } from "react";
import { getStats, type StatsResponse } from "../api/client";

interface Props {
  refreshTrigger: number;
}

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

function formatRows(n: number): string {
  return n.toLocaleString();
}

export default function StatsCard({ refreshTrigger }: Props) {
  const [stats, setStats] = useState<StatsResponse | null>(null);

  const load = useCallback(async () => {
    try {
      setStats(await getStats());
    } catch {
      // Keep the last successful value; same pattern as DocumentList.load().
    }
  }, []);

  useEffect(() => {
    load();
  }, [load, refreshTrigger]);

  if (!stats) {
    return null;
  }

  const pct = stats.ram.system_percent_used;
  // Warn-only policy: never block ingestion, only colour the panel.
  // Amber when system RAM is under heavy pressure, red when it's nearly gone.
  const ramTone =
    pct >= 90 ? "stats-critical" : pct >= 75 ? "stats-warn" : "";

  return (
    <div className="stats-card">
      <p className="settings-title">Capacity</p>
      <div className="stats-row">
        <span className="stats-label">Total rows</span>
        <span className="stats-value">{formatRows(stats.rows.total_tabular_rows)}</span>
      </div>
      <div className={`stats-row ${ramTone}`}>
        <span className="stats-label">RAM free</span>
        <span className="stats-value">
          {formatBytes(stats.ram.system_available_bytes)} /{" "}
          {formatBytes(stats.ram.system_total_bytes)}
        </span>
      </div>
      <div className="stats-row stats-row-muted">
        <span className="stats-label">Fundlenz</span>
        <span className="stats-value">{formatBytes(stats.ram.process_rss_bytes)}</span>
      </div>
    </div>
  );
}
