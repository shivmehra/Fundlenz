import pandas as pd

from app.chunkers.common import make_meta
from app.id.canonical import normalize_name
from app.rag.schema import ChunkMeta


WINDOWS: dict[str, int] = {"30d": 30, "90d": 90, "365d": 365}


def detect_timeseries(df: pd.DataFrame, name_col: str) -> str | None:
    """Returns the date column name iff some entity has multiple distinct dates
    for any datetime column. Otherwise None — the file is a snapshot."""
    if name_col not in df.columns:
        return None
    date_cols = list(df.select_dtypes(include="datetime").columns)
    if not date_cols:
        return None
    canon_series = df[name_col].astype(str).map(normalize_name)
    for dc in date_cols:
        counts = pd.DataFrame({"_c": canon_series, "_d": df[dc]}).dropna().groupby("_c")["_d"].nunique()
        if (counts > 1).any():
            return dc
    return None


def _stats_for_series(series: pd.Series) -> dict | None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    return {
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
        "min": float(s.min()),
        "max": float(s.max()),
        "count": int(s.size),
    }


def _cagr(slab: pd.DataFrame, date_col: str, value_col: str) -> float | None:
    s = pd.to_numeric(slab[value_col], errors="coerce")
    df = pd.DataFrame({"_d": slab[date_col], "_v": s}).dropna().sort_values("_d")
    if len(df) < 2:
        return None
    start_v = float(df["_v"].iloc[0])
    end_v = float(df["_v"].iloc[-1])
    if start_v <= 0:
        return None
    span_days = (df["_d"].iloc[-1] - df["_d"].iloc[0]).days
    if span_days <= 0:
        return None
    yrs = span_days / 365.25
    return float((end_v / start_v) ** (1.0 / yrs) - 1.0)


def build_time_windows(
    df: pd.DataFrame,
    file: str,
    file_id: str,
    *,
    sheet: str | None,
    name_col: str,
    date_col: str,
    value_cols: list[str],
) -> list[tuple[ChunkMeta, str]]:
    """One chunk per (canonical_id, window) carrying stats for each value_col.
    CAGR is included for the 365d window only. Caller is responsible for
    skipping when detect_timeseries returns None."""
    out: list[tuple[ChunkMeta, str]] = []
    work = df.copy()
    work["_canon"] = work[name_col].astype(str).map(normalize_name)
    work = work[work["_canon"] != ""]
    if work.empty or date_col not in work.columns:
        return out

    latest = work[date_col].max()
    if pd.isna(latest):
        return out

    for canon, grp in work.groupby("_canon", sort=True):
        for win_label, days in WINDOWS.items():
            cutoff = latest - pd.Timedelta(days=days)
            slab = grp[grp[date_col] >= cutoff]
            if len(slab) < 2:
                continue
            stats: dict[str, dict] = {}
            for vc in value_cols:
                if vc not in slab.columns:
                    continue
                col_stats = _stats_for_series(slab[vc])
                if col_stats is None:
                    continue
                if win_label == "365d":
                    cagr = _cagr(slab, date_col, vc)
                    if cagr is not None:
                        col_stats["cagr"] = cagr
                stats[vc] = col_stats
            if not stats:
                continue
            body = (
                f"Time window {win_label} for {canon} in {file}: "
                + "; ".join(
                    f"{vc} mean={s['mean']:.4f} std={s['std']:.4f} "
                    f"min={s['min']:.4f} max={s['max']:.4f}"
                    + (f" cagr={s['cagr']:.4f}" if "cagr" in s else "")
                    for vc, s in stats.items()
                )
                + "."
            )
            meta = make_meta(
                "time_window",
                file=file,
                file_id=file_id,
                sheet=sheet,
                canonical_id=canon,
                text=body,
                window=win_label,  # type: ignore[arg-type]
                stats=stats,
            )
            out.append((meta, body))
    return out
