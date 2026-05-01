import re
from io import StringIO
from pathlib import Path

import pandas as pd


# Values that mean "missing" in fund/finance exports.
_NULL_TOKENS = {"", "-", "--", "n/a", "na", "none", "null", "nan", "#n/a"}
# Currency, thousands separators, stray whitespace — strip before numeric coercion.
_NUMERIC_STRIP = re.compile(r"[\$£€₹,\s]")
# Column-name hints that justify trying to parse strings as dates.
_DATE_HINTS = ("date", "month", "year", "period", "as of", "asof")


def parse_tabular(path: Path) -> list[dict]:
    """Parse a CSV or Excel file into one or more sheet records.

    CSV files always produce a single record. Multi-sheet Excel files produce
    one record per non-empty sheet, with logical_name = "{filename} :: {sheet}".
    Single-sheet Excel files use the bare filename like CSVs.

    Returns: [{"logical_name", "df", "summary"}, ...]
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = _read_csv_with_header_detection(path)
        return [_finalize(df, path.name)] if not df.empty else []
    if suffix in (".xlsx", ".xls"):
        return _read_excel_all_sheets(path)
    raise ValueError(f"Unsupported tabular format: {suffix}")


def _read_csv_with_header_detection(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not _has_unnamed_columns(df):
        return df
    # Title rows above the real header — try later rows.
    for header_row in range(1, 5):
        try:
            cand = pd.read_csv(path, header=header_row)
        except Exception:
            continue
        if not cand.empty and not _has_unnamed_columns(cand):
            return cand
    return df


def _read_excel_all_sheets(path: Path) -> list[dict]:
    xls = pd.ExcelFile(path)
    sheet_names = xls.sheet_names
    is_multi = len(sheet_names) > 1
    out: list[dict] = []
    for sheet in sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        if df.empty:
            continue
        if _has_unnamed_columns(df):
            for header_row in range(1, 5):
                try:
                    cand = pd.read_excel(xls, sheet_name=sheet, header=header_row)
                except Exception:
                    continue
                if not cand.empty and not _has_unnamed_columns(cand):
                    df = cand
                    break
        if df.empty:
            continue
        logical_name = f"{path.name} :: {sheet}" if is_multi else path.name
        out.append(_finalize(df, logical_name))
    return out


def _has_unnamed_columns(df: pd.DataFrame, threshold: float = 0.3) -> bool:
    """Pandas auto-names empty header cells `Unnamed: N`. A high ratio of those
    is a strong signal that the real header sits one or more rows down."""
    if df.empty:
        return False
    cols = [str(c) for c in df.columns]
    unnamed = sum(1 for c in cols if c.startswith("Unnamed:"))
    return unnamed / len(cols) > threshold


def _finalize(df: pd.DataFrame, logical_name: str) -> dict:
    cleaned, report = _clean(df)
    summary = _summarize(cleaned, logical_name, report)
    return {"logical_name": logical_name, "df": cleaned, "summary": summary}


def _clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Light cleaning so downstream RAG and pandas analysis see usable types.

    Order matters: trim names → normalize nulls → drop empty rows/cols →
    coerce numeric strings → coerce date-named columns. Coercion is opt-in
    (only commits if ≥80% of non-null values parse) so we don't destroy a
    legitimately textual column that happens to contain a few numbers.
    """
    before_rows, before_cols = df.shape
    df = df.copy()

    df.columns = [str(c).strip() for c in df.columns]

    def text_cols() -> list[str]:
        return list(df.select_dtypes(include=["object", "string"]).columns)

    for col in text_cols():
        s = df[col].astype("string").str.strip()
        df[col] = s.mask(s.str.lower().isin(_NULL_TOKENS))

    df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)

    coerced_numeric: list[str] = []
    for col in text_cols():
        s = df[col].astype("string")
        is_pct = s.str.endswith("%", na=False)
        cleaned = s.str.replace(_NUMERIC_STRIP, "", regex=True).str.rstrip("%")
        nums = pd.to_numeric(cleaned, errors="coerce")
        non_null = int(s.notna().sum())
        if non_null > 0 and nums.notna().sum() / non_null >= 0.8:
            df[col] = nums.where(~is_pct, nums / 100)
            coerced_numeric.append(col)

    coerced_date: list[str] = []
    for col in text_cols():
        if not any(tok in col.lower() for tok in _DATE_HINTS):
            continue
        # format="mixed" tells pandas we knowingly want per-element parsing
        # (fund data mixes ISO, slash-separated, and "Jan"-style dates). Without
        # it, pandas tries to infer one consistent format, fails, and warns.
        parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
        non_null = int(df[col].notna().sum())
        if non_null > 0 and parsed.notna().sum() / non_null >= 0.8:
            df[col] = parsed
            coerced_date.append(col)

    return df, {
        "rows_dropped": before_rows - df.shape[0],
        "cols_dropped": before_cols - df.shape[1],
        "numeric_coerced": coerced_numeric,
        "date_coerced": coerced_date,
    }


def row_chunks(logical_name: str, df: pd.DataFrame, max_rows: int = 500) -> list[str]:
    """One chunk per row, formatted as `column=value; ...` so retrieval can
    surface specific rows for point-lookup queries (e.g. "expense ratio of
    Fund X"). Capped at max_rows to keep the index from exploding on huge
    files; for personal-portfolio scope this is plenty."""
    out: list[str] = []
    n = min(len(df), max_rows)
    for i in range(n):
        row = df.iloc[i]
        parts: list[str] = []
        for col, val in row.items():
            if pd.isna(val):
                continue
            parts.append(f"{col}={val}")
        if parts:
            out.append(f"Row from {logical_name}: " + "; ".join(parts) + ".")
    return out


def enumerations(
    logical_name: str, df: pd.DataFrame, max_cardinality: int = 500
) -> list[str]:
    """For each text column with reasonable cardinality, emit a chunk listing
    all distinct values. Lets "list all X" / "what are all the Y" queries hit
    a complete listing instead of a sampled head/tail summary."""
    out: list[str] = []
    for col in df.select_dtypes(include=["object", "string"]).columns:
        unique = df[col].dropna().unique()
        n = len(unique)
        if n < 2 or n > max_cardinality:
            continue
        values = sorted(str(v) for v in unique)
        out.append(
            f"All distinct values in column '{col}' from {logical_name} "
            f"({n} total): {', '.join(values)}."
        )
    return out


def synopsis(logical_name: str, df: pd.DataFrame) -> str:
    """One-paragraph descriptive synopsis suitable as a high-recall first chunk.

    Designed for "what's in fund.csv?" / "what does this dataset contain?"
    questions — the embedded summary is a stats dump, this is narrative."""
    parts = [f"{logical_name} is a tabular dataset with {len(df)} rows and {len(df.columns)} columns."]

    text_cols = list(df.select_dtypes(include=["object", "string"]).columns)
    num_cols = list(df.select_dtypes(include="number").columns)
    date_cols = list(df.select_dtypes(include="datetime").columns)

    if text_cols:
        parts.append(f"Text columns: {', '.join(text_cols)}.")
    if num_cols:
        parts.append(f"Numeric columns: {', '.join(num_cols)}.")
    if date_cols:
        for col in date_cols:
            try:
                lo, hi = df[col].min(), df[col].max()
                parts.append(f"{col} ranges from {lo.date()} to {hi.date()}.")
            except Exception:
                continue

    # Top-3 values from the most-distinctive text column (if any) hint at content.
    if text_cols:
        col = text_cols[0]
        top = df[col].value_counts().head(3)
        if len(top) > 0:
            sample = ", ".join(f"{k}" for k in top.index)
            parts.append(f"Common values in {col}: {sample}.")

    return " ".join(parts)


def _summarize(df: pd.DataFrame, filename: str, report: dict) -> str:
    buf = StringIO()
    buf.write(f"File: {filename}\n")
    buf.write(f"Rows: {len(df)}, Columns: {len(df.columns)}\n")
    if report["rows_dropped"] or report["cols_dropped"]:
        buf.write(
            f"Cleaning: dropped {report['rows_dropped']} empty rows, "
            f"{report['cols_dropped']} empty columns.\n"
        )
    if report["numeric_coerced"]:
        buf.write(f"Coerced to numeric: {', '.join(report['numeric_coerced'])}\n")
    if report["date_coerced"]:
        buf.write(f"Coerced to date: {', '.join(report['date_coerced'])}\n")

    buf.write("\nColumns and dtypes:\n")
    for col, dt in df.dtypes.items():
        buf.write(f"  - {col}: {dt}\n")

    n = len(df)
    if n <= 10:
        buf.write("\nAll rows:\n")
        buf.write(df.to_markdown(index=False))
    else:
        buf.write("\nFirst 5 rows:\n")
        buf.write(df.head().to_markdown(index=False))
        buf.write("\n\nLast 5 rows:\n")
        buf.write(df.tail().to_markdown(index=False))
        if n > 30:
            mid_start = n // 2 - 2
            buf.write("\n\nMiddle 5 rows:\n")
            buf.write(df.iloc[mid_start : mid_start + 5].to_markdown(index=False))

    buf.write("\n\nNumeric summary:\n")
    try:
        numeric_desc = df.describe(include="number")
        if not numeric_desc.empty:
            buf.write(numeric_desc.to_markdown())
    except Exception:
        pass

    cat_cols = [c for c in df.select_dtypes(include=["object", "string"]).columns]
    if cat_cols:
        buf.write("\n\nTop values in text columns:\n")
        for col in cat_cols:
            top = df[col].value_counts().head(5)
            if len(top) > 0:
                pairs = ", ".join(f"{k}={v}" for k, v in top.items())
                buf.write(f"  - {col}: {pairs}\n")

    date_cols = list(df.select_dtypes(include="datetime").columns)
    if date_cols:
        buf.write("\nDate ranges:\n")
        for col in date_cols:
            try:
                lo, hi = df[col].min(), df[col].max()
                buf.write(f"  - {col}: {lo} to {hi}\n")
            except Exception:
                continue

    return buf.getvalue()
