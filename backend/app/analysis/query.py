"""Filter / sort / select / limit on a DataFrame. Pure logic, no I/O.

Backs the `query_table` LLM tool, which complements `compute_metric`:
- compute_metric: aggregate to one number or chart
- query_table:   return a subset of rows
"""
from typing import Any

import pandas as pd


def query_table(
    df: pd.DataFrame,
    filters: list[dict] | None = None,
    sort_by: str | None = None,
    sort_desc: bool = True,
    select_columns: list[str] | None = None,
    limit: int = 50,
) -> pd.DataFrame:
    """Apply optional filters, sort, column selection, and limit. Each filter
    is {column, op, value} where op ∈ {==,!=,>,>=,<,<=,contains,in}."""
    if filters:
        for f in filters:
            df = _apply_filter(df, f)
    if sort_by:
        if sort_by not in df.columns:
            raise ValueError(
                f"Sort column '{sort_by}' not found. Available: {list(df.columns)}"
            )
        df = df.sort_values(sort_by, ascending=not sort_desc)
    if select_columns:
        valid = [c for c in select_columns if c in df.columns]
        if valid:
            df = df[valid]
    if limit is None or limit < 0:
        limit = 50
    return df.head(limit)


def _apply_filter(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    col = f["column"]
    op = f["op"]
    val = f["value"]
    if col not in df.columns:
        raise ValueError(
            f"Filter column '{col}' not found. Available: {list(df.columns)}"
        )
    s = df[col]
    if op == "==":
        return df[s == val]
    if op == "!=":
        return df[s != val]
    if op == ">":
        return df[s > _num(val)]
    if op == ">=":
        return df[s >= _num(val)]
    if op == "<":
        return df[s < _num(val)]
    if op == "<=":
        return df[s <= _num(val)]
    if op == "contains":
        return df[s.astype(str).str.contains(str(val), case=False, na=False)]
    if op == "in":
        vals = val if isinstance(val, list) else [val]
        return df[s.isin(vals)]
    raise ValueError(f"Unknown filter op: {op!r}")


def _num(val: Any) -> Any:
    """Coerce JSON-stringy numbers ("100", "1.5") to floats so comparisons work
    against numeric columns. Leaves non-numeric values unchanged."""
    if isinstance(val, (int, float)):
        return val
    try:
        return float(val)
    except (TypeError, ValueError):
        return val
