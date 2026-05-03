"""Match a query phrase against DataFrame column names. Used by the
numeric_threshold bypass path to decide which column the user meant when they
said "AUM" or "NAV regular"."""
import re

import pandas as pd


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> set[str]:
    return set(_TOKEN_RE.findall(s.lower()))


def resolve_column(query: str, df: pd.DataFrame) -> tuple[str | None, float]:
    """Find the numeric column whose name best matches the query phrase.

    Score = fraction of column-name tokens that appear in the query, weighted
    so longer matches win (e.g. "NAV Regular" beats "NAV Direct" when the
    query says "NAV regular"). Returns (column_name, score). Score is 0 when
    no column shares any token with the query.

    Numeric columns only — threshold filters are meaningless on text columns.
    """
    q_toks = _tokens(query)
    if not q_toks:
        return None, 0.0

    numeric_cols = list(df.select_dtypes(include="number").columns)
    best_col: str | None = None
    best_score = 0.0
    for col in numeric_cols:
        c_toks = _tokens(str(col))
        if not c_toks:
            continue
        overlap = c_toks & q_toks
        if not overlap:
            continue
        # fraction-of-column-tokens-matched + tiny tiebreak for raw count so
        # a 2/2 match beats a 1/1 match (more specific column name wins).
        score = len(overlap) / len(c_toks) + 0.001 * len(overlap)
        if score > best_score:
            best_score = score
            best_col = str(col)

    return best_col, best_score
