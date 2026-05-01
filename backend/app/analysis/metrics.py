from typing import Literal

import pandas as pd


Op = Literal["mean", "sum", "count", "min", "max", "top_n", "trend"]


def compute(
    df: pd.DataFrame,
    op: Op,
    column: str,
    group_by: str | None = None,
    n: int | None = None,
) -> pd.DataFrame:
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not in dataframe. Available: {list(df.columns)}")

    if op == "top_n":
        n = n or 10
        return df.nlargest(n, column).reset_index(drop=True)

    if op == "trend":
        if group_by is None:
            raise ValueError("'trend' requires a group_by (typically a date/period column).")
        if group_by not in df.columns:
            raise ValueError(f"group_by column '{group_by}' not found.")
        return (
            df.groupby(group_by)[column]
            .mean()
            .reset_index()
            .sort_values(group_by)
        )

    agg_map = {"mean": "mean", "sum": "sum", "count": "count", "min": "min", "max": "max"}
    fn = agg_map[op]
    if group_by:
        return df.groupby(group_by)[column].agg(fn).reset_index()
    return pd.DataFrame({"metric": [op], "value": [getattr(df[column], fn)()]})
