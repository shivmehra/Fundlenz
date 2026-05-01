import pandas as pd


def chart_spec(result: pd.DataFrame, op: str, column: str, group_by: str | None) -> dict:
    """Return a Plotly figure spec (JSON-serializable) for the given result."""
    if op == "trend" and group_by:
        x = result[group_by].astype(str).tolist()
        y = result[column].tolist()
        return {
            "data": [{"type": "scatter", "mode": "lines+markers", "x": x, "y": y, "name": column}],
            "layout": {"title": f"Trend of {column} by {group_by}"},
        }

    if op == "top_n":
        # Pick a meaningful label column: explicit group_by, else the first non-numeric
        # column in the result, else fall back to the row index.
        if group_by and group_by in result.columns:
            label_col = group_by
        else:
            non_numeric = result.select_dtypes(exclude="number").columns
            label_col = non_numeric[0] if len(non_numeric) > 0 else None
        x = result[label_col].astype(str).tolist() if label_col else result.index.astype(str).tolist()
        y = result[column].tolist()
        return {
            "data": [{"type": "bar", "x": x, "y": y, "name": column}],
            "layout": {"title": f"Top {len(result)} by {column}"},
        }

    if group_by and group_by in result.columns:
        x = result[group_by].astype(str).tolist()
        y = result[column].tolist()
        return {
            "data": [{"type": "bar", "x": x, "y": y, "name": column}],
            "layout": {"title": f"{op.capitalize()} of {column} by {group_by}"},
        }

    value = float(result["value"].iloc[0])
    return {
        "data": [{"type": "indicator", "mode": "number", "value": value, "title": {"text": f"{op}({column})"}}],
        "layout": {"title": f"{op.capitalize()} of {column}"},
    }
