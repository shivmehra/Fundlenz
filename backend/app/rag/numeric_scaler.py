import json
from pathlib import Path

import numpy as np
import pandas as pd


class NumericScaler:
    """Per-column z-score. Fit at ingest, persist mean/std, transform at query
    time. NaN inputs become 0 post-scale (treated as "average"). Constant columns
    get std=1 to avoid divide-by-zero."""

    def __init__(self) -> None:
        self.columns: list[str] = []
        self.mean: dict[str, float] = {}
        self.std: dict[str, float] = {}

    def fit(self, df: pd.DataFrame, columns: list[str]) -> None:
        self.columns = list(columns)
        for c in self.columns:
            s = pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series(dtype="float64")
            if s.notna().any():
                mu = float(s.mean())
                sd = float(s.std(ddof=0))
            else:
                mu, sd = 0.0, 0.0
            self.mean[c] = mu
            self.std[c] = sd if sd > 1e-9 else 1.0

    def transform_row(self, row: pd.Series) -> np.ndarray:
        vec = np.zeros(len(self.columns), dtype="float32")
        for i, c in enumerate(self.columns):
            v = row.get(c, np.nan)
            try:
                fv = float(v)
            except (TypeError, ValueError):
                fv = float("nan")
            if np.isnan(fv):
                vec[i] = 0.0
            else:
                vec[i] = (fv - self.mean[c]) / self.std[c]
        return vec

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"columns": self.columns, "mean": self.mean, "std": self.std}),
            encoding="utf-8",
        )

    def load(self, path: Path) -> None:
        d = json.loads(path.read_text(encoding="utf-8"))
        self.columns = d["columns"]
        self.mean = {k: float(v) for k, v in d["mean"].items()}
        self.std = {k: float(v) for k, v in d["std"].items()}
