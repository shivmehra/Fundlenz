from app.retrieval.router import Intent


# Hand-tuned chunk-type prior per intent. Missing entries default to 0.1 — a
# small non-zero floor so a chunk with strong text similarity can still surface
# even if the type isn't the obvious fit for the intent.
_TYPE_PRIOR: dict[Intent, dict[str, float]] = {
    "point_lookup":      {"row": 1.0, "entity": 0.9, "table": 0.4, "text": 0.2, "tabular_summary": 0.3},
    "list_distinct":     {"enumeration": 1.0, "synopsis": 0.4},
    "aggregate":         {"tabular_summary": 0.6, "entity": 0.4, "row": 0.3, "synopsis": 0.3},
    "filter_rows":       {"row": 1.0, "entity": 0.6, "tabular_summary": 0.3},
    "numeric_threshold": {"numeric_vector": 0.8, "row": 0.7, "entity": 0.5},
    "trend":             {"time_window": 1.0, "tabular_summary": 0.3},
    "qualitative":       {"text": 1.0, "table": 0.6, "synopsis": 0.4, "entity": 0.4},
}


def type_prior(chunk_type: str, intent: Intent) -> float:
    return _TYPE_PRIOR.get(intent, {}).get(chunk_type, 0.1)


def deterministic_score(
    *,
    exact_id: bool,
    exact_cell: bool,
    text_sim: float,
    numeric_dist: float | None,
    chunk_type: str,
    intent: Intent,
) -> float:
    """Linear combination of signals. Coefficients are intentionally not
    learned — this is a deterministic rerank that the user can reason about
    and the test suite can pin. A cross-encoder can replace the text_sim
    coefficient slot in a future iteration."""
    score = 0.0
    if exact_id:
        score += 1.00
    if exact_cell:
        score += 0.40
    score += 0.50 * float(text_sim)
    if numeric_dist is not None:
        # closer is better; clamp so a wildly-distant outlier can't dominate.
        score += 0.30 * (-min(numeric_dist, 9.9) / 10.0)
    score += 0.20 * type_prior(chunk_type, intent)
    return score
