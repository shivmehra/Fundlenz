import re
from dataclasses import dataclass, field
from typing import Literal


Intent = Literal[
    "point_lookup",
    "list_distinct",
    "aggregate",
    "filter_rows",
    "numeric_threshold",
    "trend",
    "qualitative",
]


@dataclass
class Plan:
    intent: Intent
    canonical_id: str | None = None  # resolved entity, if any
    raw_entity_phrases: list[str] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    threshold: tuple[str, float] | None = None  # ("op", value)
    window: str | None = None


# Already in main.py; duplicated here so the router stays standalone.
_QUANT_KEYWORDS = {
    "average", "avg", "mean", "median", "sum", "total", "top", "bottom",
    "trend", "plot", "chart", "graph", "calculate", "compute",
    "highest", "lowest", "max", "maximum", "min", "minimum",
    "compare", "rank", "count", "aggregate", "by year", "by month",
    "analyz", "analys", "breakdown", "distribution", "histogram",
}

_LIST_DISTINCT_PATTERNS = (
    "list all", "show all", "all distinct", "all the categories",
    "all benchmarks", "all schemes", "all categories", "what categories",
    "what are all", "list distinct", "distinct values",
)

_FILTER_ROWS_PATTERNS = (
    "show me", "show the", "find all", "find the", "find me",
    "which fund", "which row", "rows with", "rows where",
    "funds with", "stocks with", "entries with", "items with",
    "give me all", "give me the", "list the funds", "list funds",
)

_TREND_PATTERNS = ("trend", "over time", "by year", "by month", "by quarter")

_THRESHOLD_RE = re.compile(
    r"\b(greater than|more than|above|less than|fewer than|below|over|under|>=|<=|>|<)\s*"
    r"(\d[\d,\.]*)\b",
    re.IGNORECASE,
)


def _detect_threshold(q: str) -> tuple[str, float] | None:
    m = _THRESHOLD_RE.search(q)
    if not m:
        return None
    word, num = m.group(1).lower(), m.group(2).replace(",", "")
    op_map = {
        "greater than": ">", "more than": ">", "above": ">", "over": ">", ">": ">", ">=": ">=",
        "less than": "<", "fewer than": "<", "below": "<", "under": "<", "<": "<", "<=": "<=",
    }
    try:
        return op_map[word], float(num)
    except (KeyError, ValueError):
        return None


# Heuristic regex for "the X of <Entity>" / "about <Entity>" / "regarding
# <Entity>" style references. Captures the entity phrase (a fund name) so the
# orchestrator can try alias resolution.
_POINT_LOOKUP_RE = re.compile(
    r"\b(?:of|for|about|regarding)\s+([A-Z][A-Za-z0-9 &\-]{2,80}?)"
    r"(?=\s*(?:\?|$|,|\.|;|in\b|on\b|doesn|does|has\b|have\b|with\b))",
)

# Capitalized 2-7 word phrases (proper noun runs). Used as a permissive fallback
# so we still capture fund names in queries that lack "of"/"for"/"about" — e.g.
# "HDFC Balanced Advantage Fund has the highest AUM". Embedded numeric tokens
# ("HDFC Top 100 Fund") are allowed mid-phrase. The orchestrator vets these
# against the alias map before treating them as resolved entities.
_CAPITALIZED_PHRASE_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9&\-]*(?:\s+(?:[A-Z][A-Za-z0-9&\-]*|[0-9][A-Za-z0-9&\-]*)){1,6})\b"
)
# Words that look proper-noun-shaped at the start of a sentence but aren't
# entity tokens. Suppress captures that consist only of these.
_SENTENCE_STARTERS = {
    "what", "which", "who", "where", "when", "why", "how",
    "list", "show", "find", "give", "tell", "is", "are", "does", "do",
    "can", "could", "would", "should", "the", "a", "an",
}


def classify_intent(query: str) -> Plan:
    """Rule-based intent classifier. LLM fallback is intentionally NOT added in
    this pass (per plan: rules first, latency budget). The classifier is
    deterministic — same input → same Plan."""
    q = query.strip()
    ql = q.lower()
    plan = Plan(intent="qualitative")

    # entity capture (always attempt; orchestrator decides whether to use).
    # Two layers: structured "of/for/about <X>" matches first, then a
    # permissive capitalized-phrase fallback that catches "HDFC Balanced
    # Advantage Fund has..." style references the structured regex misses.
    seen: set[str] = set()
    for m in _POINT_LOOKUP_RE.finditer(q):
        phrase = m.group(1).strip()
        if phrase and phrase not in seen:
            plan.raw_entity_phrases.append(phrase)
            seen.add(phrase)
    for m in _CAPITALIZED_PHRASE_RE.finditer(q):
        phrase = m.group(1).strip()
        if not phrase or phrase in seen:
            continue
        first_word = phrase.split()[0].lower()
        if first_word in _SENTENCE_STARTERS and m.start() == 0:
            # "What about HDFC ..." — drop the leading "What" but keep "HDFC ...".
            tail = " ".join(phrase.split()[1:])
            if tail and tail not in seen:
                plan.raw_entity_phrases.append(tail)
                seen.add(tail)
            continue
        plan.raw_entity_phrases.append(phrase)
        seen.add(phrase)

    # threshold
    plan.threshold = _detect_threshold(q)

    # explicit pattern matches, in priority order
    if any(pat in ql for pat in _LIST_DISTINCT_PATTERNS):
        plan.intent = "list_distinct"
        return plan

    if any(pat in ql for pat in _TREND_PATTERNS):
        plan.intent = "trend"
        # Prefer a 1y window if "year" appears; default to None.
        if "year" in ql:
            plan.window = "365d"
        elif "month" in ql:
            plan.window = "90d"
        return plan

    if plan.threshold is not None:
        plan.intent = "numeric_threshold"
        return plan

    # Point lookup is checked BEFORE the quant-keyword fallback because entity
    # names can contain quant keywords as substrings (e.g. "HDFC Top 100" — the
    # "top" substring would otherwise misroute to "aggregate").
    if plan.raw_entity_phrases and (
        ql.startswith("what")
        or ql.startswith("how much")
        or ql.startswith("tell me")
        or " of " in ql
    ):
        plan.intent = "point_lookup"
        return plan

    if any(pat in ql for pat in _QUANT_KEYWORDS):
        plan.intent = "aggregate"
        return plan

    if any(pat in ql for pat in _FILTER_ROWS_PATTERNS):
        plan.intent = "filter_rows"
        return plan

    return plan
