import re

from app.id.canonical import AliasMap, normalize_name
from app.index.composite import CompositeIndex
from app.retrieval.router import Plan


# "X = Y" or "X is Y" / "X: Y" cell predicates pulled out of free-form queries.
# Conservative — only triggers when the predicate is unambiguous.
_CELL_PREDICATE_RE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_ \-]{1,40}?)\s*(?:=|==|\bis\b|:)\s*\"?([A-Za-z0-9 \-_/]{1,60})\"?",
    re.IGNORECASE,
)


def resolve_canonical_id(plan: Plan, idx: CompositeIndex) -> str | None:
    """Try to resolve raw_entity_phrases against the alias map, longest-first
    so multi-word fund names beat partial matches. For each candidate phrase
    we additionally try its progressively shorter suffixes/substrings — e.g.
    "HDFC Balanced Advantage Fund" → also tries "HDFC Balanced Advantage" and
    so on — to handle cases where the alias map only knows the canonical form
    that lacks the "Fund" suffix. Returns the first hit, or None."""
    phrases = sorted(plan.raw_entity_phrases or [], key=len, reverse=True)
    for phrase in phrases:
        canon = idx.aliases.resolve(phrase)
        if canon:
            return canon
        # Direct canonical_id check (the inverted index uses these as keys).
        norm = normalize_name(phrase)
        if norm and idx.inverted.lookup_id(norm):
            return norm
        # Progressively trim trailing tokens. Helps when the user includes
        # noise like "Fund" or "Plan" in a way the noise-token strip already
        # handled at ingest, but the user's phrasing also tacks on "Direct
        # Growth" / "Regular" qualifiers we want to ignore for resolution.
        tokens = norm.split()
        for cut in range(1, len(tokens)):
            candidate = " ".join(tokens[: len(tokens) - cut])
            if not candidate:
                break
            if idx.inverted.lookup_id(candidate):
                return candidate
            via_alias = idx.aliases.alias_to_canon.get(candidate)
            if via_alias:
                return via_alias
    return None


def extract_cell_predicates(query: str) -> list[tuple[str, str]]:
    """Return a list of (column, value) pairs implied by the query. Best-effort
    — the inverted index lookup is itself exact so a false predicate here
    just means a missed lookup, never a wrong row."""
    out: list[tuple[str, str]] = []
    for m in _CELL_PREDICATE_RE.finditer(query):
        col = m.group(1).strip()
        val = m.group(2).strip()
        if col and val:
            out.append((col, val))
    return out
