import json
import re
import unicodedata
from pathlib import Path


# Words dropped to canonicalize fund names. "Regular/Direct/Growth/IDCW" are
# intentionally NOT here — they distinguish real distinct schemes.
_NOISE_TOKENS = {"fund", "scheme", "plan", "the", "an", "a"}
_NONALPHA = re.compile(r"[^a-z0-9 ]")
_WS = re.compile(r"\s+")


def normalize_name(name: str) -> str:
    """Deterministic, idempotent canonicalization for fund/entity names."""
    if not name:
        return ""
    s = unicodedata.normalize("NFKC", str(name)).lower()
    s = _NONALPHA.sub(" ", s)
    toks = [t for t in s.split() if t and t not in _NOISE_TOKENS]
    return _WS.sub(" ", " ".join(toks)).strip()


class AliasMap:
    """Bidirectional alias <-> canonical_id, persisted as plain JSON so the user
    can hand-edit synonyms (e.g. "HDFC Top 100" -> "hdfc top 100 direct growth")."""

    def __init__(self, path: Path):
        self.path = path
        self.alias_to_canon: dict[str, str] = {}
        self.canon_to_aliases: dict[str, list[str]] = {}

    def load(self) -> None:
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.alias_to_canon = data.get("alias_to_canon", {})
            self.canon_to_aliases = data.get("canon_to_aliases", {})
        else:
            self.alias_to_canon = {}
            self.canon_to_aliases = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "alias_to_canon": self.alias_to_canon,
                    "canon_to_aliases": self.canon_to_aliases,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def register(self, raw_name: str) -> str:
        """Register raw_name as an alias of its normalized canonical form.
        Returns the canonical_id."""
        canon = normalize_name(raw_name)
        if not canon:
            return ""
        self.alias_to_canon.setdefault(canon, canon)
        norm_alias = normalize_name(raw_name)
        self.alias_to_canon.setdefault(norm_alias, canon)
        self.canon_to_aliases.setdefault(canon, [])
        if raw_name and raw_name not in self.canon_to_aliases[canon]:
            self.canon_to_aliases[canon].append(raw_name)
        return canon

    def resolve(self, query_token: str) -> str | None:
        """Resolve a free-form query token to a canonical_id, or None."""
        return self.alias_to_canon.get(normalize_name(query_token))

    def aliases_for(self, canonical_id: str) -> list[str]:
        return list(self.canon_to_aliases.get(canonical_id, []))
