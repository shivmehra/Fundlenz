"""Universal text normalization. Run on raw text from every parser before chunking.

PDFs and Word documents commonly contain Unicode look-alikes (smart quotes,
non-breaking spaces, em dashes), hyphenated line wraps, and runaway whitespace
that bloat embeddings without adding meaning. This collapses all of that to a
canonical form so identical content embeds identically regardless of source.
"""
import re
import unicodedata


# Unicode characters that are visually equivalent to ASCII but break embeddings
# and confuse the LLM. NFKC normalization handles many of these, but a few
# (smart quotes, en/em dash, ellipsis, BOM) need explicit replacement because
# NFKC leaves them alone.
_REPLACEMENTS = {
    " ": " ",    # non-breaking space
    " ": " ",    # thin space
    "​": "",     # zero-width space
    "‘": "'",    # left single quote
    "’": "'",    # right single quote
    "“": '"',    # left double quote
    "”": '"',    # right double quote
    "–": "-",    # en dash
    "—": "-",    # em dash
    "…": "...",  # ellipsis
    "﻿": "",     # BOM
}

_REPLACE_RE = re.compile("|".join(re.escape(k) for k in _REPLACEMENTS))
# "expense-\nratio" → "expenseratio". Only joins when both sides are word chars
# so we don't merge across "X-\nY" where "-" is meaningful (e.g., "co-\nfounder").
# The trade-off: we lose those few legitimate hyphens. For fund docs this is fine.
_DEHYPHEN_RE = re.compile(r"(\w)-\n(\w)")
_TRAILING_SPACE_RE = re.compile(r"[ \t]+(?=\n)")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def normalize(text: str) -> str:
    """NFKC normalize, replace Unicode look-alikes with ASCII, dehyphenate
    line breaks, and collapse runaway whitespace. Preserves single newlines
    and paragraph breaks (one blank line)."""
    if not text:
        return text
    text = unicodedata.normalize("NFKC", text)
    text = _REPLACE_RE.sub(lambda m: _REPLACEMENTS[m.group(0)], text)
    text = _DEHYPHEN_RE.sub(r"\1\2", text)
    text = _TRAILING_SPACE_RE.sub("", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text
