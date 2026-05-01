import re
from pathlib import Path

import pdfplumber

from app.text import normalize


# Lines like "Page 3 of 12", "3 / 12", "Page 5". Matched per-line.
_PAGE_NUMBER_RE = re.compile(
    r"^\s*(?:page\s*)?\d+(?:\s*(?:of|/)\s*\d+)?\s*$",
    re.IGNORECASE,
)

# Two strategies for finding tables. The "lines" strategy looks for ruled
# borders — reliable when present, silent when not. The "text" strategy aligns
# columns by whitespace alone — catches borderless tables (common in fund
# factsheets) but produces more false positives, which is why we validate.
_LINES_SETTINGS = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance": 3,
    "intersection_tolerance": 5,
}
_TEXT_SETTINGS = {
    "vertical_strategy": "text",
    "horizontal_strategy": "text",
    "min_words_vertical": 3,
    "min_words_horizontal": 1,
    "snap_tolerance": 3,
}


def parse_pdf(path: Path) -> list[dict]:
    """Return per-page records: {page, text, tables_md}.

    Pipeline:
    1. Try ruled-line table detection first; fall back to text-aligned detection.
       This is the robust knob — bordered tables get caught by `lines`,
       borderless ones by `text`.
    2. Validate each detected table — drop ones with <2 rows, <2 columns, or
       >70% empty cells. Stops multi-column page bodies and stray figures from
       being promoted to "tables".
    3. Clean cells: strip whitespace, fold multi-line cells to single line,
       drop fully-empty rows and columns, pad ragged rows.
    4. Strip page numbers and recurring boilerplate from body text.
    5. Apply Unicode/whitespace normalization to text.
    """
    with pdfplumber.open(path) as pdf:
        per_page_text: list[str] = []
        per_page_tables: list[list[str]] = []
        for page in pdf.pages:
            tables = _extract_validated_tables(page)
            per_page_tables.append([_table_to_markdown(t) for t in tables])
            per_page_text.append(page.extract_text() or "")

        boilerplate = _detect_boilerplate(per_page_text)
        records: list[dict] = []
        for i, (raw_text, tables_md) in enumerate(
            zip(per_page_text, per_page_tables), start=1
        ):
            text = _strip_boilerplate(raw_text, boilerplate)
            text = normalize(text)
            records.append({"page": i, "text": text, "tables_md": tables_md})
    return records


def _extract_validated_tables(page) -> list[list[list[str]]]:
    """Try lines strategy first; if nothing usable, try text strategy.
    Return only tables that survive cleaning and validation."""
    raw_tables = _find_tables_safe(page, _LINES_SETTINGS)
    valid = _clean_and_validate(raw_tables)
    if valid:
        return valid
    raw_tables = _find_tables_safe(page, _TEXT_SETTINGS)
    return _clean_and_validate(raw_tables)


def _clean_and_validate(raw_tables: list[list[list[str | None]]]) -> list[list[list[str]]]:
    out: list[list[list[str]]] = []
    for raw in raw_tables:
        cleaned = _clean_table_rows(raw)
        if _is_valid_table(cleaned):
            out.append(cleaned)
    return out


def _find_tables_safe(page, settings) -> list[list[list[str | None]]]:
    """Wrap pdfplumber's table API. Exotic PDFs sometimes raise on the way in
    (find_tables) or on the way out (extract); we'd rather silently skip the
    page's tables than fail the whole ingestion."""
    try:
        tables = page.find_tables(table_settings=settings)
    except Exception:
        return []
    out: list[list[list[str | None]]] = []
    for table in tables:
        try:
            out.append(table.extract())
        except Exception:
            continue
    return out


def _clean_table_rows(rows: list[list[str | None]]) -> list[list[str]]:
    """Normalize whitespace, fold multi-line cells to single line, drop rows
    and columns that are entirely empty, pad ragged rows."""
    if not rows:
        return []
    cleaned: list[list[str]] = []
    for row in rows:
        c_row = [_clean_cell(c) for c in row]
        if any(c for c in c_row):
            cleaned.append(c_row)
    if not cleaned:
        return []
    n_cols = max(len(r) for r in cleaned)
    cleaned = [r + [""] * (n_cols - len(r)) for r in cleaned]
    keep_cols = [i for i in range(n_cols) if any(r[i] for r in cleaned)]
    return [[r[i] for i in keep_cols] for r in cleaned]


def _clean_cell(c: str | None) -> str:
    if c is None:
        return ""
    # Multi-line cells (wrapped fund names, multi-line headers) become one line.
    s = c.replace("\r", " ").replace("\n", " ")
    # Collapse repeated whitespace from the join above.
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_valid_table(rows: list[list[str]]) -> bool:
    """Reject obvious false positives. A real table has at least 2 rows AND
    2 columns, and a meaningful fraction of non-empty cells."""
    if len(rows) < 2:
        return False
    n_cols = max(len(r) for r in rows)
    if n_cols < 2:
        return False
    total = sum(len(r) for r in rows)
    if total == 0:
        return False
    empty = sum(1 for r in rows for c in r if not c)
    if empty / total > 0.7:
        return False
    return True


def _detect_boilerplate(pages: list[str], min_repeat: int = 3) -> set[str]:
    """A line that appears (after strip) on at least `min_repeat` distinct pages
    is almost certainly header/footer/disclaimer boilerplate."""
    if len(pages) < min_repeat:
        return set()
    counts: dict[str, int] = {}
    for page in pages:
        seen_on_page: set[str] = set()
        for line in page.splitlines():
            line = line.strip()
            if len(line) < 3 or line in seen_on_page:
                continue
            seen_on_page.add(line)
            counts[line] = counts.get(line, 0) + 1
    return {line for line, n in counts.items() if n >= min_repeat}


def _strip_boilerplate(text: str, boilerplate: set[str]) -> str:
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped in boilerplate:
            continue
        if _PAGE_NUMBER_RE.match(stripped):
            continue
        out.append(line)
    return "\n".join(out)


def _table_to_markdown(table: list[list[str | None]]) -> str:
    """Render a table to GFM markdown. Defensive against None/whitespace cells
    so it can be called on raw rows too — `_clean_table_rows` is canonical
    cleaning, but this fallback keeps `_table_to_markdown` standalone."""
    if not table:
        return ""
    rows = [[(c or "").strip() for c in row] for row in table]
    header = rows[0]
    body = rows[1:] if len(rows) > 1 else []
    md = "| " + " | ".join(header) + " |\n"
    md += "| " + " | ".join("---" for _ in header) + " |\n"
    for row in body:
        md += "| " + " | ".join(row) + " |\n"
    return md
