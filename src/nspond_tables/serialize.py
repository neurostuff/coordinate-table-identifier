"""Turn a publisher's table into a Grid.

Four input shapes reach this pipeline:

    pubget, ace   <tr>/<td>          HTML
    elsevier      <row>/<entry>      CALS; spans are namest/nameend and morerows
    pdf           comma-separated    Docling's DataFrame written with to_csv

Tags are most of an HTML table's bytes but not all of them are noise: a spanning
cell is how a paper writes a row-group header, and that is the strongest signal
for which rows belong to which analysis. Spans survive; styling does not.

The CSV branch needs more care than it reads. Docling keeps the header as dotted
column names and repeats rowspan values, so the loss is smaller than it looks --
but a naive `split(",")` breaks quoted fields, and 75.1% of pdf tables carried a
stray quote because of it (by_source.py, 494 pdf tables). Reading it with
`csv.reader` and marking the first numberless row as the header took pdf from 0%
to 100% on the header-directed read.
"""

from __future__ import annotations

import csv
import io
import re
from typing import Iterator, List, Optional

from .grid import Cell, Grid

_DROP = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)

# Oxford Academic hides a sort marker in every <th>: `<span aria-hidden="true"
# style="display: none"> . </span>`. A reader never sees it, but it rode into
# the cell text, and `x .` is not an axis name, so three of the seventeen real
# coordinate tables in the uncertain band read as none.
_HIDDEN = re.compile(
    r"""<(\w+)\b[^>]*?(?:aria-hidden\s*=\s*["']?true|
        style\s*=\s*["'][^"']*display\s*:\s*none)[^>]*>(.*?)</\1\s*>""",
    re.S | re.I | re.X)
_ROW_TAG = re.compile(r"<(?:tr|t[dh]|row|entry)\b", re.I)


def _drop_hidden(raw: str) -> str:
    """Remove elements a sighted reader never sees, unless one holds the table.

    A whole table inside a collapsed container is hidden too, and removing it
    would leave nothing at all, so anything carrying rows of its own stays.
    """
    def repl(m):
        return " " if not _ROW_TAG.search(m.group(2)) else m.group(0)
    return _HIDDEN.sub(repl, raw)
_TAG = re.compile(r"<[^>]+>")
_ENTITY = {
    "&nbsp;": " ", "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"',
    "&#x2212;": "-", "&minus;": "-", "&ndash;": "-", "&mdash;": "-", "&#8722;": "-",
}
_HTML_ROW = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.S | re.I)
_HTML_CELL = re.compile(r"<(t[dh])\b([^>]*)>(.*?)</\1>", re.S | re.I)
_CALS_ROW = re.compile(r"<row\b[^>]*>(.*?)</row>", re.S | re.I)
_CALS_CELL = re.compile(r"<entry\b([^>]*)>(.*?)</entry>", re.S | re.I)
_SPAN = re.compile(r'\b(colspan|rowspan|morerows)\s*=\s*["\']?(\d+)', re.I)
_NAMES = re.compile(
    r'\bnamest\s*=\s*["\']?([^"\'\s>]+)[^>]*?\bnameend\s*=\s*["\']?([^"\'\s>]+)', re.I)
_COLNUM = re.compile(r"(\d+)")

# Newlines are collapsed with the other whitespace. They were not, and a cell
# holding a line break split its row across lines so the fragments read as rows
# of their own -- 7.2% of real tables (serializer_audit.py).
_WS = re.compile(r"[ \t \r\n\f\v]+")

# A pdf-to-CSV conversion leaves NUL and other C0 bytes in the file. csv.reader
# raises on NUL, and that raise silently dropped 10.5% of pdf tables -- 12 of
# them carrying coordinates -- because a dropped table is not counted anywhere.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean(raw: str) -> str:
    s = _CONTROL.sub("", _TAG.sub("", raw))
    for k, v in _ENTITY.items():
        s = s.replace(k, v)
    s = re.sub(r"&[a-z#0-9]+;", " ", s)
    s = s.replace("−", "-").replace("–", "-").replace("�", "-")
    return _WS.sub(" ", s).strip()


def from_html(raw: str) -> Grid:
    grid = Grid()
    for rm in _HTML_ROW.finditer(_drop_hidden(_DROP.sub(" ", raw or ""))):
        cells: List[Cell] = []
        for cm in _HTML_CELL.finditer(rm.group(1)):
            tag, attrs, inner = cm.group(1).lower(), cm.group(2), cm.group(3)
            span = {k.lower(): int(v) for k, v in _SPAN.findall(attrs)}
            cells.append(Cell(
                text=clean(inner),
                header=(tag == "th"),
                colspan=span.get("colspan", 1),
                rowspan=span.get("rowspan", 1),
            ))
        if cells:
            grid.add(cells)
    return grid


def from_cals(raw: str) -> Grid:
    body = _drop_hidden(_DROP.sub(" ", raw or ""))
    head_end = body.lower().find("</thead>") if re.search(r"<thead\b", body, re.I) else -1
    grid = Grid()
    for rm in _CALS_ROW.finditer(body):
        header = head_end > 0 and rm.start() < head_end
        cells: List[Cell] = []
        for cm in _CALS_CELL.finditer(rm.group(1)):
            attrs, inner = cm.group(1), cm.group(2)
            span = {k.lower(): int(v) for k, v in _SPAN.findall(attrs)}
            colspan = 1
            named = _NAMES.search(attrs)
            if named:
                a, b = (_COLNUM.search(x) for x in named.groups())
                if a and b:
                    colspan = max(1, int(b.group(1)) - int(a.group(1)) + 1)
            cells.append(Cell(
                text=clean(inner),
                header=header,
                colspan=colspan,
                rowspan=span.get("morerows", 0) + 1,
            ))
        if cells:
            grid.add(cells)
    return grid


def from_csv(raw: str) -> Grid:
    body = _CONTROL.sub("", (raw or "").replace("\r\n", "\n"))
    rows = []
    try:
        parsed = list(csv.reader(io.StringIO(body)))
    except csv.Error:
        # Malformed quoting. A naive split loses a field that holds a comma,
        # which is worse than the old behaviour but better than no table.
        parsed = [line.split(",") for line in body.split("\n")]
    for parts in parsed:
        cells = [clean(p) for p in parts]
        if any(cells):
            rows.append(cells)
    head = next((i for i, r in enumerate(rows)
                 if not any(re.search(r"\d", c) for c in r)), None)
    grid = Grid()
    for i, row in enumerate(rows):
        grid.add([Cell(text=c, header=(i == head)) for c in row])
    return grid


def from_source(raw: str, source: Optional[str] = None) -> Grid:
    """Dispatch on the markup, not on the source name.

    The source is advisory: an ace article can arrive as CALS and an elsevier one
    as HTML, so what is in the bytes decides.
    """
    body = (raw or "").lower()
    if "<row" in body and "<entry" in body:
        return from_cals(raw)
    if "<tr" in body:
        return from_html(raw)
    return from_csv(raw)


def render(grid: Grid, max_chars: int = 0) -> str:
    return grid.render(max_chars=max_chars)


def serialize(raw: str, source: Optional[str] = None, max_chars: int = 0) -> str:
    """Raw bytes to rendered text in one call.

    Never raises. A caller that drops a table on exception loses it silently,
    and a lost table is not counted as a miss by anything downstream -- which is
    how 12 pdf tables carrying coordinates went unnoticed. A table that cannot
    be parsed comes back as its cleaned text instead of nothing.
    """
    try:
        text = render(from_source(raw, source), max_chars=max_chars)
    except Exception:
        text = ""
    if text.strip():
        return text
    fallback = clean(raw or "")
    return fallback[:max_chars] if max_chars else fallback
