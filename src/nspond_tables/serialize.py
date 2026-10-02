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

# A cell may hold several logical rows, one block element each: a journal
# writes `<td><p>33, 39, 15</p><p>27, 51, 3</p></td>` where the printed table
# shows two lines. Stripping those tags to nothing fused the two into
# `33, 39, 1527, 51, 3`, and `<p>7.26</p><p>4.17</p>` into `7.264.17` -- which
# has two decimal points, so the model echoing it emitted JSON that would not
# parse and the whole table was lost.
#
# Only block elements get the separator. An inline tag must still vanish
# without a trace, because `-<em>45</em>` has to stay `-45` and not become
# `- 45`, which is not a number.
_BLOCK = re.compile(r"</?(?:p|div|br|li|tr|h[1-6])\b[^>]*/?>", re.I)
_ENTITY = {
    "&nbsp;": " ", "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"',
    "&#x2212;": "-", "&minus;": "-", "&ndash;": "-", "&mdash;": "-", "&#8722;": "-",
}
_HTML_ROW = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.S | re.I)
# A cell may be self-closing: `<td rowspan="1" colspan="1"/>` is how
# several publishers write an empty cell. Requiring a closing tag did not
# just miss it -- the engine ran on to the NEXT `</td>`, swallowing the
# empty cell and the one after it into a single match, so every value in
# the row shifted one column left. A cortical thickness table read its
# P-value as Z that way. 27.1% of coordinate tables carry one (5,772
# tables sampled), across 40.4% of articles.
_HTML_CELL = re.compile(r"<(t[dh])\b([^>]*?)(?:/>|>(.*?)</\1\s*>)", re.S | re.I)
_CALS_ROW = re.compile(r"<row\b[^>]*>(.*?)</row>", re.S | re.I)
_CALS_CELL = re.compile(r"<entry\b([^>]*?)(?:/>|>(.*?)</entry\s*>)", re.S | re.I)
_SPAN = re.compile(r'\b(colspan|rowspan|morerows)\s*=\s*["\']?(\d+)', re.I)
_NAMES = re.compile(
    r'\bnamest\s*=\s*["\']?([^"\'\s>]+)[^>]*?\bnameend\s*=\s*["\']?([^"\'\s>]+)', re.I)
_COLNUM = re.compile(r"(\d+)")

# Newlines are collapsed with the other whitespace. They were not, and a cell
# holding a line break split its row across lines so the fragments read as rows
# of their own -- 7.2% of real tables (serializer_audit.py).
#
# `\s` rather than a hand-written class: the class left out the thin space,
# the non-breaking space and the en space, and a journal uses all three
# inside a number.
_WS = re.compile(r"[\s\u200b\ufeff]+")

# A journal writes the minus of a coordinate apart from its digits:
# `<td>-\u200945</td>`, a thin space between them -- `&#x02009;` is the
# fourth commonest entity in the corpus. Collapsing the whitespace leaves
# `- 45`, which reads as `45`: the sign is gone and a left-hemisphere focus
# lands on the right. ACE's own rewrite strips it, so only the tables found
# by scanning the article were wrong -- which also made the two renderings
# of one table disagree about their numbers and escape the duplicate check.
#
# Only a sign starting the cell is rejoined. A dash between two numbers is a
# range or a subtraction, and `10 - 20` must not become `10 -20`.
_LEADING_SIGN = re.compile(r"^([+-])\s+(?=[.\d])")

# A pdf-to-CSV conversion leaves NUL and other C0 bytes in the file. csv.reader
# raises on NUL, and that raise silently dropped 10.5% of pdf tables -- 12 of
# them carrying coordinates -- because a dropped table is not counted anywhere.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean(raw: str) -> str:
    s = _CONTROL.sub("", _TAG.sub("", _BLOCK.sub(" ", raw or "")))
    for k, v in _ENTITY.items():
        s = s.replace(k, v)
    # Case matters: `&#xA0;` has an uppercase A, so a lowercase-only pattern
    # left it in the cell and `-&#xA0;45` was not a number.
    s = re.sub(r"&[a-zA-Z#0-9]+;", " ", s)
    s = s.replace("−", "-").replace("–", "-").replace("�", "-")
    return _LEADING_SIGN.sub(r"\1", _WS.sub(" ", s).strip())


def from_html(raw: str) -> Grid:
    grid = Grid()
    for rm in _HTML_ROW.finditer(_drop_hidden(_DROP.sub(" ", raw or ""))):
        cells: List[Cell] = []
        for cm in _HTML_CELL.finditer(rm.group(1)):
            tag, attrs, inner = cm.group(1).lower(), cm.group(2), cm.group(3) or ""
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
            attrs, inner = cm.group(1), cm.group(2) or ""
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
