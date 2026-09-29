"""The cell grid every source and the generator build, and its rendered form.

A table is a list of rows; a row is a list of cells; a cell knows how far it
spans. Column *index* is not stored, because in markup it is not given -- a cell
covered by a rowspan above it is simply absent -- so `Grid.resolve` computes it.

The rendered form marks a cell with a prefix:

    #        header cell
    <N:      spans N columns
    ^N:      spans N rows
    ~        a cell covered by a rowspan above it

The colon matters. Without it `^2` followed by the text `37` renders `^237`,
which reads equally as rowspan 2 over "37", rowspan 23 over "7", or rowspan 237.
16.9% of real tables carried one of those (serializer_audit.py, 11,215 tables).

`~` matters because omitting a covered cell shifts every later column in that row
with nothing to mark it, which is why a reader following the axis header landed
in the wrong column on more than half of real tables.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Sequence

SEP = " | "
COVERED = "~"
SPAN_DELIM = ":"

_PREFIX = re.compile(r"^(#?)(?:<(\d+):)?(?:\^(\d+):)?")


@dataclass
class Cell:
    text: str = ""
    header: bool = False
    colspan: int = 1
    rowspan: int = 1
    covers: str = ""     # on a filler: the text of the cell spanning into it

    def __post_init__(self) -> None:
        self.colspan = max(1, int(self.colspan))
        self.rowspan = max(1, int(self.rowspan))

    @property
    def is_filler(self) -> bool:
        """A cell standing in for one a rowspan above it covers."""
        return self.text == COVERED and not self.header

    def value(self) -> Optional[float]:
        """The cell as a number, or None.

        Tolerates a leading inequality, a trailing footnote mark, a thousands
        separator written with a comma or a space, and a unicode minus.
        """
        return as_number(self.text)


@dataclass
class Placed:
    """A cell together with the column it occupies once spans are resolved."""

    cell: Cell
    row: int
    col: int

    @property
    def cols(self) -> range:
        return range(self.col, self.col + self.cell.colspan)


@dataclass
class Grid:
    rows: List[List[Cell]] = field(default_factory=list)

    def add(self, cells: Sequence[Cell]) -> "Grid":
        self.rows.append(list(cells))
        return self

    def __len__(self) -> int:
        return len(self.rows)

    # -- geometry ---------------------------------------------------------
    def resolve(self) -> List[List[Placed]]:
        """Place every cell at its true column, inserting nothing.

        Walks the grid tracking how many further rows each column is covered
        for. A filler cell already present in the row is consumed rather than
        re-inserted, so resolving a grid that came from `parse` is idempotent.
        """
        pending: dict = {}
        covering: dict = {}
        out: List[List[Placed]] = []
        for r, row in enumerate(self.rows):
            placed: List[Placed] = []
            col = 0
            for cell in row:
                if not cell.is_filler:
                    while pending.get(col, 0) > 0:
                        pending[col] -= 1
                        placed.append(Placed(
                            Cell(COVERED, covers=covering.get(col, "")), r, col))
                        col += 1
                else:
                    pending[col] = max(0, pending.get(col, 0) - 1)
                    cell.covers = cell.covers or covering.get(col, "")
                placed.append(Placed(cell, r, col))
                if cell.rowspan > 1:
                    for k in range(col, col + cell.colspan):
                        pending[k] = cell.rowspan - 1
                        covering[k] = cell.text
                col += cell.colspan
            while pending.get(col, 0) > 0:
                pending[col] -= 1
                placed.append(Placed(Cell(COVERED, covers=covering.get(col, "")), r, col))
                col += 1
            out.append(placed)
        return out

    def width(self) -> int:
        """Columns in the widest row, spans expanded."""
        return max((sum(c.cell.colspan for c in row) for row in self.resolve()), default=0)

    def filled(self) -> "Grid":
        """A copy in which every covered cell is present as a filler.

        This is what `render` emits, and what makes column position readable
        without the reader tracking spans itself.
        """
        g = Grid()
        for row in self.resolve():
            g.add([p.cell for p in row])
        return g

    # -- rendering --------------------------------------------------------
    def render(self, max_chars: int = 0) -> str:
        lines = []
        for row in self.resolve():
            if not any(_informative(p.cell) for p in row):
                continue
            lines.append(SEP.join(render_cell(p.cell) for p in row))
        text = "\n".join(lines)
        return text[:max_chars] if max_chars else text

    def header_rows(self) -> Iterator[List[Placed]]:
        for row in self.resolve():
            if any(p.cell.header for p in row):
                yield row

    def body_rows(self) -> Iterator[List[Placed]]:
        """Rows carrying data.

        A row is a header row only when *every* cell in it is a header. HTML
        marks the leftmost label column `<th>` as well as the top row -- a row
        header, not a column header -- and treating any header cell as
        disqualifying dropped real data rows: Docling's export of one table
        yielded 11 points where its CSV yielded 14, purely because the region
        column was tagged.

        Some publishers mark every cell `<th>`, which leaves no body at all by
        that rule. When that happens a row holding a number is data whatever its
        tag says, because the header rows of such a table are the ones with no
        numbers in them.
        """
        rows = self.resolve()
        all_header = is_header_row
        if rows and all(all_header(row) for row in rows):
            for row in rows:
                if any(as_number(p.cell.text) is not None for p in row):
                    yield row
            return
        for row in rows:
            if not all_header(row):
                yield row

    def all_header_rows(self) -> bool:
        """True when every row is nothing but headers, as some publishers write."""
        rows = self.resolve()
        return bool(rows) and all(is_header_row(row) for row in rows)


def is_header_row(row: Sequence[Placed]) -> bool:
    """A row is a header row only when every cell in it is a header.

    HTML marks the leftmost label column `<th>` as well as the top row -- a row
    header, not a column header. Treating any header cell as disqualifying
    dropped real data rows: Docling's HTML export of one table yielded 11 points
    where its own CSV yielded 14, purely because the region column was tagged.
    """
    cells = [p.cell for p in row if not p.cell.is_filler and p.cell.text.strip()]
    return bool(cells) and all(c.header for c in cells)


def _informative(cell: Cell) -> bool:
    """A row of nothing but markers and separators carries no content."""
    if cell.is_filler:
        return False
    return bool(cell.text.strip(" #<^|~:0123456789\\")) or bool(re.search(r"\d", cell.text))


def render_cell(cell: Cell) -> str:
    if cell.is_filler:
        return COVERED
    mark = "#" if cell.header else ""
    if cell.colspan > 1:
        mark += "<%d%s" % (cell.colspan, SPAN_DELIM)
    if cell.rowspan > 1:
        mark += "^%d%s" % (cell.rowspan, SPAN_DELIM)
    return mark + escape(cell.text)


def escape(text: str) -> str:
    """Keep a pipe or a leading hash in cell text from reading as syntax."""
    text = text.replace("|", r"\|")
    if text.startswith("#"):
        text = "\\" + text
    return text


def unescape(text: str) -> str:
    if text.startswith("\\#"):
        text = text[1:]
    return text.replace(r"\|", "|")


def parse(text: str) -> Grid:
    """Read a rendered grid back. The inverse of `render`, for tests and readers."""
    g = Grid()
    for line in text.split("\n"):
        if not line.strip():
            continue
        cells = []
        for raw in re.split(r"(?<!\\)\|", line):
            raw = raw.strip()
            if raw == COVERED:
                cells.append(Cell(COVERED))
                continue
            m = _PREFIX.match(raw)
            cells.append(Cell(
                text=unescape(raw[m.end():]),
                header=bool(m.group(1)),
                colspan=int(m.group(2) or 1),
                rowspan=int(m.group(3) or 1),
            ))
        g.add(cells)
    return g


_INEQ = re.compile(r"^[<>~≤≥=¡]\s*")
_FOOT = "*†‡§¶°abcde"


def as_number(text: str) -> Optional[float]:
    """The number a cell denotes, or None.

    Curated labels read numbers out of cells like `t(79)=6.24` and `4.03a`, so a
    reader that only accepts a bare float disagrees with them on 4.63% of values
    (34(a) in scans/PRE_V17.md). This handles the simple decorations; a composite
    cell needs `read.numbers_in`.
    """
    s = unescape(str(text)).strip().lstrip("#").strip()
    s = s.rstrip(_FOOT).strip()
    s = _INEQ.sub("", s).strip()
    s = s.replace("−", "-").replace("–", "-")
    # "- 50" is how some publishers set a negative, and the space must go
    # before the thousands separator is stripped or it reads as one. Missing
    # this dropped whole coordinate tables: a row reading
    # `Middle frontal gyrus | L | 527 | - 50 | 34 | 34` parsed as having no
    # numeric triple at all.
    s = re.sub(r"^([-+])\s+(?=[\d.])", r"\1", s)
    s = re.sub(r"(?<=\d)[,  ](?=\d{3}(?!\d))", "", s)
    try:
        return float(s)
    except ValueError:
        return None
