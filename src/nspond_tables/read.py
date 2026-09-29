"""Deterministic extraction from a rendered grid.

The header names the columns; the columns hold the values. Where that holds, no
model is needed: find the cell whose text is `x`, take its resolved column
index, and read straight down. Measured over 240 tables re-serialised from
source, this is exactly right on 61.7% of them and partly right on 74.2%
(evalkit/ser_modes.py). Against the previous serialiser the same reader scored
30.4% and 43.8%, and the whole difference is the grid work in `grid.py`.

What it deliberately does not do:

* guess. A field the table does not state comes back None, because a target that
  invents one teaches a model to invent them.
* group. Which rows belong to which analysis is the judgement this pipeline
  keeps a model for; `sections` only reports the banner rows it can see.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import fields
from .grid import Cell, Grid, Placed, as_number, is_header_row, parse

AXES = ("x", "y", "z")
# Generous bounds on a human head in MNI/Talairach millimetres. A triple outside
# them is not a coordinate, whatever column it sits in: a model-specification
# table yielded (22, 961, 706) and a features table (31, 112, 641), both of which
# a bound check rejects and nothing else does.
LIMITS = {"x": 90, "y": 126, "z": 108}
_AXIS = {a: re.compile(r"^\(?\s*%s\s*\)?(?:\s*\(?\s*mm\s*\)?)?$" % a, re.I) for a in AXES}
_AXIS_SUFFIX = {a: re.compile(r"[.\s\-]%s$" % a, re.I) for a in AXES}
_EXTENT = re.compile(
    r"cluster\s*(?:size|extent)|extent|\bsize\b|\bvoxels?\b|\bn\s*vox|\bk\b|\bke\b|mm\s*\^?3",
    re.I)
# "Values shown are T statistics", "T values are given for the peak voxel".
# Deliberately narrow: a threshold sentence must not match.
_VALUES_ARE = re.compile(
    r"values?\s+(?:shown\s+)?(?:are|is|represent\w*)\s+(?:the\s+)?([A-Za-z]{1,6})\b"
    r"|\b([A-Za-z]{1,6})\s+values?\s+(?:are\s+)?(?:given|shown|reported|listed)",
    re.I)
# A sign may be written apart from its digits (`- 50`) and a positive
# coordinate is often printed with its plus (`+68 -16 34`, `-44 -90 + 12`).
# Both forms are common enough in elsevier tables to matter, and neither
# parsed.
_SIGNED = r"[-+\u2212\u2013]?\s*\d{1,3}(?:\.\d+)?"
_TRIPLE = re.compile(r"^\s*(%s)[\s,;]+(%s)[\s,;]+(%s)\s*$" % ((_SIGNED,) * 3))
# Several peaks reported on one row, one per sub-value: `-8/12//-6`.
_SPLIT = re.compile(r"[/|]+")
# `(-51, 20, 24)`, `[-51 20 24]` -- the brackets are decoration.
_WRAPPED = re.compile(r"^\s*[\(\[\{]\s*(.*?)\s*[\)\]\}]\s*$", re.S)
# `-10-42 16` is three numbers: a minus straight after a digit starts a new one.
_RUN_ON = re.compile(r"[-+\u2212\u2013]?\s*\d{1,3}(?:\.\d+)?")
# One cell, several peaks: `-16, -54, 46; -22, -54, 52`.
_PEAK_SPLIT = re.compile(r"[;/]|\band\b", re.I)


def _looks_like_a_coordinate(v: Optional[float]) -> bool:
    """A coordinate is a millimetre count, so it is whole or nearly whole.

    This is the one judgement the reader makes about a number, because no
    paper reports a peak to five decimal places and reading `0.051118` as a
    millimetre is a parse error rather than a close call. Everything else the
    reader finds is a candidate, and rejecting candidates is the gate's job.
    """
    return v is not None and abs(round(v, 2) - v) < 1e-9


def triples_in(text: str) -> List[Tuple[float, float, float]]:
    """Every coordinate triple a single cell states, in order.

    Handles the four forms the corpus actually uses: `-42 -55 -18`,
    `(-42, -55, -18)`, `-42-55 -18` with the signs run together, and several
    peaks in one cell separated by a semicolon.
    """
    def one(part):
        wrapped = _WRAPPED.match(part)
        if wrapped:
            part = wrapped.group(1)
        nums = [as_number(m.group()) for m in _RUN_ON.finditer(part)]
        if len(nums) != 3 or not all(_looks_like_a_coordinate(v) for v in nums):
            return None
        trio = tuple(nums)
        return trio if in_head(trio) else None

    # The whole cell first. A semicolon separates several peaks in one cell,
    # but it also separates the axes of a single one -- `10.2; 20.5; 19.5` --
    # and splitting before trying the cell whole lost those.
    whole = one(str(text or ""))
    if whole:
        return [whole]
    out = []
    for part in _PEAK_SPLIT.split(str(text or "")):
        trio = one(part)
        if trio:
            out.append(trio)
    return out
_NUMBERS = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


@dataclass
class Point:
    x: float
    y: float
    z: float
    statistic_type: Optional[str] = None
    statistic_value: Optional[float] = None
    extent: Optional[float] = None
    row: int = -1
    label: str = ""

    def as_tuple(self) -> Tuple:
        return (self.x, self.y, self.z, self.statistic_type,
                self.statistic_value, self.extent)


@dataclass
class Result:
    points: List[Point] = field(default_factory=list)
    space: Optional[str] = None
    measure: Optional[str] = None
    sections: List[Tuple[int, str]] = field(default_factory=list)
    axis_columns: Optional[Dict[str, int]] = None
    packed_column: Optional[int] = None
    sign_disagreements: int = 0

    @property
    def located_by(self) -> str:
        if self.axis_columns:
            return "header"
        if self.packed_column is not None:
            return "packed cell"
        return "none"


def numbers_in(text: str) -> List[float]:
    """Every number in a cell, for composite forms like `t(79)=6.24`.

    Curated labels read values out of these, so a reader that only accepts a
    bare float disagrees with them on 4.63% of values.
    """
    out = []
    s = str(text).replace("−", "-").replace("–", "-")
    s = re.sub(r"(?<=\d)[,  ](?=\d{3}(?!\d))", "", s)
    for m in _NUMBERS.finditer(s):
        try:
            out.append(float(m.group()))
        except ValueError:
            pass
    return out


def _header_cells(grid: Grid) -> List[Placed]:
    """Header cells, or the leading text rows when the table marks none.

    Three of the seventeen real coordinate tables in the uncertain band write
    their header in <td>. The axis row is right there and reads exactly as it
    would in <th>, so the only thing standing between the reader and the
    coordinates is a tag the publisher chose. Rows before the first number,
    and at most three of them, stand in when there is nothing else.
    """
    rows = grid.resolve()
    marked = [p for row in rows for p in row if p.cell.header]
    if marked:
        return marked
    stand_in = []
    for row in rows[:3]:
        if any(as_number(p.cell.text) is not None for p in row):
            break
        stand_in.extend(row)
    return stand_in


def axis_columns(grid: Grid) -> Optional[Dict[str, int]]:
    """Column index of x, y and z, from the header.

    Accepts a bare axis name and a qualified one -- Docling's DataFrame export
    writes `MNI.x`, and papers write `x (mm)`.

    The three must be *consecutive*, which is not a nicety. A Z-statistic column
    sits beside a z coordinate in a great many real tables, and its header `Z`
    matches the axis name exactly. Taking the first match per axis picked the
    statistic column and read the statistic as the z coordinate on 4.2% of
    generated tables. Requiring x, y and z to be adjacent resolves it without
    needing to know which column is which kind.
    """
    candidates: Dict[str, List[int]] = {a: [] for a in AXES}
    for placed in _header_cells(grid):
        text = placed.cell.text.strip()
        for axis in AXES:
            if _AXIS[axis].match(text) or _AXIS_SUFFIX[axis].search(text):
                if placed.col not in candidates[axis]:
                    candidates[axis].append(placed.col)
    for x in sorted(candidates["x"]):
        if x + 1 in candidates["y"] and x + 2 in candidates["z"]:
            return {"x": x, "y": x + 1, "z": x + 2}
    # No adjacent triple: accept one unambiguous candidate each, in order.
    if all(len(candidates[a]) == 1 for a in AXES):
        cols = [candidates[a][0] for a in AXES]
        if len(set(cols)) == 3 and cols == sorted(cols):
            return dict(zip(AXES, cols))
    return None


_SPANS_COORDS = re.compile(
    r"coord|\bMNI\b|talairach|\bTAL\b|stereotax|\bpeak\b|x\s*[,/ ]\s*y\s*[,/ ]\s*z", re.I)


def spanned_axis_columns(grid: Grid) -> Optional[Dict[str, int]]:
    """Three columns under one `MNI coordinates` header that names no axes.

    Six of the seventeen real tables in the uncertain band head their
    coordinates once, spanning three columns, and never write x, y and z at
    all. The span says which columns they are; the body says whether the
    reading is sane, so the columns are only accepted when most rows put three
    numbers there and those numbers fit in a head.
    """
    for placed in _header_cells(grid):
        if placed.cell.colspan != 3 or not _SPANS_COORDS.search(placed.cell.text):
            continue
        cols = list(placed.cols)
        if _columns_hold_coordinates(grid, cols):
            return dict(zip(AXES, cols))
    return None


def _columns_hold_coordinates(grid: Grid, cols: Sequence[int]) -> bool:
    """Whether three columns actually read as coordinates in the body.

    A header row that spans, followed by a sub-header row with only the spanned
    cells in it, lands those sub-headers at the left edge instead of under the
    span: `x | y | z` registers at columns 0, 1, 2, where column 0 is the
    region name. The axis names are then found in the wrong place and the
    reader quietly returns nothing. The body settles it.
    """
    good = total = 0
    for row in grid.body_rows():
        by_col = {c: pl.cell for pl in row for c in pl.cols}
        vals = [as_number(by_col.get(c, Cell()).text) for c in cols]
        if all(v is None for v in vals):
            continue
        total += 1
        if all(v is not None for v in vals) and in_head(vals):
            good += 1
    return bool(total) and good >= 1 and good >= 0.5 * total


def packed_column(grid: Grid) -> Optional[int]:
    """The column whose cells each hold a whole coordinate triple.

    47% of pdf tables and ~5% elsewhere write coordinates this way. A column
    qualifies when most of its non-empty body cells parse as a triple.
    """
    hits: Dict[int, int] = {}
    total: Dict[int, int] = {}
    for row in grid.body_rows():
        for placed in row:
            if placed.cell.is_filler or not placed.cell.text.strip():
                continue
            total[placed.col] = total.get(placed.col, 0) + 1
            if triples_in(placed.cell.text):
                hits[placed.col] = hits.get(placed.col, 0) + 1
    best = [c for c, n in hits.items() if n >= 2 and n >= 0.6 * total.get(c, 1)]
    return min(best) if best else None


def _column_header_text(grid: Grid, col: int) -> str:
    """Every header cell covering a column, outermost first.

    A column under `#<3:MNI coordinates` and `#x` belongs to both, and the unit
    or the statistic name usually sits in the outer one.
    """
    parts = []
    for row in grid.resolve():
        for placed in row:
            if placed.cell.header and col in placed.cols:
                if placed.cell.text.strip():
                    parts.append(placed.cell.text.strip())
    return " ".join(parts)


def _statistic_columns(grid: Grid, skip: Sequence[int],
                       context: str = "") -> List[Tuple[int, str]]:
    """Columns holding a statistic, and which one.

    The header names it on 44% of real tables. Another 4% name it only in a
    footnote -- "Values shown are T statistics" -- and for those the column is
    still identifiable: it is the numeric column left over once the coordinates
    and the extent are claimed. Reading the footnote recovers the type without
    guessing, so it is done here rather than left to a model.
    """
    numeric = set(_numeric_columns(grid, skip))
    out = []
    for col in range(grid.width()):
        if col in skip or col not in numeric:
            continue
        kind = fields.statistic_type(_column_header_text(grid, col))
        if kind:
            out.append((col, kind))
    if out or not context:
        return out
    # Only a sentence that says what the VALUES are counts. "The statistical
    # threshold was set at p<0.05" names a threshold, not the column's
    # statistic, and reading it as one both invented a P type and claimed the
    # cluster-size column as the statistic, losing the extent with it.
    kind = claimed_statistic(context)
    if not kind:
        return out
    for col in _numeric_columns(grid, skip):
        if _EXTENT.search(_column_header_text(grid, col)):
            continue
        return [(col, kind)]
    return out


def _unnamed_value_column(grid: Grid, skip: Sequence[int]) -> Optional[int]:
    """A leftover numeric column holding a statistic of unstated kind.

    Papers head these "Value", or leave them blank, on roughly half of real
    tables. The number is printed, so a reader can report it; what it cannot do
    is name it. Reporting the value with a null type is the honest answer, and
    it is what the curated labels do.
    """
    cols = [c for c in _numeric_columns(grid, skip)
            if not _EXTENT.search(_column_header_text(grid, c))]
    return cols[0] if len(cols) == 1 else None


def claimed_statistic(context: str) -> Optional[str]:
    """The statistic a footnote says the values ARE, or None.

    Narrow on purpose. "The statistical threshold was set at p<0.05" names a
    threshold, and a footer often carries both that and a real claim, so a
    reader that takes any statistic mention picks the wrong one.
    """
    claim = _VALUES_ARE.search(context or "")
    if not claim:
        return None
    named = next((g for g in claim.groups() if g), None)
    return fields.statistic_type(named) if named else None


def _numeric_columns(grid: Grid, skip: Sequence[int]) -> List[int]:
    """Columns whose body cells are mostly numbers, left to right."""
    hits: Dict[int, int] = {}
    total: Dict[int, int] = {}
    for row in grid.body_rows():
        for placed in row:
            if placed.cell.is_filler or not placed.cell.text.strip():
                continue
            total[placed.col] = total.get(placed.col, 0) + 1
            if as_number(placed.cell.text) is not None:
                hits[placed.col] = hits.get(placed.col, 0) + 1
    return [c for c in sorted(hits)
            if c not in skip and hits[c] >= 0.6 * total.get(c, 1)]


def _extent_column(grid: Grid, skip: Sequence[int]) -> Optional[int]:
    for col in range(grid.width()):
        if col in skip:
            continue
        if _EXTENT.search(_column_header_text(grid, col)):
            return col
    return None


def _row_label(row: Sequence[Placed], skip: Sequence[int]) -> str:
    """The row's own name, inheriting through a rowspan.

    A row whose region cell is covered from above belongs to that region, so a
    filler resolves to the text covering it rather than to nothing.
    """
    for placed in row:
        if placed.col in skip:
            continue
        text = placed.cell.covers if placed.cell.is_filler else placed.cell.text
        if text.strip() and as_number(text) is None:
            return text.strip()
    return ""


def is_section(row: Sequence[Placed], width: int) -> bool:
    """A banner row: one piece of text and nothing else on the row.

    Both renderings count -- a lone full-width cell, and a partial span with the
    rest of the row empty or filled. The second form is 1% of real tables and
    the synthetic generator never produced it.
    """
    texts = [p for p in row if not p.cell.is_filler and p.cell.text.strip()]
    if len(texts) != 1:
        return False
    cell = texts[0].cell
    if as_number(cell.text) is not None:
        return False
    return cell.colspan > 1 or len(row) == 1 or all(
        p.cell.is_filler or not p.cell.text.strip() for p in row if p is not texts[0])


def extract(text_or_grid, caption: str = "", footer: str = "",
            abstract: str = "") -> Result:
    """Everything the table itself determines. Grouping is left to a model."""
    grid = parse(text_or_grid) if isinstance(text_or_grid, str) else text_or_grid
    res = Result()
    res.space = fields.visible_space(grid.render(), caption, footer, abstract)

    # The named columns win unless a spanning header disagrees and the body
    # sides with the span. Checking the named columns on their own would
    # reject thirteen tables that read correctly, so the check only settles a
    # disagreement.
    axes = axis_columns(grid)
    spanned = spanned_axis_columns(grid)
    if axes and spanned and axes != spanned \
            and not _columns_hold_coordinates(grid, list(axes.values())):
        axes = spanned
    axes = axes or spanned
    packed = None if axes else packed_column(grid)
    res.axis_columns, res.packed_column = axes, packed
    coord_cols = list(axes.values()) if axes else ([packed] if packed is not None else [])
    if not coord_cols:
        return res

    stat_cols = _statistic_columns(
        grid, coord_cols, " ".join([caption or "", footer or ""]))
    ext_col = _extent_column(grid, coord_cols + [c for c, _ in stat_cols])
    if not stat_cols:
        loose = _unnamed_value_column(
            grid, coord_cols + ([ext_col] if ext_col is not None else []))
        if loose is not None:
            stat_cols = [(loose, None)]
    res.measure = fields.cluster_measure(
        _column_header_text(grid, ext_col)) if ext_col is not None else None

    width = grid.width()
    all_header = grid.all_header_rows()
    for r, row in enumerate(grid.resolve()):
        if is_header_row(row) and not all_header:
            continue
        if all_header and not any(as_number(p.cell.text) is not None for p in row):
            continue
        if is_section(row, width):
            label = next(p.cell.text.strip() for p in row
                         if not p.cell.is_filler and p.cell.text.strip())
            res.sections.append((r, label))
            continue
        by_col: Dict[int, Cell] = {}
        for placed in row:
            for c in placed.cols:
                by_col.setdefault(c, placed.cell)
        triples = _coords_from_row(by_col, axes, packed)
        if not triples:
            continue
        stat_type = stat_val = None
        for col, kind in stat_cols:   # kind is None for an unnamed value column
            v = as_number(by_col.get(col, Cell()).text)
            if v is None:
                nums = numbers_in(by_col.get(col, Cell()).text)
                v = nums[-1] if nums else None
            if v is not None:
                stat_type, stat_val = kind, v
                break
        extent = as_number(by_col.get(ext_col, Cell()).text) if ext_col is not None else None
        label = _row_label(row, coord_cols)
        for xyz in triples:
            point = Point(*xyz, statistic_type=stat_type, statistic_value=stat_val,
                          extent=extent, row=r, label=label)
            if fields.sign_agrees(point.x, label) is False:
                res.sign_disagreements += 1
            res.points.append(point)
    return res


def in_head(xyz: Sequence[float]) -> bool:
    """Whether a triple could be a brain coordinate at all."""
    return all(abs(v) <= LIMITS[a] for a, v in zip(AXES, xyz))


def _coords_from_row(by_col: Dict[int, Cell], axes, packed) -> List[Tuple[float, float, float]]:
    """Every coordinate the row states -- usually one, sometimes several.

    A cluster with more than one peak is written `-8/12//-6 | 56/48/52 |
    38/48/30`: one row, three points, the axes still in their own columns.
    Reading only the whole cell lost all three.
    """
    if axes:
        parts = []
        for axis in AXES:
            cell = by_col.get(axes[axis])
            if cell is None:
                return []
            vals = [as_number(t) for t in _SPLIT.split(cell.text) if t.strip()]
            if not vals or any(v is None for v in vals):
                return []
            parts.append(vals)
        if len({len(v) for v in parts}) != 1:
            return []
        out = [tuple(v[i] for v in parts) for i in range(len(parts[0]))]
        return [xyz for xyz in out if in_head(xyz)]
    cell = by_col.get(packed)
    return triples_in(cell.text) if cell is not None else []
