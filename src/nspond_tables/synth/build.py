"""Build a synthetic table as a Grid, with the truth it should yield.

The generator returns a `Grid`, never text. `serialize.render` then produces the
serialised form, which is the same function real tables go through -- so the two
halves of the training set cannot disagree about the format. The previous
generator wrote strings itself, and every format defect found during the v16-v18
runs was a symptom of that: banners whose spans were never widened, rowspan
cells blanked where real ones were omitted, targets asserting values the table
never printed.

Two rules the truth obeys:

* a field the table does not print is null. `Point.statistic_value` is set only
  when a statistic cell was actually emitted, so no target can assert a value a
  reader cannot see.
* a coordinate is drawn from its region, so the sign agrees with the laterality
  in the row's own text and nothing lands outside a head.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..grid import Cell, Grid
from . import vocab
from .weights import DEFAULT, Weights


@dataclass
class TruthPoint:
    x: float
    y: float
    z: float
    statistic_type: Optional[str] = None
    statistic_value: Optional[float] = None
    extent: Optional[float] = None

    def as_list(self) -> List:
        return [self.x, self.y, self.z, self.statistic_type,
                self.statistic_value, self.extent]


@dataclass
class TruthAnalysis:
    name: str
    points: List[TruthPoint] = field(default_factory=list)
    measure: Optional[str] = None


@dataclass
class Truth:
    space: Optional[str] = None
    analyses: List[TruthAnalysis] = field(default_factory=list)

    def as_target(self) -> Dict:
        out: Dict = {"space": self.space, "analyses": []}
        for a in self.analyses:
            item: Dict = {"name": a.name}
            if a.measure is not None:
                item["measure"] = a.measure
            item["points"] = [p.as_list() for p in a.points]
            out["analyses"].append(item)
        return out


@dataclass
class Table:
    grid: Grid
    truth: Truth
    caption: str = ""
    footer: str = ""
    notes: Dict = field(default_factory=dict)   # which levers fired, for auditing


def _round(rng: random.Random, value: float, w: Weights, *,
           fractional: bool = False) -> float:
    if fractional:
        return round(value + rng.choice([-0.5, 0.5, 0.2, -0.2]), 1)
    # Papers report on a 2mm grid far more often than not, so an even
    # coordinate is not evidence of anything.
    return float(int(value) // 2 * 2)


def _coord(rng: random.Random, region: vocab.Region, w: Weights,
           zero_x: bool = False) -> Tuple[float, float, float]:
    # One roll for the point, not three. Rolling per axis made 1 - (1 -
    # 0.11)^3 = 29.5% of points fractional where the weight says 11%.
    frac = rng.random() < w.non_integer_coordinates
    x = 0.0 if zero_x else _round(rng, rng.uniform(*region.x), w, fractional=frac)
    y = _round(rng, rng.uniform(*region.y), w, fractional=frac)
    z = _round(rng, rng.uniform(*region.z), w, fractional=frac)
    return (max(-w.x_limit, min(w.x_limit, x)),
            max(-w.y_limit, min(w.y_limit, y)),
            max(-w.z_limit, min(w.z_limit, z)))


def _stat_value(rng: random.Random, kind: str) -> float:
    if kind == "P":
        return round(rng.choice([0.001, 0.005, 0.01, 0.02, 0.04]), 4)
    if kind == "R":
        return round(rng.uniform(0.2, 0.8) * rng.choice([1, -1]), 2)
    if kind == "B":
        return round(rng.uniform(0.05, 3.0) * rng.choice([1, -1]), 2)
    return round(rng.uniform(2.5, 12.0), 2)


def _analysis_name(rng: random.Random) -> str:
    kind = rng.random()
    if kind < 0.45:
        a, b = rng.sample(vocab.CONDITIONS, 2)
        return "%s %s %s" % (a.capitalize(), rng.choice([">", "<", "vs."]), b)
    if kind < 0.70:
        a, b = rng.sample(vocab.GROUPS, 2)
        return "%s %s %s" % (a, rng.choice([">", "<"]), b)
    if kind < 0.85:
        return rng.choice(vocab.EFFECTS)
    return "%s in %s" % (rng.choice(vocab.EFFECTS), rng.choice(vocab.GROUPS))


@dataclass
class _Layout:
    """Which columns exist, in order, and what each one holds."""

    columns: List[str]
    space: Optional[str]
    space_in_table: bool
    stat_kind: Optional[str]
    stat_in_header: bool
    stat_in_footnote: bool
    measure: Optional[str]
    axes_named: bool
    side_column: bool
    packed: bool = False            # x, y and z share one cell
    packed_brackets: bool = False   # and that cell is `(-42, -55, -18)`
    header_marked: bool = True      # the header row is written in <th>

    def index(self, role: str) -> Optional[int]:
        return self.columns.index(role) if role in self.columns else None


def _layout(rng: random.Random, w: Weights) -> _Layout:
    space = rng.choice(["MNI", "TAL", "MNI"])
    roll = rng.random()
    space_in_table = roll < w.space_in_table
    if roll >= w.space_in_table + w.space_in_context_only:
        space = None                        # stated nowhere; the target is null

    stat_kind = None
    if rng.random() < w.statistic_printed:
        stat_kind = "T" if rng.random() < w.statistic_is_t else \
            rng.choice(["Z", "Z", "F", "P", "R", "B"])
    stat_in_header = stat_kind is not None and rng.random() < (
        w.statistic_in_header / max(w.statistic_printed, 1e-9))
    # Named only in a footnote. Decided here, not in `_context`, because a row
    # may only claim a statistic TYPE the document states somewhere.
    stat_in_footnote = (stat_kind is not None and not stat_in_header
                        and rng.random() < (w.statistic_in_footnote
                                            / max(1e-9, w.statistic_printed
                                                  - w.statistic_in_header)))

    # A numeric column before x is the lever that broke v14's minus-sign rule,
    # so it is drawn first and the cluster-size column is added to serve it. The
    # weight was previously conditional on having one, which quietly halved the
    # rate to 0.62 * 0.74.
    lead_extent = rng.random() < w.numeric_column_before_x
    measure = (rng.choice(["voxels", "voxels", "mm^3"])
               if lead_extent or rng.random() < 0.40 else None)

    # Nearly a quarter of real coordinate tables put the whole triple in one
    # cell. The generator never produced one, so a model trained on it had no
    # reason to look inside a cell for three numbers.
    packed = rng.random() < w.coordinates_packed_in_one_cell

    cols = ["region"]
    if rng.random() < 0.30:
        cols.append("side")
    if lead_extent:
        cols.append("extent")
    cols += ["xyz"] if packed else ["x", "y", "z"]
    if stat_kind:
        cols.append("stat")
    if measure is not None and not lead_extent:
        cols.append("extent")
    return _Layout(columns=cols, space=space, space_in_table=space_in_table,
                   stat_kind=stat_kind, stat_in_header=stat_in_header,
                   stat_in_footnote=stat_in_footnote,
                   measure=measure,
                   # A packed column has no axis columns to name.
                   axes_named=(not packed) and rng.random() < w.axes_named_in_header,
                   side_column="side" in cols,
                   packed=packed,
                   packed_brackets=packed and rng.random() < w.packed_in_brackets,
                   # 7% of real coordinate tables mark no header at all -- the
                   # header is written in <td> and only its position says what
                   # it is.
                   header_marked=rng.random() >= w.header_row_unmarked)


def _header(rng: random.Random, lay: _Layout, w: Weights) -> List[List[Cell]]:
    """One or two header rows, with the coordinate columns grouped or not."""
    xi = lay.index("xyz") if lay.packed else lay.index("x")
    top: List[Cell] = []
    second: List[Cell] = []
    two_rows = lay.axes_named and (lay.space_in_table or rng.random() < 0.5)

    for role in lay.columns[:xi]:
        label = {"region": rng.choice(vocab.REGION_HEADERS),
                 "side": rng.choice(vocab.SIDE_HEADERS),
                 "extent": rng.choice(vocab.EXTENT_HEADERS[lay.measure or "voxels"]),
                 }[role]
        top.append(Cell(label, header=True, rowspan=2 if two_rows else 1))

    if lay.packed:
        label = (rng.choice(vocab.SPACE_HEADERS[lay.space])
                 if lay.space_in_table and lay.space
                 else rng.choice(vocab.BARE_COORD_HEADERS))
        if rng.random() < 0.5:
            label += rng.choice([" (x, y, z)", " x, y, z", " (mm)"])
        top.append(Cell(label, header=True))
    elif two_rows:
        group = rng.choice(vocab.SPACE_HEADERS[lay.space]) if (
            lay.space_in_table and lay.space) else rng.choice(vocab.BARE_COORD_HEADERS)
        top.append(Cell(group, header=True, colspan=3))
        axes = rng.choice(vocab.AXIS_HEADERS)
        second = [Cell(a, header=True) for a in axes]
    elif lay.axes_named:
        prefix = ""
        if lay.space_in_table and lay.space:
            prefix = rng.choice(["MNI ", "Talairach "] if lay.space == "TAL" else ["MNI "])
        for a in rng.choice(vocab.AXIS_HEADERS):
            top.append(Cell(prefix + a, header=True))
    else:
        # The axes are not named. A reader must find the triple another way,
        # which is 55% of real tables.
        label = rng.choice(vocab.BARE_COORD_HEADERS)
        if lay.space_in_table and lay.space:
            label = rng.choice(vocab.SPACE_HEADERS[lay.space])
        top.append(Cell(label, header=True, colspan=3))

    for role in lay.columns[xi + (1 if lay.packed else 3):]:
        label = (rng.choice(vocab.STAT_HEADERS[lay.stat_kind])
                 if role == "stat" and lay.stat_in_header
                 else "Value" if role == "stat"
                 else rng.choice(vocab.EXTENT_HEADERS[lay.measure or "voxels"]))
        top.append(Cell(label, header=True, rowspan=2 if two_rows else 1))

    # A header carries a footnote marker as often as a value does -- `X (mm)d`,
    # `Region (Brodmann)a` -- and an axis name wearing one was invisible to
    # the reader.
    #
    # A bare letter only after a bracket or a digit, which is where papers put
    # one and the only place a reader can tell it apart from the word it
    # follows. Gluing one to a letter makes `MNI` into `MNIc`, which nothing
    # can undo, and the space the target asserts stops being stated anywhere.
    for row in ([top] + ([second] if second else [])):
        for cell in row:
            text = cell.text.strip()
            if not text or rng.random() >= w.header_markers:
                continue
            if text[-1] in ")]" or text[-1].isdigit():
                cell.text = text + rng.choice(MARKERS)
            else:
                cell.text = text + rng.choice(("*", "**", "\u2020", "\u2021"))

    rows = [top]
    if second:
        rows.append(second)
    if not lay.header_marked:
        # Written in <td>. Nothing about the text changes; only the tag, which
        # is the whole difficulty -- a reader looking for marked headers finds
        # none and gives up on a table it could otherwise read.
        rows = [[Cell(c.text, header=False, colspan=c.colspan, rowspan=c.rowspan)
                 for c in row] for row in rows]
    return rows


def _data_row(rng: random.Random, lay: _Layout, region: vocab.Region,
              side: Optional[str], w: Weights, zero_x: bool,
              label_side: Optional[str] = None) -> Tuple[List[Cell], TruthPoint]:
    x, y, z = _coord(rng, region.sided(side), w, zero_x=zero_x)
    stat_val = _stat_value(rng, lay.stat_kind) if lay.stat_kind else None
    extent = float(rng.choice([8, 14, 22, 31, 48, 76, 120, 210, 380, 640, 1180, 2400]))

    shown = label_side if label_side is not None else side
    name = region.name[0].upper() + region.name[1:]
    if shown and not region.midline and not lay.side_column:
        name = "%s %s" % (rng.choice([shown, {"L": "Left", "R": "Right"}[shown]]), name)

    cells: List[Cell] = []
    for role in lay.columns:
        if role == "region":
            cells.append(Cell(name))
        elif role == "side":
            cells.append(Cell(shown or "B"))
        elif role == "xyz":
            trio = "%s, %s, %s" % (_fmt(x), _fmt(y), _fmt(z))
            cells.append(Cell("(%s)" % trio if lay.packed_brackets else trio))
        elif role == "x":
            cells.append(Cell(_fmt(x)))
        elif role == "y":
            cells.append(Cell(_fmt(y)))
        elif role == "z":
            cells.append(Cell(_fmt(z)))
        elif role == "stat":
            cells.append(Cell(_fmt(stat_val)))
        elif role == "extent":
            cells.append(Cell(_fmt(extent)))
    # The number is printed, so the value is assertable. The KIND is only
    # assertable where the document names it -- a target claiming "T" on a
    # table whose header says "Value" and whose footnotes are silent teaches a
    # model to invent statistic types.
    named = lay.stat_in_header or lay.stat_in_footnote
    truth = TruthPoint(x, y, z,
                       statistic_type=lay.stat_kind if named else None,
                       statistic_value=stat_val,
                       extent=extent if lay.measure is not None else None)
    return cells, truth


#: What a paper hangs off a value to point at a footnote.
MARKERS = ("*", "**", "***", "a", "b", "c", "\u2020", "\u2021")
MISSING = ("-", "\u2013", "n.s.", "N/A", "ns")


def _rough_up(rng: random.Random, cells: List[Cell], lay: "_Layout",
              w: Weights, truth: TruthPoint) -> None:
    """Make one row as untidy as a real one, without making the target a lie.

    A marker is cosmetic and the number under it is unchanged, so the target
    keeps it. A blanked or dashed cell is not: whatever it held is no longer
    stated, and a target still asserting it would teach the model to read a
    value that is not there.

    Coordinate columns are left alone. A row missing one of its coordinates is
    a different case, handled where the row is built, because the whole point
    has to leave the target with it.
    """
    axis_roles = {"x", "y", "z", "xyz"}
    # A row continuing a rowspan has had its first cell trimmed, so cell i is
    # column i + offset. Reading the roles straight off `lay.columns` blanked
    # a coordinate while nulling the extent, which both lost the point and
    # left the target asserting a number no longer printed.
    offset = len(lay.columns) - len(cells)
    for i, cell in enumerate(cells):
        role = lay.columns[i + offset]
        if role in axis_roles:
            continue
        roll = rng.random()
        if roll < w.blank_cells:
            cell.text = ""
        elif roll < w.blank_cells + w.dash_for_missing and role in ("stat", "extent"):
            # A dash stands where a number was expected. A paper does not write
            # `n.s.` where a region name goes.
            cell.text = rng.choice(MISSING)
        elif rng.random() < w.footnote_markers and cell.text.strip():
            cell.text += rng.choice(MARKERS)
            continue                       # cosmetic: the target is unchanged
        else:
            continue
        # The cell no longer states what it held.
        if role == "stat":
            truth.statistic_type = None
            truth.statistic_value = None
        elif role == "extent":
            truth.extent = None


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return ""
    if float(v).is_integer():
        return str(int(v))
    return ("%f" % v).rstrip("0").rstrip(".")


def _divider(rng: random.Random, text: str, width: int, w: Weights) -> List[Cell]:
    """A section banner, full width or as a partial span with the row left empty."""
    if rng.random() < w.partial_span_divider and width > 3:
        span = rng.randint(2, max(2, width // 2))
        return [Cell(text, colspan=span)] + [Cell("") for _ in range(width - span)]
    return [Cell(text, colspan=width)]


def _column_grouped(rng: random.Random, w: Weights) -> Table:
    """A table whose analyses are column blocks, not row blocks.

    Two or three contrasts share every row, and only the spanning header above
    a column says which contrast a number belongs to. A banner cannot help
    here, and neither can reading down: the grouping is sideways.

    A region that one contrast found and another did not leaves blanks under
    that contrast, which is how the real ones look and is why the target has
    different counts per analysis.
    """
    n_groups = 2 if rng.random() < 0.65 else 3
    stat_kind = "T" if rng.random() < w.statistic_is_t else rng.choice(["Z", "F"])
    per_group = ["x", "y", "z"] + (["stat"] if rng.random() < w.statistic_printed else [])
    # Stated in the caption, because these headers have no room for it: the
    # top row carries the contrast names and the second the axes. A space the
    # target asserts and the document never states is the defect this
    # generator exists to avoid.
    space = rng.choice(["MNI", "TAL"]) if rng.random() < (
        w.space_in_table + w.space_in_context_only) else None

    names = []
    while len(names) < n_groups:
        name = _analysis_name(rng)
        if name not in names:
            names.append(name)

    axes = rng.choice(vocab.AXIS_HEADERS)
    # One label for the statistic, not one per group: a table does not head the
    # same quantity `z` under one contrast and `Z value` under the next.
    stat_header = rng.choice(vocab.STAT_HEADERS[stat_kind])
    top = [Cell(rng.choice(vocab.REGION_HEADERS), header=True, rowspan=2)]
    second: List[Cell] = []
    for name in names:
        top.append(Cell(name, header=True, colspan=len(per_group)))
        for role in per_group:
            second.append(Cell(axes["xyz".index(role)] if role in "xyz"
                               else stat_header, header=True))
    grid = Grid()
    grid.add(top)
    grid.add(second)

    truth = Truth(space=space,
                  analyses=[TruthAnalysis(name=n, measure=None) for n in names])
    lobe = rng.choice(list(vocab.LOBE_SECTIONS))
    pool = [r for r in vocab.REGIONS if r.lobe == lobe] or list(vocab.REGIONS)
    bag = list(pool)
    rng.shuffle(bag)
    for _ in range(rng.randint(4, 10)):
        if not bag:
            bag = list(pool)
            rng.shuffle(bag)
        region = bag.pop()
        side = None if region.midline else rng.choice(["L", "R"])
        name = region.name[0].upper() + region.name[1:]
        if side:
            name = "%s %s" % (side, name)
        cells = [Cell(name)]
        # At least one group has to report this region, or the row is empty.
        found = [rng.random() < 0.7 for _ in names]
        if not any(found):
            found[rng.randrange(len(found))] = True
        for gi, present in enumerate(found):
            if not present:
                cells.extend(Cell("") for _ in per_group)
                continue
            x, y, z = _coord(rng, region.sided(side), w)
            value = _stat_value(rng, stat_kind)
            for role in per_group:
                cells.append(Cell(_fmt({"x": x, "y": y, "z": z}[role])
                                  if role in "xyz" else _fmt(value)))
            truth.analyses[gi].points.append(TruthPoint(
                x, y, z, statistic_type=stat_kind if "stat" in per_group else None,
                statistic_value=value if "stat" in per_group else None))
        # As untidy as any other table: a marker on the region name is
        # cosmetic and the target is unchanged by it.
        if rng.random() < w.footnote_markers and cells[0].text:
            cells[0].text += rng.choice(MARKERS)
        grid.add(cells)

    caption = "Table %d. Regions activated in each contrast." % rng.randint(1, 6)
    if space:
        caption += " Coordinates are in %s space." % (
            "Talairach" if space == "TAL" else "MNI")
    elif rng.random() >= w.caption_present:
        caption = ""
    footer = ""
    if rng.random() < w.footer_present:
        footer = rng.choice([
            "L: left; R: right.",
            "The statistical threshold was set at p<0.05 (FWE-corrected).",
            "Blank cells indicate the region did not survive threshold in that contrast.",
        ])[:w.footer_chars]
    return Table(grid=grid, truth=truth, caption=caption, footer=footer,
                 notes={"layout": ["region"] + per_group * n_groups,
                        "column_grouped": True, "groups": n_groups,
                        "dividers": 0, "space_in_table": False,
                        "axes_named": True, "identical_neighbours": False,
                        "zero_x_rows": 0})


def build(seed: int = 0, weights: Weights = DEFAULT) -> Table:
    """One synthetic table and the target a faithful reader should produce."""
    rng = random.Random(seed)
    w = weights
    if rng.random() < w.analyses_in_column_groups:
        return _column_grouped(rng, w)
    lay = _layout(rng, w)
    header = _header(rng, lay, w)
    width = sum(c.colspan for c in header[0])

    n_analyses = 6 + rng.randint(0, 3) if rng.random() < w.six_or_more_analyses \
        else rng.randint(1, 4)
    truth = Truth(space=lay.space)
    grid = Grid()
    for row in header:
        grid.add(row)

    notes = {"layout": list(lay.columns), "axes_named": lay.axes_named,
             "space_in_table": lay.space_in_table, "dividers": 0,
             "identical_neighbours": False, "zero_x_rows": 0}

    # Adjacent analyses with identical structure: the banner is then the only
    # thing telling them apart.
    identical = n_analyses > 1 and rng.random() < w.structurally_identical_neighbours
    notes["identical_neighbours"] = identical

    unbannered = None
    for ai in range(n_analyses):
        name = _analysis_name(rng)
        analysis = TruthAnalysis(name=name, measure=lay.measure)
        use_banner = rng.random() < min(1.0, w.banners_per_table / max(n_analyses, 1)) or ai > 0
        if use_banner:
            grid.add(_divider(rng, name, width, w))
            notes["dividers"] += 1
        else:
            # No banner, so the table does not say this analysis's name. The
            # caption has to, or the target asserts something no part of the
            # document states -- the same defect as claiming a statistic the
            # table never printed, and it teaches the same habit.
            unbannered = name

        # Real coordinate tables carry 14.5 points; a generator that makes
        # 10.8 is training on the easy end of the corpus.
        n_points = rng.randint(3, 7) if identical else rng.randint(1, 9)
        lobe = rng.choice(list(vocab.LOBE_SECTIONS))
        pool = [r for r in vocab.REGIONS if r.lobe == lobe] or list(vocab.REGIONS)
        # Without replacement: a paper does not list the same structure five
        # times in one contrast, and repeating it hands the model a cue that
        # adjacent rows belong together for the wrong reason.
        bag = list(pool)
        rng.shuffle(bag)
        # An anatomical sub-heading inside one analysis: a spanning row that is
        # NOT a boundary. Real tables do this; a model that splits on every
        # banner gets it wrong.
        if rng.random() < 0.25 and n_points >= 3:
            grid.add(_divider(rng, rng.choice(vocab.LOBE_SECTIONS[lobe]), width, w))
            notes["dividers"] += 1
            notes["subheadings"] = notes.get("subheadings", 0) + 1

        rows_for_label = 0
        held: Optional[Tuple[vocab.Region, Optional[str]]] = None
        for pi in range(n_points):
            if rows_for_label > 0 and held is not None:
                # Rows covered by a region rowspan are sub-peaks of that region,
                # so they keep its identity. Drawing a fresh region for them
                # made the inherited label describe the wrong structure.
                region, side = held
            else:
                if not bag:
                    bag = list(pool)
                    rng.shuffle(bag)
                region = bag.pop()
                side = None if region.midline else rng.choice(["L", "R"])
            # Papers do mislabel a hemisphere occasionally. Flip the name only:
            # flipping `side` moved the label and the coordinate box together,
            # so the two never actually disagreed.
            label_side = side
            if side and rng.random() > w.laterality_agrees:
                label_side = {"L": "R", "R": "L"}[side]
            zero_x = rng.random() < w.zero_coordinate
            if zero_x:
                notes["zero_x_rows"] += 1
            cells, point = _data_row(rng, lay, region, side, w, zero_x,
                                     label_side=label_side)
            # A rowspan in column 0 carries the label down: the load-bearing form.
            if rows_for_label == 0 and n_points - pi >= 2 and \
                    rng.random() < w.rowspan_in_first_column:
                rows_for_label = min(n_points - pi, rng.randint(2, 3))
                cells[0] = Cell(cells[0].text, rowspan=rows_for_label)
                held = (region, side)
            elif rows_for_label > 0:
                cells = cells[1:]                     # covered by the span above
            if rows_for_label > 0:
                rows_for_label -= 1
                if rows_for_label == 0:
                    held = None
            # A row naming a region and stating no coordinates for it: a
            # sub-heading that is not a banner. Its point leaves the target
            # with it, so a model that invents one is wrong.
            # Only where the row still has its own region cell. A row covered
            # by a rowspan has had that cell trimmed, so blanking the rest
            # leaves a row with nothing in it -- and an all-empty row vanishes
            # when the grid renders, which shifts every following row one
            # column along and loses its coordinates.
            if (len(cells) == len(lay.columns) and n_points - pi > 1
                    and rng.random() < w.rows_without_coordinates):
                for i, role in enumerate(lay.columns):
                    if role in ("x", "y", "z", "xyz", "stat", "extent"):
                        cells[i].text = ""
                grid.add(cells)
                continue

            _rough_up(rng, cells, lay, w, point)
            grid.add(cells)
            analysis.points.append(point)
        truth.analyses.append(analysis)

    caption, footer = _context(rng, lay, w, names_in_caption=unbannered)
    return Table(grid=grid, truth=truth, caption=caption, footer=footer, notes=notes)


def _context(rng: random.Random, lay: _Layout, w: Weights,
             names_in_caption: Optional[str] = None) -> Tuple[str, str]:
    """Caption and footer, kept consistent with what the target claims.

    The space is stated in exactly the place the layout says it is. If the
    layout put it in the context rather than the table, the context must
    actually say it -- leaving it to a coin flip produced targets asserting a
    space no part of the document named, which is the defect `space_rule` was
    written to remove in the first place.
    """
    say_space = lay.space is not None and not lay.space_in_table
    space_word = "Talairach" if lay.space == "TAL" else "MNI"
    in_caption = say_space and rng.random() < 0.65

    caption = ""
    # An analysis with no banner is named in the caption or nowhere, so a
    # caption is not optional when there is one to name.
    if names_in_caption or rng.random() < w.caption_present:
        caption = "Table %d. %s" % (
            rng.randint(1, 6),
            rng.choice(["Regions showing significant activation",
                        "Peak activations for each contrast",
                        "Clusters surviving whole-brain correction",
                        "Local maxima of significant clusters"]))
        caption += (" for %s." % names_in_caption) if names_in_caption else "."
        if in_caption:
            caption += " Coordinates are in %s space." % space_word
    else:
        in_caption = False          # no caption to carry it

    # Not every table carries a footnote: 47% of real ones do. A generator
    # that always writes one teaches the model to expect it.
    must_speak = say_space or lay.stat_in_footnote
    if not must_speak and rng.random() >= w.footer_present:
        return caption, ""

    parts = []
    if say_space and not in_caption:
        parts.append("Coordinates are reported in %s space." % space_word)
    used = set()
    while len(" ".join(parts)) < w.footer_chars and len(used) < len(vocab.FOOTER_PARTS):
        candidate = rng.choice(vocab.FOOTER_PARTS)
        if "{space}" in candidate:
            continue                # the space sentence is placed above, or not at all
        # Against the template, not the formatted sentence: "Cluster size
        # threshold was {k} voxels" formats differently each time, so checking
        # the output let the same sentence in twice with different numbers.
        if candidate in used:
            continue
        used.add(candidate)
        # A footnote may only name the statistic where the layout says the
        # statistic is named. Otherwise the document states a type the target
        # calls null -- the same defect as the space one above, and a reader
        # that believes the document is then marked wrong for being right.
        if "{stat}" in candidate and not lay.stat_in_footnote:
            continue
        parts.append(candidate.format(k=rng.choice([5, 10, 20]),
                                      space=space_word,
                                      stat=lay.stat_kind or "T"))
    if lay.stat_in_footnote:
        parts.append("Values shown are %s statistics." % lay.stat_kind)
    return caption, " ".join(parts)
