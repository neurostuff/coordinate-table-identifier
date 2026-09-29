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


def _round(rng: random.Random, value: float, w: Weights) -> float:
    if rng.random() < w.non_integer_coordinates:
        return round(value + rng.choice([-0.5, 0.5, 0.2, -0.2]), 1)
    # Papers report on a 2mm grid far more often than not, so an even
    # coordinate is not evidence of anything.
    return float(int(value) // 2 * 2)


def _coord(rng: random.Random, region: vocab.Region, w: Weights,
           zero_x: bool = False) -> Tuple[float, float, float]:
    x = 0.0 if zero_x else _round(rng, rng.uniform(*region.x), w)
    y = _round(rng, rng.uniform(*region.y), w)
    z = _round(rng, rng.uniform(*region.z), w)
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

    cols = ["region"]
    if rng.random() < 0.30:
        cols.append("side")
    if lead_extent:
        cols.append("extent")
    cols += ["x", "y", "z"]
    if stat_kind:
        cols.append("stat")
    if measure is not None and not lead_extent:
        cols.append("extent")
    return _Layout(columns=cols, space=space, space_in_table=space_in_table,
                   stat_kind=stat_kind, stat_in_header=stat_in_header,
                   stat_in_footnote=stat_in_footnote,
                   measure=measure,
                   axes_named=rng.random() < w.axes_named_in_header,
                   side_column="side" in cols)


def _header(rng: random.Random, lay: _Layout, w: Weights) -> List[List[Cell]]:
    """One or two header rows, with the coordinate columns grouped or not."""
    xi = lay.index("x")
    top: List[Cell] = []
    second: List[Cell] = []
    two_rows = lay.axes_named and (lay.space_in_table or rng.random() < 0.5)

    for role in lay.columns[:xi]:
        label = {"region": rng.choice(vocab.REGION_HEADERS),
                 "side": rng.choice(vocab.SIDE_HEADERS),
                 "extent": rng.choice(vocab.EXTENT_HEADERS[lay.measure or "voxels"]),
                 }[role]
        top.append(Cell(label, header=True, rowspan=2 if two_rows else 1))

    if two_rows:
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

    for role in lay.columns[xi + 3:]:
        label = (rng.choice(vocab.STAT_HEADERS[lay.stat_kind])
                 if role == "stat" and lay.stat_in_header
                 else "Value" if role == "stat"
                 else rng.choice(vocab.EXTENT_HEADERS[lay.measure or "voxels"]))
        top.append(Cell(label, header=True, rowspan=2 if two_rows else 1))

    rows = [top]
    if second:
        rows.append(second)
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


def build(seed: int = 0, weights: Weights = DEFAULT) -> Table:
    """One synthetic table and the target a faithful reader should produce."""
    rng = random.Random(seed)
    w = weights
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

    for ai in range(n_analyses):
        name = _analysis_name(rng)
        analysis = TruthAnalysis(name=name, measure=lay.measure)
        use_banner = rng.random() < min(1.0, w.banners_per_table / max(n_analyses, 1)) or ai > 0
        if use_banner:
            grid.add(_divider(rng, name, width, w))
            notes["dividers"] += 1

        n_points = rng.randint(2, 5) if identical else rng.randint(1, 6)
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
            grid.add(cells)
            analysis.points.append(point)
        truth.analyses.append(analysis)

    caption, footer = _context(rng, lay, w)
    return Table(grid=grid, truth=truth, caption=caption, footer=footer, notes=notes)


def _context(rng: random.Random, lay: _Layout, w: Weights) -> Tuple[str, str]:
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
    if rng.random() < w.caption_present:
        caption = "Table %d. %s." % (
            rng.randint(1, 6),
            rng.choice(["Regions showing significant activation",
                        "Peak activations for each contrast",
                        "Clusters surviving whole-brain correction",
                        "Local maxima of significant clusters"]))
        if in_caption:
            caption += " Coordinates are in %s space." % space_word
    else:
        in_caption = False          # no caption to carry it

    parts = []
    if say_space and not in_caption:
        parts.append("Coordinates are reported in %s space." % space_word)
    while len(" ".join(parts)) < w.footer_chars and len(parts) < len(vocab.FOOTER_PARTS):
        candidate = rng.choice(vocab.FOOTER_PARTS)
        if "{space}" in candidate:
            continue                # the space sentence is placed above, or not at all
        if candidate in parts:
            continue
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
