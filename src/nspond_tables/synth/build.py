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
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..grid import Cell, Grid
from . import vocab
from .trainset import STATISTIC_PRIORITY
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


def _to_voxel(v: float, axis: int) -> float:
    """A millimetre coordinate as the voxel index a 1mm template would give.

    `92 | 132 | 96` under a header reading `Peak MNI`. Every value is positive
    and every one is outside a head, so the reader's bounds reject all three
    -- which is exactly why these arrive on the residual route.
    """
    origin = (90.0, 126.0, 72.0)[axis]
    return float(int(round(origin + v)))


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
    if kind in ("D", "G"):
        # A standardised mean difference. Signed, and rarely past about 2.
        return round(rng.uniform(0.2, 1.9) * rng.choice([1, -1]), 2)
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
    second_stat_kind: Optional[str]
    stat_in_header: bool
    stat_in_footnote: bool
    measure: Optional[str]
    axes_named: bool
    side_column: bool
    packed: bool = False            # x, y and z share one cell
    packed_form: str = "plain"      # and how that cell is written
    axis_names: Tuple[str, str, str] = ("x", "y", "z")
    header_marked: bool = True      # the header row is written in <th>
    voxel_indices: bool = False     # coordinates are voxel indices, not mm
    signs_spaced: bool = False      # `- 34` rather than `-34`
    coords_first: bool = False      # the triple precedes the region name
    coords_in_name: bool = False    # the region cell carries its own triple
    unmarked_span: bool = False     # one header cell over three columns

    def index(self, role: str) -> Optional[int]:
        return self.columns.index(role) if role in self.columns else None


def _packed_form(rng: random.Random, w: Weights) -> str:
    """Which way this table writes a packed triple. One way per table: a paper
    does not alternate between `(-42, -55, -18)` and `-42-55 -18`."""
    roll = rng.random()
    for name, share in (("brackets", w.packed_in_brackets),
                        ("run_on", w.packed_signs_run_on),
                        ("semicolons", w.packed_semicolons),
                        ("after_statistic", w.packed_after_a_statistic),
                        ("with_note", w.packed_with_a_note)):
        if roll < share:
            return name
        roll -= share
    return "plain"


def _layout(rng: random.Random, w: Weights) -> _Layout:
    space = rng.choice(["MNI", "TAL", "MNI"])
    roll = rng.random()
    space_in_table = roll < w.space_in_table
    if roll >= w.space_in_table + w.space_in_context_only:
        space = None                        # stated nowhere; the target is null

    stat_kind = None
    if rng.random() < w.statistic_printed:
        stat_kind = "T" if rng.random() < w.statistic_is_t else \
            rng.choice(["Z", "Z", "F", "P", "R", "B", "D"])
    # A real table often prints a test statistic beside its significance
    # level -- `#t | #p(FWE)` -- and one of the two is what it reports.
    # Without a second column the generator never shows that shape, so the
    # priority order is learned from nothing and the reader is on its own.
    second_stat_kind = None
    if stat_kind and rng.random() < w.second_statistic_printed:
        others = [k for k in ("P", "P", "P", "P", "Z", "T", "R", "D", "G")
                  if k != stat_kind]
        second_stat_kind = rng.choice(others)
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

    # The reader reads nothing out of any of these, which is the point: they
    # are the residual gate's positive class, and it had 34 examples of it.
    voxel_indices = rng.random() < w.voxel_indices
    signs_spaced = rng.random() < w.signs_spaced
    coords_in_name = not packed and rng.random() < w.coordinates_in_the_region_name
    coords_first = (not packed and not coords_in_name
                    and rng.random() < w.coordinates_before_the_region)
    unmarked_span = (not packed and not coords_in_name and not coords_first
                     and rng.random() < w.span_without_a_marker)

    if coords_in_name:
        space_in_table = False

    cols = ["region"]
    if rng.random() < 0.30:
        cols.append("side")
    if lead_extent:
        cols.append("extent")
    if coords_in_name:
        # No column holds the triple: the region cell carries it, as
        # `Insula (-33, 21, 3)`. Everything else is what it was.
        pass
    elif coords_first:
        # `-46 | 14 | -4 | L. GTs | 22`: the triple opens the row and the
        # region is named after it.
        cols = ["x", "y", "z"] + cols
    else:
        cols += ["xyz"] if packed else ["x", "y", "z"]
    if stat_kind:
        cols.append("stat")
    if second_stat_kind:
        cols.append("stat2")
    if measure is not None and not lead_extent:
        cols.append("extent")
    axis_names = _axis_names(rng, w, space if space_in_table else None)
    return _Layout(columns=cols, voxel_indices=voxel_indices,
                   signs_spaced=signs_spaced, coords_first=coords_first,
                   coords_in_name=coords_in_name, unmarked_span=unmarked_span,
                   space=space, space_in_table=space_in_table,
                   stat_kind=stat_kind, second_stat_kind=second_stat_kind,
                   stat_in_header=stat_in_header,
                   stat_in_footnote=stat_in_footnote,
                   measure=measure,
                   # A packed column has no axis columns to name.
                   axes_named=(not packed) and rng.random() < w.axes_named_in_header,
                   side_column="side" in cols,
                   packed=packed,
                   packed_form=_packed_form(rng, w) if packed else "plain",
                   # 7% of real coordinate tables mark no header at all -- the
                   # header is written in <td> and only its position says what
                   # it is.
                   header_marked=rng.random() >= w.header_row_unmarked,
                   axis_names=axis_names)


def _axis_names(rng: random.Random, w: Weights,
                space: Optional[str] = None) -> Tuple[str, str, str]:
    """What this table calls its axes.

    `X coor` and `y coordinate` say which kind of coordinate the column holds;
    `R`, `A`, `S` name the anatomical directions instead. Both head real
    coordinate columns and neither existed here -- `X coor` is three headers
    in 6,897, and the one table that used it was lost entire.
    """
    roll = rng.random()
    if roll < w.ras_instead_of_xyz:
        return rng.choice(vocab.RAS_HEADERS)
    if roll < w.ras_instead_of_xyz + w.axis_names_its_kind:
        # An axis naming a space has to name the one the table is in. A span
        # reading `MNI coordinates` over axes reading `x (Talairach)` is a
        # table contradicting itself, which no paper does and no target can
        # describe.
        kinded = [k for k in vocab.AXIS_HEADERS_KINDED
                  if _names_space(k) in (None, space)]
        if kinded:
            return rng.choice(kinded)
    return rng.choice(vocab.AXIS_HEADERS)


def _names_space(axes: Tuple[str, str, str]) -> Optional[str]:
    joined = " ".join(axes).lower()
    if "talairach" in joined:
        return "TAL"
    if "mni" in joined:
        return "MNI"
    return None


def _header(rng: random.Random, lay: _Layout, w: Weights) -> List[List[Cell]]:
    """One or two header rows, with the coordinate columns grouped or not.

    Driven by the column list rather than by where the coordinates usually
    sit. They do not always sit there: a table can put the triple first and
    name the region after it, or carry the triple inside the region cell and
    have no coordinate column at all.
    """
    coord_roles = ("x", "y", "z", "xyz")
    first_coord = next((i for i, r in enumerate(lay.columns)
                        if r in coord_roles), None)
    top: List[Cell] = []
    second: List[Cell] = []
    two_rows = (first_coord is not None and lay.axes_named
                and not lay.packed and not lay.unmarked_span
                and (lay.space_in_table or rng.random() < 0.5))

    def plain(role: str) -> str:
        return {"region": rng.choice(vocab.REGION_HEADERS),
                "side": rng.choice(vocab.SIDE_HEADERS),
                "stat": rng.choice(vocab.STAT_HEADERS[lay.stat_kind or "Z"]),
                "stat2": rng.choice(
                    vocab.STAT_HEADERS[lay.second_stat_kind or "P"]),
                "extent": rng.choice(
                    vocab.EXTENT_HEADERS[lay.measure or "voxels"]),
                }[role]

    def grouped() -> str:
        return (rng.choice(vocab.SPACE_HEADERS[lay.space])
                if lay.space_in_table and lay.space
                else rng.choice(vocab.BARE_COORD_HEADERS))

    done_coords = False
    for role in lay.columns:
        if role not in coord_roles:
            top.append(Cell(plain(role), header=True,
                            rowspan=2 if two_rows else 1))
            continue
        if done_coords:
            continue
        done_coords = True
        if lay.packed:
            label = grouped()
            if rng.random() < 0.5:
                label += rng.choice([" (x, y, z)", " x, y, z", " (mm)"])
            top.append(Cell(label, header=True,
                            rowspan=2 if two_rows else 1))
        elif lay.unmarked_span:
            # One cell naming all three axes with nothing marking the span, so
            # the header is one short of the body and every row reads one
            # column to the left of where it belongs.
            top.append(Cell("%s (%s, %s, %s%s)"
                            % (grouped(), lay.axis_names[0], lay.axis_names[1],
                               lay.axis_names[2],
                               " mm" if rng.random() < 0.5 else ""),
                            header=True))
        elif two_rows:
            top.append(Cell(grouped(), header=True, colspan=3))
            second = [Cell(a, header=True) for a in lay.axis_names]
        elif lay.axes_named:
            prefix = ""
            if lay.space_in_table and lay.space:
                prefix = rng.choice(["MNI ", "Talairach "]
                                    if lay.space == "TAL" else ["MNI "])
            top.extend(Cell(prefix + a, header=True) for a in lay.axis_names)
        else:
            # The axes are not named. A reader must find the triple another
            # way, which is 55% of real tables.
            label = (rng.choice(vocab.SPACE_HEADERS[lay.space])
                     if lay.space_in_table and lay.space
                     else rng.choice(vocab.BARE_COORD_HEADERS))
            top.append(Cell(label, header=True, colspan=3))

    if not lay.header_marked:
        for cell in top + second:
            cell.header = False
    rows = [top]
    if second:
        rows.append(second)
    return rows

def _data_row(rng: random.Random, lay: _Layout, region: vocab.Region,
              side: Optional[str], w: Weights, zero_x: bool,
              label_side: Optional[str] = None) -> Tuple[List[Cell], TruthPoint]:
    x, y, z = _coord(rng, region.sided(side), w, zero_x=zero_x)
    stat_val = _stat_value(rng, lay.stat_kind) if lay.stat_kind else None
    stat2_val = (_stat_value(rng, lay.second_stat_kind)
                 if lay.second_stat_kind else None)
    extent = float(rng.choice([8, 14, 22, 31, 48, 76, 120, 210, 380, 640, 1180, 2400]))

    shown = label_side if label_side is not None else side
    name = region.name[0].upper() + region.name[1:]
    if shown and not region.midline and not lay.side_column:
        name = "%s %s" % (rng.choice([shown, {"L": "Left", "R": "Right"}[shown]]), name)

    shown_x, shown_y, shown_z = (
        (_to_voxel(x, 0), _to_voxel(y, 1), _to_voxel(z, 2)) if lay.voxel_indices
        else (x, y, z))
    if lay.coords_in_name:
        name = "%s (%s, %s, %s)" % (name, _fmt(shown_x), _fmt(shown_y),
                                    _fmt(shown_z))

    cells: List[Cell] = []
    for role in lay.columns:
        if role == "region":
            cells.append(Cell(name))
        elif role == "side":
            cells.append(Cell(shown or "B"))
        elif role == "xyz":
            cells.append(Cell(_packed(rng, shown_x, shown_y, shown_z, lay, w,
                                      stat_val)))
        elif role == "x":
            cells.append(Cell(_fmt(shown_x)))
        elif role == "y":
            cells.append(Cell(_fmt(shown_y)))
        elif role == "z":
            cells.append(Cell(_fmt(shown_z)))
        elif role == "stat":
            cells.append(Cell(_fmt(stat_val)))
        elif role == "stat2":
            cells.append(Cell(_fmt(stat2_val)))
        elif role == "extent":
            cells.append(Cell(_fmt(extent)))
    # The number is printed, so the value is assertable. The KIND is only
    # assertable where the document names it -- a target claiming "T" on a
    # table whose header says "Value" and whose footnotes are silent teaches a
    # model to invent statistic types.
    named = lay.stat_in_header or lay.stat_in_footnote
    # The target states what the table prints. A voxel index is the
    # coordinate this table reports, and claiming the millimetres it does not
    # print would be inventing a normalisation the document never states.
    reported, reported_val = lay.stat_kind, stat_val
    if lay.second_stat_kind and STATISTIC_PRIORITY.index(lay.second_stat_kind) \
            < STATISTIC_PRIORITY.index(lay.stat_kind):
        reported, reported_val = lay.second_stat_kind, stat2_val
    truth = TruthPoint(shown_x, shown_y, shown_z,
                       statistic_type=reported if named else None,
                       statistic_value=reported_val,
                       extent=extent if lay.measure is not None else None)
    return cells, truth


#: How a paper says a contrast found nothing. Read off real tables: the row
#: under the banner, or the banner itself, carries one of these.
NOTHING_FOUND = (
    "no significant results", "No significant activation",
    "No suprathreshold voxels", "No clusters reach threshold",
    "n.s.", "No significant clusters survived correction",
    "No significant differences were observed", "-",
)

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
        before = cell.text
        roll = rng.random()
        if roll < w.blank_cells:
            cell.text = ""
        elif roll < w.blank_cells + w.dash_for_missing and role in ("stat", "stat2", "extent"):
            # A dash stands where a number was expected. A paper does not write
            # `n.s.` where a region name goes.
            cell.text = rng.choice(MISSING)
        elif rng.random() < w.footnote_markers and cell.text.strip():
            cell.text += rng.choice(MARKERS)
            continue                       # cosmetic: the target is unchanged
        else:
            continue
        # The cell no longer states what it held.
        #
        # Which statistic column the target quotes is decided by priority, not
        # by position, so a table printing `#t | #p` has a target reading the
        # t and blanking the p must leave it alone. The cell that was blanked
        # is the one whose number the target holds, or it is not.
        if role in ("stat", "stat2"):
            if truth.statistic_value is not None and before == _fmt(truth.statistic_value):
                truth.statistic_type = None
                truth.statistic_value = None
        elif role == "extent":
            truth.extent = None


#: What a note after a triple says: the cytoarchitectonic area or the Brodmann
#: area, never part of the coordinate.
NOTES = ("OP1", "OP4", "BA 40", "BA 6", "BA 44", "hOc4lp", "FG3", "TE 1.0",
         "PGa", "V1", "V2")


def _packed(rng: random.Random, x: float, y: float, z: float, lay: "_Layout",
            w: Weights, stat: Optional[float]) -> str:
    """One cell holding a whole triple, written the way a paper writes it.

    Every form here was read off a real table. `-10-42 16` runs the signs
    together, `9.91 [3, 15, 51]` puts the statistic first and the peak in
    brackets, `-56 -30 28 (OP1)` names the area afterwards. A reader that
    only knew `-42, -55, -18` lost all of them.
    """
    form = lay.packed_form
    a, b, c = _fmt(x), _fmt(y), _fmt(z)
    if form == "brackets":
        return "(%s, %s, %s)" % (a, b, c)
    if form == "run_on":
        # A minus straight after a digit, which is how a paper saves a space.
        return "%s%s %s" % (a, b if b.startswith("-") else " " + b, c)
    if form == "semicolons":
        return "%s; %s; %s" % (a, b, c)
    if form == "after_statistic":
        value = _fmt(stat) if stat is not None else _fmt(round(rng.uniform(2, 12), 2))
        return "%s %s%s, %s, %s%s" % (value, rng.choice("[("), a, b, c,
                                      rng.choice("])"))
    if form == "with_note":
        return "%s %s %s (%s)" % (a, b, c, rng.choice(NOTES))
    return "%s, %s, %s" % (a, b, c)


def _space_the_sign(text: str) -> str:
    """`-34` -> `- 34`. Several publishers' HTML arrives this way, and the
    reader sees a dash and a number rather than a negative coordinate."""
    return re.sub(r"(?<![\d.])-(\d)", r"- \1", text)


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


def _bilateral(cells: List[Cell], lay: "_Layout", point: TruthPoint
               ) -> Optional[TruthPoint]:
    """Write one row as `\u00b130 -80 6`, standing for both hemispheres.

    The row states two peaks and prints one number, so a reader taking cells
    at face value gets a single point with a sign it cannot resolve. Only a
    model reading `\u00b1` for what it means recovers both, which is why this is
    generated: the corpus has them and nothing else here teaches it.

    Returns the mirrored point, or None where the row cannot carry the form --
    a packed triple, a row covered by a rowspan, or a midline peak, where
    `\u00b10` would say nothing.
    """
    if "x" not in lay.columns or len(cells) != len(lay.columns):
        return None
    xi = lay.columns.index("x")
    if abs(point.x) < 1:
        return None
    cells[xi].text = "\u00b1%s" % _fmt(abs(point.x))
    for i, role in enumerate(lay.columns):
        if role == "side":
            cells[i].text = "B"
        elif role == "region":
            text = cells[i].text
            for prefix in ("Left ", "Right ", "L ", "R "):
                if text.startswith(prefix):
                    text = text[len(prefix):]
                    break
            cells[i].text = "Bilateral " + text[0].lower() + text[1:]
    point.x = abs(point.x)
    return TruthPoint(-abs(point.x), point.y, point.z,
                      statistic_type=point.statistic_type,
                      statistic_value=point.statistic_value,
                      extent=point.extent)


class _PackedAs:
    """`_packed` reads one field off a layout, and the column-grouped builder
    has no layout to give it."""

    def __init__(self, packed_form: str) -> None:
        self.packed_form = packed_form


def _plain_footer(rng: random.Random, w: Weights) -> str:
    """A footnote for the layouts that do not build one of their own."""
    if rng.random() >= w.footer_present:
        return ""
    return rng.choice([
        "L: left; R: right.",
        "The statistical threshold was set at p<0.05 (FWE-corrected).",
        "Cluster extents are reported in voxels.",
    ])[:w.footer_chars]


def _transposed(rng: random.Random, w: Weights) -> Table:
    """The axes are rows and the peaks are columns.

    `z | 4 | 4 | 2 | -8 | -22` down the side with a column per cluster. No
    reader here follows it, and the only way a model will ever see one is if
    this makes them.
    """
    lobe = rng.choice(list(vocab.LOBE_SECTIONS))
    pool = [r for r in vocab.REGIONS if r.lobe == lobe] or list(vocab.REGIONS)
    bag = list(pool); rng.shuffle(bag)
    # Without replacement, and the table is narrower than the lobe rather than
    # refilling: two columns headed `Vermis` in one contrast is not something
    # a paper prints, and it hands the model a cue for the wrong reason.
    n = min(rng.randint(3, 7), len(bag))
    regions = []
    pts = []
    for _ in range(n):
        region = bag.pop()
        side = None if region.midline else rng.choice(["L", "R"])
        x, y, z = _coord(rng, region.sided(side), w)
        name = region.name[0].upper() + region.name[1:]
        regions.append("%s %s" % (side, name) if side else name)
        pts.append((x, y, z))
    space = rng.choice(["MNI", "TAL"])
    grid = Grid()
    grid.add([Cell("", header=True)] + [Cell(r, header=True) for r in regions])
    axes = rng.choice(vocab.AXIS_HEADERS)
    for i, axis in enumerate(axes):
        grid.add([Cell(axis, header=True)] + [Cell(_fmt(p[i])) for p in pts])
    # The axes are the row labels here, so a statistic headed `z` sits
    # directly under the row headed `z (mm)` and nothing tells them apart.
    # A paper writes `Z value` or `Z-max` when its axes are already z.
    kind = rng.choice(["Z", "T"])
    lower = {a.split(" ")[0].lower() for a in axes}
    stat = rng.choice([h for h in vocab.STAT_HEADERS[kind]
                       if h.lower() not in lower] or ["%s value" % kind])
    grid.add([Cell(stat, header=True)]
             + [Cell(_fmt(_stat_value(rng, kind))) for _ in pts])
    name = _analysis_name(rng)
    truth = Truth(space=space, analyses=[TruthAnalysis(
        name=name, points=[TruthPoint(*p) for p in pts])])
    caption = "Table %d. %s. Coordinates are in %s space." % (
        rng.randint(1, 6), name, "Talairach" if space == "TAL" else "MNI")
    return Table(grid=grid, truth=truth, caption=caption,
                 footer=_plain_footer(rng, w),
                 notes={"layout": ["transposed"], "transposed": True,
                        "reader_cannot": True, "dividers": 0,
                        "space_in_table": False, "axes_named": True,
                        "identical_neighbours": False, "zero_x_rows": 0,
                        "packed_form": "plain"})


def _roi_centroids(rng: random.Random, w: Weights) -> Table:
    """The coordinates are in the column headers, naming the regions.

    `#<2:Left DLPFC (-45, 15, 35) | #<2:Right DLPFC (40, 20, 35)` over a body
    of counts. Whether those count as an analysis is a later stage's question;
    they are coordinates and the table states them.
    """
    lobe = rng.choice(list(vocab.LOBE_SECTIONS))
    pool = [r for r in vocab.REGIONS if r.lobe == lobe] or list(vocab.REGIONS)
    bag = list(pool); rng.shuffle(bag)
    n = min(rng.randint(2, 5), len(bag))          # one column per region
    heads, pts = [], []
    for _ in range(n):
        region = bag.pop()
        side = None if region.midline else rng.choice(["L", "R"])
        x, y, z = _coord(rng, region.sided(side), w)
        label = region.name[0].upper() + region.name[1:]
        if side:
            label = "%s %s" % ({"L": "Left", "R": "Right"}[side], label)
        heads.append("%s (%s, %s, %s)" % (label, _fmt(x), _fmt(y), _fmt(z)))
        pts.append((x, y, z))
    space = rng.choice(["MNI", "TAL"])
    grid = Grid()
    grid.add([Cell("", header=True)] + [Cell(h, header=True, colspan=2) for h in heads])
    grid.add([Cell("", header=True)]
             + [Cell(x, header=True) for _ in heads for x in ("Medial", "Lateral")])
    for label in rng.sample(vocab.CONDITIONS, min(5, len(vocab.CONDITIONS))):
        grid.add([Cell(label.capitalize())]
                 + [Cell(str(rng.randint(0, 12))) for _ in heads for _ in (0, 1)])
    name = _analysis_name(rng)
    truth = Truth(space=space, analyses=[TruthAnalysis(
        name=name, points=[TruthPoint(*p) for p in pts])])
    return Table(grid=grid, truth=truth,
                 caption="Table %d. Regions of interest and their centroids. "
                         "Coordinates are in %s space."
                         % (rng.randint(1, 6), "Talairach" if space == "TAL" else "MNI"),
                 footer=_plain_footer(rng, w),
                 notes={"layout": ["roi_centroids"], "roi_centroids": True,
                        "reader_cannot": True, "dividers": 0,
                        "space_in_table": False, "axes_named": False,
                        "identical_neighbours": False, "zero_x_rows": 0,
                        "packed_form": "plain"})


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
    # One column per group holding the whole triple, rather than three. The
    # group header then sits directly over a single column, so nothing spans
    # and the only thing saying which contrast a peak belongs to is which
    # column it is in. 157 of the tables the old filter kept are built this
    # way, and the generator made none of them.
    packed = rng.random() < w.triple_column_per_group
    if packed:
        per_group = ["xyz"] + (["stat"] if rng.random() < w.statistic_printed
                               else [])
    else:
        per_group = ["x", "y", "z"] + (["stat"]
                                       if rng.random() < w.statistic_printed
                                       else [])
    form = _packed_form(rng, w)
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

    # An axis that names its kind states the space, so the target says so
    # too: `x (Talairach)` in the header with a null space in the target is
    # the same defect as the other way round, read backwards.
    axes = _axis_names(rng, w, space)
    space = _names_space(axes) or space
    # One label for the statistic, not one per group: a table does not head the
    # same quantity `z` under one contrast and `Z value` under the next.
    stat_header = rng.choice(vocab.STAT_HEADERS[stat_kind])
    coord_header = rng.choice(vocab.BARE_COORD_HEADERS)
    top = [Cell(rng.choice(vocab.REGION_HEADERS), header=True, rowspan=2)]
    second: List[Cell] = []
    for name in names:
        top.append(Cell(name, header=True, colspan=len(per_group)))
        for role in per_group:
            if role == "xyz":
                second.append(Cell(coord_header, header=True))
            else:
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
                if role == "xyz":
                    cells.append(Cell(_packed(rng, x, y, z,
                                              _PackedAs(form), w, value)))
                else:
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
                        "triple_column_per_group": packed,
                        "packed_form": form if packed else "plain",
                        "dividers": 0, "space_in_table": False,
                        "axes_named": True, "identical_neighbours": False,
                        "zero_x_rows": 0})


def build(seed: int = 0, weights: Weights = DEFAULT) -> Table:
    """One synthetic table and the target a faithful reader should produce."""
    rng = random.Random(seed)
    w = weights
    roll = rng.random()
    if roll < w.coordinates_along_rows:
        return _transposed(rng, w)
    if roll < w.coordinates_along_rows + w.roi_centroid_in_header:
        return _roi_centroids(rng, w)
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

    # Five shapes the reader reads nothing out of, which is what they are for:
    # they are the residual gate's positive class, and it had 34 examples.
    reader_blind = (lay.voxel_indices or lay.signs_spaced or lay.coords_first
                    or lay.coords_in_name or lay.unmarked_span)
    notes = {"layout": list(lay.columns), "axes_named": lay.axes_named,
             "packed_form": lay.packed_form,
             "voxel_indices": lay.voxel_indices,
             "signs_spaced": lay.signs_spaced,
             "coords_first": lay.coords_first,
             "coords_in_name": lay.coords_in_name,
             "unmarked_span": lay.unmarked_span,
             # A shape no reader can follow: the anatomical axes rather than
             # x, y and z. Generated on purpose, because only a model can do
             # it and it will never see one otherwise.
             "reader_cannot": (reader_blind
                               or lay.axis_names[0] in ("R", "Right")),
             "space_in_table": lay.space_in_table, "dividers": 0,
             "identical_neighbours": False, "zero_x_rows": 0}

    # A paper that writes one peak as `±30` writes all of them that way, so
    # the form is chosen per table and then used on roughly a third of its
    # rows. Rolling it per row instead put a bilateral row in one table in
    # six, which is five times the rate the corpus shows.
    bilateral = rng.random() < w.bilateral_pair

    # Every contrast in the table found nothing. The table still names them,
    # so the target still holds them -- each with no points. A target that
    # dropped them would say the paper never ran the contrasts.
    nothing_at_all = rng.random() < w.analysis_found_nothing / 2

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

        # A contrast the paper ran that found nothing. The analysis stays in
        # the target with an empty point list: the table names it, so the
        # document states that it was run, and a target that drops it says the
        # paper never looked. It needs a banner to be named at all.
        if use_banner and (nothing_at_all
                           or rng.random() < w.analysis_found_nothing):
            grid.add([Cell(rng.choice(NOTHING_FOUND), colspan=width)])
            notes["empty_analyses"] = notes.get("empty_analyses", 0) + 1
            truth.analyses.append(analysis)
            continue

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
            # A rowspan in column 0 carries the label down: the load-bearing
            # form. Only where column 0 IS the label. A table that opens on
            # the triple would otherwise span the x column and trim x from
            # every row beneath it, which loses a coordinate per row and
            # leaves the target asserting numbers the table no longer prints.
            if rows_for_label == 0 and n_points - pi >= 2 \
                    and lay.columns[0] == "region" \
                    and rng.random() < w.rowspan_in_first_column:
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

            mirrored = None
            if bilateral and rng.random() < 0.35:
                mirrored = _bilateral(cells, lay, point)
            if lay.signs_spaced:
                offset = len(lay.columns) - len(cells)
                for ci, cell in enumerate(cells):
                    if lay.columns[ci + offset] in ("x", "y", "z", "xyz"):
                        cell.text = _space_the_sign(cell.text)
                    elif lay.coords_in_name and lay.columns[ci + offset] == "region":
                        cell.text = _space_the_sign(cell.text)
            _rough_up(rng, cells, lay, w, point)
            grid.add(cells)
            analysis.points.append(point)
            if mirrored is not None:
                # `_rough_up` may have withdrawn a statistic or an extent the
                # row no longer prints; the mirror states exactly what the
                # original does, so it is copied after, not before.
                mirrored.statistic_type = point.statistic_type
                mirrored.statistic_value = point.statistic_value
                mirrored.extent = point.extent
                analysis.points.append(mirrored)
                notes["bilateral_rows"] = notes.get("bilateral_rows", 0) + 1
                notes["reader_cannot"] = True
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
