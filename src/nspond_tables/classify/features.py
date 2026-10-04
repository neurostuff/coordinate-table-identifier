"""What a table looks like, as numbers a model can learn from.

Every feature is cheap: no model, no network, one pass over a parsed grid plus
the caption and footer. That is deliberate. A coordinate table is a distinctive
object -- three adjacent columns of small signed integers, under a header that
usually names them, beside a statistic and a cluster size -- and the separating
signal is almost entirely surface-level. Reach for a text encoder only if this
plateaus below the recall you need.

The asymmetry that shapes the whole design: `create_analyses` drops a table with
no coordinates, so a **false negative loses the article permanently** while a
false positive costs one wasted call that returns nothing. Features are chosen
to make recall cheap, and the threshold is set on recall at a fixed precision
rather than on accuracy.
"""

from __future__ import annotations

import collections
import math
import re
from typing import Dict, List, Optional, Sequence

from .. import read
from ..grid import Cell, Grid, as_number, parse

# A coordinate triple occupies consecutive columns; the reader needs the same
# bound, so it lives in one place.
LIMITS = read.LIMITS

COORD_WORDS = re.compile(
    r"\bMNI\b|\bICBM\b|talairach|tournoux|\bTAL\b|co\s?-?ordinate"
    r"|\bpeak\s+voxel|local\s+maxima|\bfoci\b|activation|deactivation"
    r"|\bcluster\b|\bvoxels?\b|\bBA\s?\d|brodmann|whole[-\s]brain"
    r"|significant\s+(?:cluster|activation|region)", re.I)

REGION_WORDS = re.compile(
    r"gyrus|cortex|\blobe\b|lobule|sulcus|nucleus|thalam|hippocamp|amygdal"
    r"|cerebell|insula|cingulate|precuneus|cuneus|putamen|caudate|pallidum"
    r"|operculum|fusiform|calcarine|vermis|brainstem|striatum|\bSMA\b", re.I)

OTHER_TOPIC = re.compile(
    r"demographic|participant\s+characteristic|inclusion\s+criteri"
    r"|questionnaire|\bscale\b|inventory|self[-\s]report|reliabilit|cronbach"
    r"|stimulus\s+(?:list|set)|stimuli\s+used|scan(?:ning)?\s+parameter"
    r"|model\s+(?:specification|parameter|comparison|fit)|software|toolbox"
    r"|studies\s+included|literature|behavio(?:u)?ral\s+(?:result|performance)"
    r"|reaction\s+time|accurac|descriptive\s+statistic|factor\s+loading"
    r"|side\s+effect|medication|dosage|genotype|\bSNP\b", re.I)

STAT_WORDS = re.compile(
    r"\bt[-\s]?(?:value|score|stat)|\bz[-\s]?(?:value|score|stat)|\bp[-\s]?value"
    r"|\bF\s*\(|\bbeta\b|\bFWE\b|\bFDR\b|corrected|uncorrected|threshold", re.I)

EXTENT_WORDS = re.compile(
    r"cluster\s*(?:size|extent)|\bextent\b|\bsize\b|\bvoxels?\b|\bk\b|\bkE\b"
    r"|mm\s*\^?3", re.I)

LATERALITY = re.compile(r"(?<![A-Za-z])(?:left|right)(?![A-Za-z])|(?<![A-Za-z/])[LR](?![A-Za-z/])")

# A header that names coordinates, as against the same words loose in the body.
# An ACE supplementary-file listing whose descriptions mention Brodmann areas
# and cortical surfaces scored 0.998 on the loose form; no header of a table
# that holds no coordinates said any of this.
COORD_HEADER = re.compile(
    r"\bMNI\b|talairach|\bTAL\b|co\s?-?ordinate|\bpeak\b|maxima|\bmaximum\b"
    r"|\bfoci\b|stereotax|centroid|x\s*[,;/ ]?\s*y\s*[,;/ ]?\s*z\b", re.I)
# `1.5 (0.3-7.8)` and `58 (43-63)` are an estimate with its interval, and both
# read as a triple. The bracket is what says so.
BRACKETED = re.compile(r"\w\s*[\(\[]")

#: Ordered, so a trained model's coefficients can be read against it.
NAMES: List[str] = [
    "n_rows", "n_cols", "n_cells",
    "frac_numeric", "frac_in_head", "frac_integer", "frac_even", "frac_negative",
    "rows_with_triple", "frac_rows_with_triple", "max_triples_in_a_row",
    "packed_triple_cells", "frac_packed",
    "has_axis_header", "n_header_rows", "frac_header_cells",
    "reader_points", "reader_by_header", "reader_by_packed", "frac_rows_read",
    "coord_words_header", "frac_triples_integer", "frac_triples_bracketed",
    "frac_triples_straddling",
    "region_words_first_column",
    "has_span", "max_colspan", "max_rowspan",
    "n_numeric_columns", "widest_numeric_run",
    "coord_words_table", "coord_words_context",
    "region_words", "frac_rows_with_region",
    "other_topic_context", "other_topic_table",
    "stat_words", "extent_words", "laterality_rows",
    "caption_len", "footer_len", "has_caption",
    "frac_triples_whole", "frac_cells_interval",
    "frac_cells_embedded_triple", "header_triple_cells",
]


_BRACKET_RUN = re.compile(r"[\(\[][^\)\]]*[\)\]]?")


def _numbers_with_groups(text: str):
    """Each number in a cell, and the bracket it sits in (None when outside one)."""
    spans = [m.span() for m in _BRACKET_RUN.finditer(text)]
    out = []
    for m in _NUMBER.finditer(text):
        i = m.start()
        group = next((k for k, (a, b) in enumerate(spans) if a < i < b), None)
        out.append((m, group))
    return out


def _straddles_a_bracket(text: str) -> bool:
    """Whether no three numbers of the cell's triple share a side of a bracket.

    `15 (9.3, 24.4)` is a median with its range, `0.31(0.11,0.86)` an odds
    ratio with its interval, and `F(1,67) = 28.05` an F with its degrees of
    freedom. All three read as a triple, and in all three the first number sits
    outside a bracket and the other two inside. A coordinate never does that:
    `(-51, 20, 24)` is wholly inside, `-51 20 24` wholly outside.

    The test is on the triple, not the cell. Asking only whether the cell's
    numbers fall on both sides of a bracket also caught `19.17 (20, -92, -12)`
    -- a statistic and then its peak, the triple wholly inside -- and the gate
    learned to reject that layout: whole tables of it scored 0.001.
    """
    return bool(_BRACKET_RUN.search(text)) and not _whole_triple(text)


def _whole_triple(text: str) -> bool:
    """Three consecutive numbers on one side of a bracket, joined only by a
    comma, a semicolon or space, and in head bounds: how a peak is written,
    and not how an interval (`1.06 (1.02-1.11)`) or a mean and SD is."""
    nums = _numbers_with_groups(text)
    for i in range(len(nums) - 2):
        run = nums[i:i + 3]
        if len({g for _, g in run}) != 1:
            continue
        gaps = [text[run[k][0].end():run[k + 1][0].start()] for k in range(2)]
        if not all(_JOIN.fullmatch(g) for g in gaps):
            continue
        trio = [as_number(m.group()) for m, _ in run]
        if all(v is not None for v in trio) and read.in_head(trio):
            return True
    return False


_NUMBER = re.compile(r"[-+\u2212]?\d+(?:\.\d+)?")
_JOIN = re.compile(r"\s*[,;]?\s*")
# `1.06 (1.02-1.11)`, `67 ± 6`, `[52-77]`, `0.08 (-0.22, 0.38)` with a decimal
# point in every value: an estimate and its spread, the commonest thing the
# reader mistakes for a peak.
_INTERVAL = re.compile(
    r"\d\s*±\s*\d"
    r"|[\(\[]\s*-?\d+(?:\.\d+)?\s*(?:-|–|to)\s*-?\d+(?:\.\d+)?\s*[\)\]]"
    r"|[\(\[]\s*-?\d*\.\d+\s*,\s*-?\d*\.\d+\s*[\)\]]")


def _numbers(row: Sequence) -> List[Optional[float]]:
    return [as_number(p.cell.text) for p in row]


def _in_head(trio: Sequence[float]) -> bool:
    return read.in_head(trio)


#: Features that are counts rather than fractions. A count has no ceiling, and
#: a linear model fitted on tables with a median `max_rowspan` of 1 and a
#: maximum of 40 has nothing to say about one of 112 -- it extrapolates, and a
#: single out-of-range count swamps every other feature.
#:
#: That is not hypothetical. A 492-row ALE source table headed `MNI/Talairach`
#: and `Coordinates`, which the reader reads 491 points out of, scored 0.0007
#: and was dropped; its first 120 rows scored 0.984. The only thing that
#: changed was `max_rowspan` going from 16 to 112.
#:
#: log1p keeps the ordering and bounds the damage: the difference between 1
#: and 3 still matters, and the difference between 40 and 112 stops deciding
#: the answer on its own.
_COUNTS = (
    "n_rows", "n_cols", "n_cells", "rows_with_triple", "max_triples_in_a_row",
    "packed_triple_cells", "n_header_rows", "max_colspan", "max_rowspan",
    "n_numeric_columns", "widest_numeric_run", "coord_words_table",
    "coord_words_context", "coord_words_header", "region_words",
    "other_topic_context", "other_topic_table", "stat_words", "extent_words",
    "laterality_rows", "caption_len", "footer_len", "header_triple_cells",
)


def vector(text_or_grid, caption: str = "", footer: str = "") -> Dict[str, float]:
    """Named features for one table. Missing signal is 0, never None."""
    grid = parse(text_or_grid) if isinstance(text_or_grid, str) else text_or_grid
    body = grid.render()
    context = " ".join([caption or "", footer or ""])
    rows = list(grid.resolve())
    body_rows = list(grid.body_rows())

    f = collections.defaultdict(float)
    f["n_rows"] = len(rows)
    f["n_cols"] = grid.width()
    cells = [p.cell for row in rows for p in row if not p.cell.is_filler]
    f["n_cells"] = len(cells)
    if not cells:
        return {k: (math.log1p(max(float(f[k]), 0.0)) if k in _COUNTS else float(f[k]))
            for k in NAMES}

    values = [as_number(c.text) for c in cells]
    numeric = [v for v in values if v is not None]
    f["frac_numeric"] = len(numeric) / len(cells)
    if numeric:
        f["frac_in_head"] = sum(
            1 for v in numeric if abs(v) <= LIMITS["y"]) / len(numeric)
        f["frac_integer"] = sum(
            1 for v in numeric if float(v).is_integer()) / len(numeric)
        ints = [v for v in numeric if float(v).is_integer()]
        f["frac_even"] = (sum(1 for v in ints if int(v) % 2 == 0) / len(ints)) if ints else 0.0
        f["frac_negative"] = sum(1 for v in numeric if v < 0) / len(numeric)

    triples = 0
    for row in body_rows:
        vals = _numbers(row)
        hits = sum(1 for i in range(len(vals) - 2)
                   if all(vals[i + k] is not None for k in range(3))
                   and _in_head([vals[i + k] for k in range(3)]))
        if hits:
            triples += 1
            f["max_triples_in_a_row"] = max(f["max_triples_in_a_row"], hits)
    f["rows_with_triple"] = triples
    f["frac_rows_with_triple"] = triples / max(len(body_rows), 1)

    packed = sum(1 for row in body_rows for p in row
                 if read._TRIPLE.match(p.cell.text or ""))
    f["packed_triple_cells"] = packed
    f["frac_packed"] = packed / max(len(body_rows), 1)

    f["has_axis_header"] = 1.0 if read.axis_columns(grid) else 0.0

    # What the reader makes of the table, as evidence rather than as an answer.
    # Every other feature here is a count of words or of shapes, and a
    # supplementary-file listing whose descriptions name Brodmann areas scores
    # 0.998 on vocabulary alone. Whether three columns actually read as
    # coordinates in head bounds is the thing none of them says.
    got = read.extract(grid, caption=caption, footer=footer)
    f["reader_points"] = math.log1p(len(got.points))
    f["reader_by_header"] = 1.0 if got.located_by == "header" else 0.0
    f["reader_by_packed"] = 1.0 if got.located_by == "packed cell" else 0.0
    f["frac_rows_read"] = len(got.points) / len(body_rows) if body_rows else 0.0

    # Where the coordinate words are matters more than how many there are.
    # The same stand-in the reader uses: 6.6% of coordinate tables write their
    # header in <td>, and reading only marked headers made `Talairach` in a
    # header invisible on every one of them.
    header_text = " ".join(p.cell.text for p in read._header_cells(grid))
    f["coord_words_header"] = float(len(COORD_HEADER.findall(header_text)))

    # What the candidate triples look like. A peak is whole and unbracketed; an
    # odds ratio with its interval is neither, and both parse the same way.
    trip_cells = [p.cell.text for row in body_rows for p in row
                  if not p.cell.is_filler and read.triples_in(p.cell.text)]
    if trip_cells:
        whole = sum(1 for t in trip_cells
                    if all(float(v).is_integer() for trio in read.triples_in(t) for v in trio))
        f["frac_triples_integer"] = whole / len(trip_cells)
        f["frac_triples_bracketed"] = sum(
            1 for t in trip_cells if BRACKETED.search(t)) / len(trip_cells)
        f["frac_triples_straddling"] = sum(
            1 for t in trip_cells if _straddles_a_bracket(t)) / len(trip_cells)
        f["frac_triples_whole"] = sum(1 for t in trip_cells if _whole_triple(t)) / len(trip_cells)

    # Anatomy in the row labels, not anywhere in the table. A file listing
    # naming cortical surfaces in its descriptions is not a coordinate table.
    labels = [read._row_label(row, ()) for row in body_rows]
    named = [l for l in labels if l]
    if named:
        f["region_words_first_column"] = sum(
            1 for l in named if REGION_WORDS.search(l)) / len(named)
    header_cells = [c for c in cells if c.header]
    f["frac_header_cells"] = len(header_cells) / len(cells)
    f["n_header_rows"] = sum(1 for row in rows
                             if row and all(p.cell.header for p in row
                                            if not p.cell.is_filler and p.cell.text.strip()))
    f["max_colspan"] = max((c.colspan for c in cells), default=1)
    f["max_rowspan"] = max((c.rowspan for c in cells), default=1)
    f["has_span"] = 1.0 if (f["max_colspan"] > 1 or f["max_rowspan"] > 1) else 0.0

    per_col = collections.defaultdict(list)
    for row in body_rows:
        for p in row:
            if not p.cell.is_filler and p.cell.text.strip():
                per_col[p.col].append(as_number(p.cell.text) is not None)
    numeric_cols = sorted(c for c, hits in per_col.items()
                          if hits and sum(hits) >= 0.6 * len(hits))
    f["n_numeric_columns"] = len(numeric_cols)
    run = best = 0
    prev = None
    for c in numeric_cols:
        run = run + 1 if prev is not None and c == prev + 1 else 1
        best = max(best, run)
        prev = c
    f["widest_numeric_run"] = best

    f["coord_words_table"] = len(COORD_WORDS.findall(body))
    f["coord_words_context"] = len(COORD_WORDS.findall(context))
    f["region_words"] = len(REGION_WORDS.findall(body))
    f["frac_rows_with_region"] = sum(
        1 for row in body_rows
        if REGION_WORDS.search(" ".join(p.cell.text for p in row))) / max(len(body_rows), 1)
    f["other_topic_context"] = len(OTHER_TOPIC.findall(context))
    f["other_topic_table"] = len(OTHER_TOPIC.findall(body[:600]))
    f["stat_words"] = len(STAT_WORDS.findall(body + " " + context))
    f["extent_words"] = len(EXTENT_WORDS.findall(body))
    f["laterality_rows"] = sum(
        1 for row in body_rows
        if LATERALITY.search(" ".join(p.cell.text for p in row))) / max(len(body_rows), 1)

    filled = [c.text for c in cells if not c.is_filler and (c.text or "").strip()]
    # A triple inside a longer cell -- `Left post hippocampus -12, -38, 4, p=.045` --
    # which the reader, wanting a cell that is only a triple, does not read; and
    # ROI centres written as column titles, `Left FG (-39, -51, -18)`.
    body_text = [p.cell.text for row in body_rows for p in row
                 if not p.cell.is_filler and (p.cell.text or "").strip()]
    f["frac_cells_embedded_triple"] = (sum(1 for t in body_text if _whole_triple(t)) / len(body_text)) if body_text else 0.0
    f["header_triple_cells"] = sum(1 for p in read._header_cells(grid) if _whole_triple(p.cell.text or ""))
    f["frac_cells_interval"] = (sum(1 for t in filled if _INTERVAL.search(t)) / len(filled)) if filled else 0.0
    f["caption_len"] = len(caption or "")
    f["footer_len"] = len(footer or "")
    f["has_caption"] = 1.0 if (caption or "").strip() else 0.0
    return {k: (math.log1p(max(float(f[k]), 0.0)) if k in _COUNTS else float(f[k]))
            for k in NAMES}


def matrix(items) -> List[List[float]]:
    """Feature rows for an iterable of (text_or_grid, caption, footer)."""
    return [[vector(t, c, ft)[n] for n in NAMES] for t, c, ft in items]
