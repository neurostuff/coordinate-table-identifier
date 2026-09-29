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

#: Ordered, so a trained model's coefficients can be read against it.
NAMES: List[str] = [
    "n_rows", "n_cols", "n_cells",
    "frac_numeric", "frac_in_head", "frac_integer", "frac_even", "frac_negative",
    "rows_with_triple", "frac_rows_with_triple", "max_triples_in_a_row",
    "packed_triple_cells", "frac_packed",
    "has_axis_header", "n_header_rows", "frac_header_cells",
    "has_span", "max_colspan", "max_rowspan",
    "n_numeric_columns", "widest_numeric_run",
    "coord_words_table", "coord_words_context",
    "region_words", "frac_rows_with_region",
    "other_topic_context", "other_topic_table",
    "stat_words", "extent_words", "laterality_rows",
    "caption_len", "footer_len", "has_caption",
]


def _numbers(row: Sequence) -> List[Optional[float]]:
    return [as_number(p.cell.text) for p in row]


def _in_head(trio: Sequence[float]) -> bool:
    return read.in_head(trio)


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
        return {k: float(f[k]) for k in NAMES}

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

    f["caption_len"] = len(caption or "")
    f["footer_len"] = len(footer or "")
    f["has_caption"] = 1.0 if (caption or "").strip() else 0.0
    return {k: float(f[k]) for k in NAMES}


def matrix(items) -> List[List[float]]:
    """Feature rows for an iterable of (text_or_grid, caption, footer)."""
    return [[vector(t, c, ft)[n] for n in NAMES] for t, c, ft in items]
