"""Decide whether a table could hold coordinates, confidently or not at all.

The point is labels without hand review. A table that the extractor found
coordinates in is a confident positive. A table that *cannot* hold coordinates
is a confident negative, and that is the class worth being careful about,
because `create_analyses.py` drops a table with no coordinates and the article
is then lost -- so a negative asserted wrongly costs an article.

Three verdicts:

    POSITIVE   a coordinate triple is present and plausible
    NEGATIVE   the table cannot hold one, on structure AND on topic
    UNCERTAIN  everything else -- the pool that needs a model, or a human

A negative requires both kinds of evidence. Structure alone is not enough: a
table printing coordinates in one packed cell has no run of three numeric
columns. Topic alone is not enough either: "Participant characteristics" is a
plausible caption for a table that also lists seed coordinates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence

from . import read
from .grid import Grid, as_number, parse

# Words that name a coordinate table, or a column of one.
COORD_WORDS = re.compile(
    r"\bMNI\b|\bICBM\b|talairach|tournoux|\bTAL\b"
    r"|co\s?-?ordinate|\bpeak\s+voxel|local\s+maxima|\bfoci\b|\bfocus\b"
    r"|activation|deactivation|\bcluster\b|\bvoxels?\b|\bBA\s?\d|brodmann"
    r"|\bROI\b|region\s+of\s+interest|\bseed\b|\bcontrast\b|\bSPM\b"
    r"|significant\s+(?:cluster|activation|region)|whole[-\s]brain",
    re.I)

# Words that name a table about something else entirely. Each one is a subject
# a paper tabulates that has no coordinates in it.
OTHER_TOPIC = re.compile(
    r"demographic|participant\s+characteristic|subject\s+characteristic"
    r"|inclusion\s+criteri|exclusion\s+criteri|search\s+(?:term|strateg)"
    r"|questionnaire|\bscale\b|\bsubscale\b|inventory|self[-\s]report"
    r"|reliabilit|validit|cronbach|inter[-\s]rater"
    r"|stimulus\s+(?:list|set)|stimuli\s+used|item\s+(?:list|pool)|word\s+list"
    r"|scan(?:ning)?\s+parameter|acquisition\s+parameter|sequence\s+parameter"
    r"|model\s+(?:specification|parameter|comparison|fit)|hyperparameter"
    r"|software|toolbox|\bversion\b"
    r"|studies\s+included|included\s+stud|literature|meta[-\s]analytic\s+sample"
    r"|behavio(?:u)?ral\s+(?:result|performance|data)|reaction\s+time"
    r"|accurac|\bRT\b\s|response\s+time|\bd['’]\b"
    r"|descriptive\s+statistic|correlation\s+matrix|factor\s+loading"
    r"|side\s+effect|adverse\s+event|medication|dosage|comorbid"
    r"|genotype|allele|\bSNP\b|primer",
    re.I)

_TRIPLE_IN_CELL = re.compile(r"-?\d{1,3}(?:\.\d+)?[\s,;]+-?\d{1,3}(?:\.\d+)?"
                             r"[\s,;]+-?\d{1,3}(?:\.\d+)?")


class Verdict(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNCERTAIN = "uncertain"


@dataclass
class Detection:
    verdict: Verdict
    reasons: List[str] = field(default_factory=list)
    points: int = 0
    located_by: str = "none"

    def __bool__(self) -> bool:
        return self.verdict is Verdict.POSITIVE


def _coordlike_runs(grid: Grid) -> int:
    """Rows holding three adjacent cells that could be x, y and z.

    Adjacent, because a coordinate triple occupies consecutive columns. The
    values must be in range, which is what stops a table of ages and scores
    from qualifying.
    """
    n = 0
    for row in grid.body_rows():
        vals = [as_number(p.cell.text) for p in row]
        for i in range(len(vals) - 2):
            trio = vals[i:i + 3]
            if all(v is not None for v in trio) and read.in_head(trio):
                n += 1
                break
    return n


def _packed_cells(grid: Grid) -> int:
    n = 0
    for row in grid.body_rows():
        for placed in row:
            m = _TRIPLE_IN_CELL.search(placed.cell.text)
            if m and read.in_head([float(g) for g in
                                   re.findall(r"-?\d{1,3}(?:\.\d+)?", m.group())[:3]]):
                n += 1
                break
    return n


def detect(text_or_grid, caption: str = "", footer: str = "",
           extractor_found: Optional[bool] = None) -> Detection:
    """Classify one table. `extractor_found` short-circuits to POSITIVE."""
    grid = parse(text_or_grid) if isinstance(text_or_grid, str) else text_or_grid
    body = grid.render()
    context = " ".join([caption or "", footer or ""])
    reasons: List[str] = []

    if extractor_found:
        return Detection(Verdict.POSITIVE, ["extractor found coordinates"])

    got = read.extract(grid, caption=caption, footer=footer)
    if len(got.points) >= 3:
        return Detection(Verdict.POSITIVE,
                         ["reader located %d points by %s" % (len(got.points), got.located_by)],
                         points=len(got.points), located_by=got.located_by)

    runs = _coordlike_runs(grid)
    packed = _packed_cells(grid)
    coord_words = bool(COORD_WORDS.search(body)) or bool(COORD_WORDS.search(context))
    other = OTHER_TOPIC.search(context) or OTHER_TOPIC.search(body[:400])

    # Structure: nothing in this table looks like a coordinate at all.
    if runs == 0 and packed == 0:
        reasons.append("no adjacent in-range numeric triple, and none packed in a cell")
        if not coord_words:
            reasons.append("no coordinate vocabulary in the table, caption or footer")
            return Detection(Verdict.NEGATIVE, reasons)
        if other:
            reasons.append("caption names another subject: %r" % other.group(0))
            return Detection(Verdict.NEGATIVE, reasons)
        return Detection(Verdict.UNCERTAIN,
                         reasons + ["but coordinate vocabulary is present"])

    # Structurally possible. A single stray run is weak evidence either way.
    if other and not coord_words:
        return Detection(Verdict.NEGATIVE,
                         ["caption names another subject: %r" % other.group(0),
                          "and no coordinate vocabulary anywhere"])
    return Detection(Verdict.UNCERTAIN,
                     ["%d rows with an in-range triple, %d packed; vocabulary %s"
                      % (runs, packed, "present" if coord_words else "absent")],
                     points=len(got.points), located_by=got.located_by)
