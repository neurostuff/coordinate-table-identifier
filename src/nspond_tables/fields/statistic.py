"""Which statistic a column holds, from its header.

Curated labels leave this null on 52% of tables because the paper never names
it. A reader must be willing to return None as often.
"""

import re
from typing import Optional

_PATTERNS = [
    ("T", re.compile(r"(?<![A-Za-z])t(?:[-\s]?(?:value|score|stat\w*))?(?![A-Za-z])|t\s*\(\s*\d+\s*\)", re.I)),
    ("Z", re.compile(r"(?<![A-Za-z])z(?:[-\s]?(?:value|score|stat\w*))?(?![A-Za-z])", re.I)),
    ("F", re.compile(r"(?<![A-Za-z])F(?:[-\s]?(?:value|score|stat\w*))?(?![A-Za-z])|F\s*\(\s*\d+", re.I)),
    ("P", re.compile(r"(?<![A-Za-z])p(?:[-\s]?(?:value|val|corr\w*|unc\w*|FWE|FDR))?(?![A-Za-z])", re.I)),
    ("D", re.compile(r"cohen'?s?\s*d\b|effect size\s*\(\s*d\s*\)|"
                     r"(?<![A-Za-z])d(?![A-Za-z])", re.I)),
    ("G", re.compile(r"hedge'?s?'?\s*g\b", re.I)),
    ("R", re.compile(r"(?<![A-Za-z])r(?:[-\s]?(?:value|score))?(?![A-Za-z])|correlation", re.I)),
    ("B", re.compile(r"(?<![A-Za-z])(?:beta|b)(?:[-\s]?(?:value|weight|coef\w*))?(?![A-Za-z])"
                     r"|coefficient|\bestimate\b", re.I)),
]
# `p` is checked before the bare-letter forms because "p(FWE cor.)" also
# contains no other statistic letter, and after the qualified forms because
# "t-value" must not be read as a p column.
#: Which statistic a document reports when it names more than one. A p-value
#: is a significance level rather than a test statistic and comes last; Cohen's
#: d and Hedges' g sit between Z and F, since a table printing an effect size
#: almost always prints the t or p it was derived from beside it.
#:
#: This is the single source of the rule. `fields.statistic_type` reads one
#: header cell with it and `synth.statistic_named_by` reads a whole document
#: through that, so there is one place to change it.
STATISTIC_PRIORITY = ("T", "Z", "D", "G", "F", "R", "B", "P")

_ORDER = list(STATISTIC_PRIORITY)


# A laterality header is not a statistic column. `L/R` matched the R pattern --
# a standalone r with no letter before it -- and claiming that column left the
# real value column unread.
_SIDE_ONLY = re.compile(
    r"^\s*(?:L\s*/\s*R|R\s*/\s*L|side|hemisphere|hemi\.?|lat\.?|H)\s*$", re.I)

# Nor is a coordinate header. One cell often spans the three axes --
# `Peak MNI coordinate (X coord, Y coord, Z coord)`, `Talairach (x, y, z)` --
# and the bare-letter rule read its `z` as a Z statistic. Three axis letters
# in order, close together, is a coordinate column and never a statistic.
# An abbreviation legend is not a claim. `L: left; R: right.` is one of the
# commonest footnotes there is, and reading its R as a correlation named a
# statistic the table never reports.
_LEGEND = re.compile(
    r"(?<![A-Za-z])[LRBlrb]\s*[:=]\s*(?:left|right|bilateral|both)\b", re.I)

_COORD_HEADER = re.compile(
    r"(?<![A-Za-z])x(?![A-Za-z]).{0,30}?(?<![A-Za-z])y(?![A-Za-z])"
    r".{0,30}?(?<![A-Za-z])z(?![A-Za-z])", re.I | re.S)


def statistic_type(header_text: Optional[str]) -> Optional[str]:
    """What one header cell names, or None when it does not say.

    A cell naming several is resolved by `STATISTIC_PRIORITY` rather than
    declined. `#Cohen's d at the peak voxel` names d; `p (FWE)` beside a `t`
    names both, and the t is what the table reports.

    .. versionchanged::
       Used to return None whenever a cell matched more than one pattern. That
       discarded the answer on exactly the tables that print a test statistic
       beside its significance level, which is the commonest multi-statistic
       shape in the corpus.
    """
    if not header_text or _SIDE_ONLY.match(header_text) \
            or _COORD_HEADER.search(header_text) or _LEGEND.search(header_text):
        return None
    hits = {name for name, pat in _PATTERNS if pat.search(header_text)}
    return best_of(hits)


def best_of(kinds) -> Optional[str]:
    """The statistic a document reports, given everything it names."""
    found = [k for k in STATISTIC_PRIORITY if k in kinds]
    return found[0] if found else None
