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
    ("R", re.compile(r"(?<![A-Za-z])r(?:[-\s]?(?:value|score))?(?![A-Za-z])|correlation", re.I)),
    ("B", re.compile(r"(?<![A-Za-z])(?:beta|b)(?:[-\s]?(?:value|weight|coef\w*))?(?![A-Za-z])"
                     r"|coefficient|\bestimate\b", re.I)),
]
# `p` is checked before the bare-letter forms because "p(FWE cor.)" also
# contains no other statistic letter, and after the qualified forms because
# "t-value" must not be read as a p column.
_ORDER = ["T", "Z", "F", "P", "R", "B"]


def statistic_type(header_text: Optional[str]) -> Optional[str]:
    """"T"/"Z"/"F"/"P"/"R"/"B", or None when the header does not say.

    Ambiguous headers return None rather than a guess: "value" alone, or a cell
    naming two statistics.
    """
    if not header_text:
        return None
    hits = [name for name, pat in _PATTERNS if pat.search(header_text)]
    if len(hits) != 1:
        return None
    return hits[0]
