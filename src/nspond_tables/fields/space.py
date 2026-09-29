"""The coordinate space a reader could actually get from the table.

Null when nothing states it, and null when the text states both. This is the
rule the production prompt already gives the LLM, so the two agree.

A serialised header cell carries its span inline. `\\bMNI\\b` needs a word
boundary, which a digit beside a letter does not give, so `<3:MNI coordinates`
was invisible to it -- 3,062 of 24,409 rows, all labelled null while their own
table named the space. Talairach has no leading \\b and was unaffected, so the
loss fell entirely on MNI. Match on a letter boundary instead.
"""

import re
from typing import Optional

_NB, _NA = r"(?<![A-Za-z])", r"(?![A-Za-z])"
MNI = re.compile(_NB + r"(?:MNI|ICBM)" + _NA + r"|MNI-?152|montreal neuro", re.I)
TAL = re.compile(r"talairach|tournoux|" + _NB + r"(?:Tal|TT)" + _NA, re.I)


def visible_space(*texts: Optional[str]) -> Optional[str]:
    """"MNI", "TAL", or None -- from the visible text alone."""
    joined = " ".join(str(t or "") for t in texts)
    mni, tal = bool(MNI.search(joined)), bool(TAL.search(joined))
    if mni and not tal:
        return "MNI"
    if tal and not mni:
        return "TAL"
    return None
