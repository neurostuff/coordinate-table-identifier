"""Left/right in a row's text, and whether the x sign agrees with it.

In MNI and Talairach the left hemisphere has negative x. Curated labels agree
with the row's own laterality marker on 97.3% of 60,727 unambiguous rows
(laterality.py over train_v18). That makes it a usable check on an extracted
sign, and a usable fallback when a table prints no minus signs at all -- there
are such tables, and a reader that says "Left inferior temporal gyrus" implies
x < 0 is doing something a sign-blind reader cannot.

It is a check, not a correction. Midline structures sit at x = 0 where the
marker carries no information, and papers do label near-midline voxels
inconsistently.
"""

import re
from enum import Enum
from typing import Optional

_BILATERAL = re.compile(r"\bL\s*/\s*R\b|\bR\s*/\s*L\b|bilat|\bboth\b|\bmidline\b", re.I)
_LEFT = re.compile(r"(?<![A-Za-z])left(?![A-Za-z])|(?<![A-Za-z/])L(?![A-Za-z/])")
_RIGHT = re.compile(r"(?<![A-Za-z])right(?![A-Za-z])|(?<![A-Za-z/])R(?![A-Za-z/])", re.I)


class Side(Enum):
    LEFT = "L"
    RIGHT = "R"
    BILATERAL = "B"


def laterality(text: Optional[str]) -> Optional[Side]:
    if not text:
        return None
    if _BILATERAL.search(text):
        return Side.BILATERAL
    left, right = bool(_LEFT.search(text)), bool(_RIGHT.search(text))
    if left and not right:
        return Side.LEFT
    if right and not left:
        return Side.RIGHT
    return None


def sign_agrees(x: Optional[float], text: Optional[str]) -> Optional[bool]:
    """True/False when both a side and a signed x are available, else None."""
    side = laterality(text)
    if side is None or side is Side.BILATERAL or x is None or x == 0:
        return None
    return (side is Side.LEFT and x < 0) or (side is Side.RIGHT and x > 0)
