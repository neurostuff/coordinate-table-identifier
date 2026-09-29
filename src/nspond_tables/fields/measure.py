"""Whether a cluster extent is in voxels or cubic millimetres.

The unit is stated in the column header far more often than the count is
stated anywhere else, and a count with no unit is a number with no meaning --
1,391 curated analyses report one (34(o) in scans/PRE_V17.md).
"""

import re
from typing import Optional

_VOXELS = re.compile(r"voxel|\bvox\b|\bk\b|\bke\b|n\s*vox", re.I)
_MM3 = re.compile(r"mm\s*(?:\^?3|³)|cubic\s*mm|\bcm\s*(?:\^?3|³)", re.I)


def cluster_measure(header_text: Optional[str]) -> Optional[str]:
    """"voxels", "mm^3", or None."""
    if not header_text:
        return None
    mm3, vox = bool(_MM3.search(header_text)), bool(_VOXELS.search(header_text))
    if mm3 and not vox:
        return "mm^3"
    if vox and not mm3:
        return "voxels"
    return None
