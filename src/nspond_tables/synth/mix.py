"""One training set, holding both kinds of table.

A caller asking for example `n` should not have to know whether it holds
coordinates. `no_coordinates` decides, from the seed, so the proportion is
reproducible and a shuffle is not needed.
"""

from __future__ import annotations

import random

from .weights import DEFAULT, Weights


def build_mixed(seed: int = 0, weights: Weights = DEFAULT):
    """One synthetic table, with or without coordinates, and its target."""
    from .build import build                # noqa: PLC0415
    from .empty import build_empty          # noqa: PLC0415

    if random.Random("mix-%d" % seed).random() < weights.no_coordinates:
        return build_empty(seed=seed, weights=weights)
    return build(seed=seed, weights=weights)
