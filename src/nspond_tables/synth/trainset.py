"""The v19 training set: real tables first, generated ones to fill the gaps.

A synthetic negative is a guess about what a non-coordinate table looks like.
A real one is not, so the real ones go in first and the generator supplies the
shapes the corpus is thin on, not the bulk.

Four sources, in descending order of how much they are worth:

* **hand-judged negatives** -- tables read one by one and confirmed to hold no
  coordinates. The most valuable, because they are the ones the heuristics
  could not place and a model will actually be handed.
* **reader false positives** -- real tables the reader pulls a triple out of
  that hold no coordinates: a median with its range, an odds ratio with its
  interval, an F with its degrees of freedom. These are what the model sees
  when the gate is wrong, so they are the negatives that matter most in
  deployment.
* **heuristic negatives** -- placed by structural and topical evidence rather
  than by eye. Sampled rather than taken whole: there are more of them than
  everything else combined, and they are the easy ones.
* **generated** -- for the shapes the corpus does not supply in quantity, and
  to keep the negative half from being memorisable.

Every non-coordinate table, whatever its source, carries the same target: the
empty structure. Not a missing field, not a refusal, and not an analysis with
an empty point list.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional

from .weights import DEFAULT, Weights

EMPTY_TARGET: Dict = {"space": None, "analyses": []}


@dataclass
class Example:
    """One training example, whatever produced it."""

    table: str
    target: Dict
    caption: str = ""
    footer: str = ""
    origin: str = "generated"          # which of the four sources
    notes: Dict = field(default_factory=dict)

    def as_row(self) -> Dict:
        return {"table": self.table, "caption": self.caption,
                "footer": self.footer, "target": self.target,
                "origin": self.origin, "notes": self.notes}


def _real_negatives(records: Iterable[Dict], *, heuristic_cap: int,
                    seed: int) -> List[Example]:
    """Real tables that hold no coordinates, best sources first."""
    from ..classify import dataset, features   # noqa: PLC0415

    hand, misread, heuristic = [], [], []
    for r in records:
        vec = features.vector(r.get("table_serialised") or "",
                              r.get("caption") or "", r.get("footer") or "")
        if dataset.label_of(r, vec) != 0:
            continue
        ex = Example(table=r.get("table_serialised") or "",
                     target=dict(EMPTY_TARGET),
                     caption=r.get("caption") or "", footer=r.get("footer") or "",
                     notes={"source": r.get("source"), "table_id": r.get("table_id")})
        if r.get("hand_judged"):
            ex.origin = "hand-judged"
            hand.append(ex)
        elif vec["reader_points"] > 0:
            ex.origin = "reader-false-positive"
            misread.append(ex)
        else:
            ex.origin = "heuristic"
            heuristic.append(ex)

    rng = random.Random(seed)
    rng.shuffle(heuristic)
    return hand + misread + heuristic[:heuristic_cap]


def build_trainset(n: int = 4000, *, records: Optional[Iterable[Dict]] = None,
                   weights: Weights = DEFAULT, seed: int = 0,
                   heuristic_cap: int = 400) -> List[Example]:
    """`n` examples, `weights.no_coordinates` of them holding nothing.

    Real negatives are used first and the generator makes up the shortfall, so
    growing `n` adds synthetic tables rather than diluting the real ones.
    """
    from .build import build                    # noqa: PLC0415
    from .empty import build_empty              # noqa: PLC0415

    want_empty = int(n * weights.no_coordinates)
    real = _real_negatives(records or [], heuristic_cap=heuristic_cap, seed=seed)
    real = real[:want_empty]
    out: List[Example] = list(real)
    for i in range(want_empty - len(real)):
        t = build_empty(seed=seed * 100003 + i, weights=weights)
        out.append(Example(table=t.grid.render(), target=t.truth.as_target(),
                           caption=t.caption, footer=t.footer,
                           origin="generated", notes=t.notes))
    for i in range(n - want_empty):
        t = build(seed=seed * 100003 + i, weights=weights)
        out.append(Example(table=t.grid.render(), target=t.truth.as_target(),
                           caption=t.caption, footer=t.footer,
                           origin="generated", notes=t.notes))
    random.Random(seed).shuffle(out)
    return out


def write_jsonl(examples: Iterable[Example], path) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex.as_row()) + "\n")
            n += 1
    return n


def census(examples: Iterable[Example]) -> Dict[str, int]:
    """What the set is made of, for the record that goes beside it."""
    import collections                          # noqa: PLC0415

    c: Dict[str, int] = collections.Counter()
    for ex in examples:
        empty = not ex.target.get("analyses")
        c["total"] += 1
        c["empty" if empty else "with coordinates"] += 1
        if empty:
            c["empty: " + ex.origin] += 1
            if ex.origin == "generated":
                c["empty: generated " + str(ex.notes.get("kind", "?"))] += 1
    return dict(c)
