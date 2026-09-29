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


def _reader_target(text: str, caption: str, footer: str) -> Optional[Dict]:
    """A target read straight off a real table, or None if it cannot be.

    Only for tables the reader reads whole: the points are the reader's, and
    the grouping is the banner rows it can see, each point belonging to the
    banner above it. That is a real analysis boundary, not a guess.

    What this cannot supply is the grouping a banner does not mark -- an
    analysis split by a column, or by a footnote. The generator carries that
    case, which is why real positives supplement the generated ones rather
    than replacing them.
    """
    from .. import read                          # noqa: PLC0415

    got = read.extract(text, caption=caption, footer=footer)
    if len(got.points) < 3:
        return None
    cuts = sorted(got.sections)                  # (row index, banner text)
    analyses: List[Dict] = []
    for point in got.points:
        name = ""
        for row, label in cuts:
            if row < point.row:
                name = label
        if not analyses or analyses[-1]["name"] != name:
            analyses.append({"name": name, "points": []})
        analyses[-1]["points"].append(
            [point.x, point.y, point.z, point.statistic_type,
             point.statistic_value, point.extent])
    return {"space": got.space, "analyses": analyses}


def real_positives(records: Iterable[Dict], *, limit: Optional[int] = None
                   ) -> List[Example]:
    """Real coordinate tables the reader reads whole, with their own targets.

    These carry the layouts the generator does not invent: a header written in
    <td>, a triple packed into one cell, three columns under a single spanning
    `MNI coordinates`. The generator now produces all three, but at rates
    chosen by hand, and a real table is not a guess about its own shape.
    """
    from ..classify import dataset, features      # noqa: PLC0415

    out: List[Example] = []
    for r in records:
        text = r.get("table_serialised") or ""
        cap, foot = r.get("caption") or "", r.get("footer") or ""
        vec = features.vector(text, cap, foot)
        if dataset.label_of(r, vec) != 1:
            continue
        target = _reader_target(text, cap, foot)
        if target is None:
            continue
        out.append(Example(table=text, target=target, caption=cap, footer=foot,
                           origin="real-positive",
                           notes={"source": r.get("source"),
                                  "table_id": r.get("table_id"),
                                  "analyses": len(target["analyses"])}))
        if limit and len(out) >= limit:
            break
    return out


def curated_positives(curated: Iterable[Dict]) -> List[Example]:
    """The examples earlier versions were trained on, as they stand.

    These are the training set's backbone and the generator does not replace
    them. They carry 2.76 analyses each and 77.5% of them hold more than one,
    which is grouping supervision nothing else here can supply: a target read
    off a table can only group by the banners the reader sees, and an analysis
    split by a column or named only in a footnote has no banner.

    Their captions, footnotes and serialised text are refreshed from the
    catalog -- 93.7% of the text changed and 1,591 gained a footnote when the
    re-extraction landed -- but their targets are left exactly as they were.
    Where the refreshed footnote appears to disagree with a target, the reader
    is at least as often the one at fault: of 13 statistic disagreements, most
    are `claimed_statistic` reading the axis letter in "X, Y and Z are
    coordinates in MNI-space" as a statistic name.
    """
    out: List[Example] = []
    for row in curated:
        target = row.get("target_json")
        if isinstance(target, str):
            target = json.loads(target)
        if not target:
            continue
        out.append(Example(
            table=row.get("table_serialised") or "",
            target=target,
            caption=row.get("caption") or "",
            footer=row.get("footer") or "",
            origin="curated",
            notes={"source": row.get("source"), "table_id": row.get("table_id"),
                   "article_id": row.get("article_id"),
                   "analyses": len(target.get("analyses") or [])},
        ))
    return out


def build_trainset(n: int = 4000, *, records: Optional[Iterable[Dict]] = None,
                   curated: Optional[Iterable[Dict]] = None,
                   weights: Weights = DEFAULT, seed: int = 0,
                   heuristic_cap: int = 1500,
                   real_positive_share: float = 0.15) -> List[Example]:
    """`n` examples that hold coordinates, plus empty ones on top.

    `n` counts the coordinate-bearing examples only, and the tables that hold
    nothing are added beside them at `weights.no_coordinates` of the total.
    Counting them inside `n` would have made v19 smaller than v18 on the thing
    both are trying to learn.

    Real examples are used before generated ones at every tier, so growing the
    set adds synthetic tables rather than diluting the real ones.
    """
    from .build import build                    # noqa: PLC0415
    from .empty import build_empty              # noqa: PLC0415

    # n is the coordinate half; the empty ones are a fraction of the whole.
    rate = min(max(weights.no_coordinates, 0.0), 0.9)
    want_empty = int(round(n * rate / (1.0 - rate))) if rate else 0
    real = _real_negatives(records or [], heuristic_cap=heuristic_cap, seed=seed)
    real = real[:want_empty]
    out: List[Example] = list(real)
    for i in range(want_empty - len(real)):
        t = build_empty(seed=seed * 100003 + i, weights=weights)
        out.append(Example(table=t.grid.render(), target=t.truth.as_target(),
                           caption=t.caption, footer=t.footer,
                           origin="generated", notes=t.notes))
    # Curated first, then real tables the reader reads whole, then generated.
    # Each tier is a worse answer than the one before it, so the generator only
    # ever makes up the shortfall.
    want_full = n
    have: List[Example] = curated_positives(curated or [])[:want_full]
    out.extend(have)
    if len(have) < want_full:
        room = want_full - len(have)
        reader_led = real_positives(records or [],
                                    limit=int(room * real_positive_share))
        out.extend(reader_led)
        have = have + reader_led
    for i in range(want_full - len(have)):
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
        if not empty:
            c["coords: " + ex.origin] += 1
            c["analyses in " + ex.origin] += len(ex.target.get("analyses") or [])
            c["multi-analysis " + ex.origin] += len(ex.target.get("analyses") or []) > 1
        if empty:
            c["empty: " + ex.origin] += 1
            if ex.origin == "generated":
                c["empty: generated " + str(ex.notes.get("kind", "?"))] += 1
    return dict(c)
