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
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

from .. import fields
from .weights import DEFAULT, Weights

logger = logging.getLogger(__name__)

EMPTY_TARGET: Dict = {"space": None, "analyses": []}


@dataclass
class Example:
    """One training example, whatever produced it."""

    table: str
    target: Dict
    caption: str = ""
    footer: str = ""
    title: str = ""
    abstract: str = ""
    origin: str = "generated"          # which of the four sources
    notes: Dict = field(default_factory=dict)

    def as_row(self) -> Dict:
        return {"table": self.table, "caption": self.caption,
                "footer": self.footer, "title": self.title,
                "abstract": self.abstract, "target": self.target,
                "origin": self.origin, "notes": self.notes}

    def as_training_row(self) -> Dict:
        """The shape the fine-tuning script reads.

        `target_json` is a string because that is what the model is trained to
        emit, and `n_tokens` is left to the caller, which has the tokenizer.
        """
        return {
            "article_id": self.notes.get("article_id") or "",
            "source": self.notes.get("source") or self.origin,
            "table_id": str(self.notes.get("table_id") or ""),
            "title": self.title,
            "abstract": self.abstract,
            "caption": self.caption,
            "footer": self.footer,
            "coordinate_space": self.target.get("space"),
            "table_serialised": self.table,
            "n_analyses": len(self.target.get("analyses") or []),
            "target_json": json.dumps(self.target, ensure_ascii=False,
                                      separators=(",", ":")),
            "origin": self.origin,
            "kind": self.notes.get("kind") or "coordinates",
        }


#: Tables read one by one and judged by eye, recorded beside the reader.
JUDGED = Path(__file__).resolve().parents[3] / "docs" / "judged-tables.json"


def judged_negatives(path=JUDGED) -> List[Example]:
    """The 143 tables read by eye and confirmed to hold no coordinates.

    These are the hard half of the negative set by construction: every one of
    them sat in the gate's uncertain band, or was a reader false positive, or
    was a table the heuristics placed one way and a reading placed the other.
    An F with its degrees of freedom, a genotype at `Chr6 | 168,336,080`, a
    voxel size written `3 x 3 x 3`. They are the tables the model is most
    likely to be handed and get wrong, and the generator can only guess at
    them.

    Their footers were not kept, so they train the empty target from the table
    and caption alone -- which is also the harder case.
    """
    try:
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("no hand-judged tables at %s", path)
        return []
    out: List[Example] = []
    for row in rows:
        if row.get("verdict") != "negative" or not row.get("text"):
            continue
        out.append(Example(
            table=row["text"], target=dict(EMPTY_TARGET),
            caption=row.get("caption") or "", footer="",
            origin="hand-judged",
            notes={"source": row.get("source"), "table_id": row.get("table_id"),
                   "article_id": row.get("article"), "shape": row.get("shape")}))
    return out


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
                     notes={"source": r.get("source"),
                            "table_id": r.get("table_id"),
                            "article_id": r.get("slug") or r.get("article_id")})
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
    # An analysis with no name is what the prompt falls back to when it cannot
    # attach coordinates to a labelled analysis, so training on one teaches the
    # model to abstain rather than to read a header. Where no banner precedes
    # the points, the caption names them -- and where there is no caption
    # either, nothing does, and the table is not usable as a positive.
    default = (caption or "").strip()
    analyses: List[Dict] = []
    for point in got.points:
        name = default
        for row, label in cuts:
            if row < point.row:
                name = label
        if not name:
            return None
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
                                  "article_id": r.get("slug") or r.get("article_id"),
                                  "analyses": len(target["analyses"])}))
        if limit and len(out) >= limit:
            break
    return out


#: Generous bounds on a human head in millimetres, from the reader.
_LIMITS = {0: 90.0, 1: 126.0, 2: 108.0}


def _looks_misaligned(target: Dict) -> bool:
    """Whether this target read its coordinates out of the wrong columns.

    A point outside a head is the symptom, not the fault. Reading the ten
    curated tables it flags shows the coordinates are fine and the target is
    not: `(-100, 14, 12)` is the row `CAL.R | 12 | -100 | 14` read as (y, z,
    x), and `(-94, -3, 24)` is a row whose AAL column is empty, so everything
    after it shifts one place left. The right calcarine really is at (12,
    -100, 14).

    So the example is dropped rather than the point. Keeping it and deleting
    the one coordinate that betrayed it would leave the rest of the target
    misaligned in exactly the same way, with nothing left to notice it by --
    and would throw away a real coordinate on the way.

    Ten of 10,015 curated examples. They are worth correcting, not guessing
    at: a permutation can be undone, a ragged-row shift needs the table.
    """
    for analysis in target.get("analyses") or []:
        for point in analysis.get("points") or []:
            if len(point) < 3:
                continue
            if any(isinstance(v, (int, float)) and abs(v) > _LIMITS[i]
                   for i, v in enumerate(point[:3])):
                return True
    return False


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
    misaligned: List[str] = []
    for row in curated:
        target = row.get("target_json")
        if isinstance(target, str):
            target = json.loads(target)
        if not target:
            continue
        if _looks_misaligned(target):
            misaligned.append(row.get("article_id"))
            continue
        out.append(Example(
            table=row.get("table_serialised") or "",
            target=target,
            caption=row.get("caption") or "",
            footer=row.get("footer") or "",
            title=row.get("title") or "",
            abstract=row.get("abstract") or "",
            origin="curated",
            notes={"source": row.get("source"), "table_id": row.get("table_id"),
                   "article_id": row.get("article_id"),
                   "analyses": len(target.get("analyses") or [])},
        ))
    if misaligned:
        logger.warning("%d curated targets read their coordinates out of the "
                       "wrong columns and were left out: %s",
                       len(misaligned), ", ".join(misaligned[:10]))
    return out


def add_context(examples: Iterable[Example], *, seed: int = 0,
                metadata: Optional[Dict[str, Sequence[str]]] = None
                ) -> List[Example]:
    """Give every example the title and abstract its paper would carry.

    Three sources, in the same order of preference as everything else here: the
    row's own, the article's from `metadata`, and -- only then -- composed from
    the table by `context`. See that module for why each rule is what it is.

    Every example gets one, including the ones whose answer is nothing. In the
    first v19 build the empty examples were the only rows without an abstract,
    which made 84.7% of the rows lacking one answerable without reading the
    table at all.
    """
    from . import context                       # noqa: PLC0415

    metadata = metadata or {}
    out: List[Example] = []
    for i, ex in enumerate(examples):
        if not ex.title:
            got = metadata.get(str(ex.notes.get("article_id") or ""))
            if got:
                ex.title, ex.abstract = (got[0] or ""), (ex.abstract or got[1] or "")
        if not ex.title or not ex.abstract:
            rng = random.Random(seed * 1000003 + i)
            title, abstract = context.title_and_abstract(
                rng, ex.table, ex.target, caption=ex.caption, footer=ex.footer)
            ex.title = ex.title or title
            ex.abstract = ex.abstract or abstract
        out.append(ex)
    return out


def set_space_from_what_is_visible(examples: Iterable[Example]) -> List[Example]:
    """The target's space is what the document states, and nothing else.

    Not article metadata: 53.5% of rows carry no visible cue, so a
    metadata-derived target teaches the model to invent a space on half its
    examples. Null when nothing states it, and null when two parts of the
    document disagree.

    A target with no analyses keeps `space: null` whatever its prose mentions.
    A table stating no coordinates states no space for them -- and the
    template-comparison negatives are headed `MNI-305` and `ICBM-152`, so the
    rule would otherwise put a space on the very tables that exist to teach
    the model to answer nothing.
    """
    from . import context                       # noqa: PLC0415

    out: List[Example] = []
    for ex in examples:
        if ex.target.get("analyses"):
            space = context.visible_space(ex.abstract, ex.caption, ex.footer,
                                          ex.table)
        else:
            space = None
        ex.target = {"space": space,
                     **{k: v for k, v in ex.target.items() if k != "space"}}
        out.append(ex)
    return out


def build_trainset(n: int = 4000, *, records: Optional[Iterable[Dict]] = None,
                   curated: Optional[Iterable[Dict]] = None,
                   weights: Weights = DEFAULT, seed: int = 0,
                   heuristic_cap: int = 1500,
                   metadata: Optional[Dict[str, Sequence[str]]] = None,
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
    # The tables read by eye come first: they are the ones the heuristics
    # could not place, so nothing else in the set covers them.
    real = judged_negatives()
    seen = {(e.notes.get("article_id"), e.notes.get("table_id")) for e in real}
    real += [e for e in _real_negatives(records or [], heuristic_cap=heuristic_cap,
                                        seed=seed)
             if (e.notes.get("article_id"), e.notes.get("table_id")) not in seen]
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
    out = add_context(out, seed=seed, metadata=metadata)
    return set_space_from_what_is_visible(out)


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


# -- the statistic a table says it reports -------------------------------

#: Re-exported so callers that read a whole document do not reach past it.
STATISTIC_PRIORITY = fields.STATISTIC_PRIORITY

# The coordinate run is masked before the header is read, because the `z` of
# `#x | #y | #z` is a coordinate and matching it was how the first measurement
# of this problem came out twice as bad as it is.
_XYZ_RUN = re.compile(r"#?\s*x\s*\|\s*#?\s*y\s*\|\s*#?\s*z\b", re.I)
_CELL = re.compile(r"[|\n]")


def statistic_named_by(table: str, caption: str = "", footer: str = "") -> Optional[str]:
    """The statistic the document reports, by priority when it names several.

    Every cell of the first three lines is put to `fields.statistic_type`, the
    one reader of this rule, and the caption and footnote are read the same way
    -- a table often puts the letter in its header and spells it out underneath.

    A table printing both a t and a p prints one test statistic and one
    significance level. Declining there threw the answer away on the commonest
    multi-statistic shape there is, so `STATISTIC_PRIORITY` resolves them.
    """
    head = _XYZ_RUN.sub(" coord ", "\n".join((table or "").split("\n")[:3]))
    hits = {fields.statistic_type(cell) for cell in _CELL.split(head)}
    hits.discard(None)
    if not hits:
        context = " ".join((caption or "", footer or ""))
        hits = {fields.statistic_type(part) for part in context.split(".")}
        hits.discard(None)
    return fields.best_of(hits)


def correct_statistic_types(examples: Iterable[Example]) -> List[Example]:
    """Relabel a point's statistic to the one its own table names.

    The curated targets come from luna, and luna does not read the statistic
    column: over 200 of its cached parses, 199 answer `T`, agreeing with the
    header 17% of the time. That went into training unexamined -- 72% of
    curated rows whose document names a statistic disagree with it, 1,071 of
    them calling a Z a T -- and v19 learned it, mislabelling 94% of Z tables
    in production.

    Only the *name* changes. The value, the coordinates and the grouping are
    luna's to keep; this is the one field it does not look at.

    Conservative on purpose:

    * a document naming two statistics is left alone -- no single answer is
      right for a table reporting both;
    * a point with no statistic type stays without one, because a missing
      label is not a wrong one and inventing it would assert what the model
      cannot see;
    * generated examples are untouched. Their targets already agree with their
      tables 2,091 times out of 2,091: the generator writes both.
    """
    out: List[Example] = []
    changed = kept = 0
    for example in examples:
        named = statistic_named_by(example.table, example.caption, example.footer)
        if named is None or example.origin == "generated":
            out.append(example)
            continue
        touched = False
        for analysis in (example.target.get("analyses") or []):
            for point in (analysis.get("points") or []):
                if not isinstance(point, list) or len(point) < 4:
                    continue
                if point[3] and point[3] != named:
                    point[3] = named
                    touched = True
                    changed += 1
                elif point[3]:
                    kept += 1
        if touched:
            example.notes = dict(example.notes or {})
            example.notes["statistic_relabelled"] = named
        out.append(example)
    logger.info("statistic types: %d relabelled from the document, %d already agreed",
                changed, kept)
    return out
