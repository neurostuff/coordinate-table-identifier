"""Labels without hand review, and the splits that keep them honest.

Positives come from the extractor having found coordinates, or from the
deterministic reader locating three. Negatives come from `detect`, which
requires both structural and topical evidence before it will call a table
empty. The UNCERTAIN band is excluded from training on purpose -- it is the
population the gate exists to sort, so scoring against it would be circular.

Splitting is by *article*, not by table. A paper's tables share a publisher, a
house style and often a caption prefix; splitting by table leaks all of that
across the boundary and flatters the score.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from .. import detect
from ..detect import Verdict
from . import features


def corroborated(vec: Dict[str, float]) -> bool:
    """Whether anything but the extractor says this table holds coordinates.

    A coordinate word in a header, anatomy in the row labels, or a triple the
    reader can find. Any one will do; the point is only that the extractor is
    not the sole witness.
    """
    return bool(vec.get("coord_words_header") or vec.get("region_words_first_column")
                or vec.get("reader_points"))


def label_of(row: Dict, vec: Optional[Dict[str, float]] = None) -> Optional[int]:
    """1, 0, or None for a row that should not be trained on.

    A table the heuristic could not place counts as positive only when the
    extractor is corroborated. Taking its word alone is circular -- its
    mistakes are what this pipeline exists to correct -- and of the seven
    tables it claimed with nothing else agreeing, five hold no coordinates:
    node positions in percent, a literature summary, two prose tables and a
    set of Mann-Whitney statistics whose `(33, 33, 0.051118)` read as a peak.
    Two real ones go with them, which is cheap against five wrong labels.
    """
    verdict = row.get("label")
    if verdict == Verdict.POSITIVE.value:
        return 1
    if verdict == Verdict.NEGATIVE.value:
        return 0
    if row.get("extractor_found"):
        if vec is None:
            vec = features.vector(row.get("table_serialised") or "",
                                  row.get("caption") or "", row.get("footer") or "")
        return 1 if corroborated(vec) else None
    return None


def rows_from_jsonl(path) -> Iterator[Dict]:
    for line in Path(path).read_text().split("\n"):
        if line.strip():
            yield json.loads(line)


CANDIDATES = "candidates"   # the reader read a triple out of these
RESIDUAL = "residual"       # it read nothing in these
EVERYTHING = "all"


def population_of(vec: Dict[str, float]) -> str:
    """Which gate decides this table.

    The reader's own output routes, and it needs no label to do so, which is
    what makes two gates possible at inference and not only in training.
    """
    return CANDIDATES if vec["reader_points"] > 0 else RESIDUAL


def build(records: Sequence[Dict], *, population: str = EVERYTHING
          ) -> Tuple[List[List[float]], List[int], List[str]]:
    """Feature rows, labels, and the article each came from.

    `population` picks which gate's training set to build. Fitting on only the
    tables the reader gets wrong was tried and is the worst of the three: every
    table the reader read is a negative there by construction, so the gate
    learns that a reader hit means the table is false, which is backwards at
    inference. It costs 19 points of recall on the residual.
    """
    x, y, groups = [], [], []
    for r in records:
        vec = features.vector(r.get("table_serialised") or "",
                              r.get("caption") or "", r.get("footer") or "")
        label = label_of(r, vec)
        if label is None:
            continue
        if population != EVERYTHING and population_of(vec) != population:
            continue
        x.append([vec[n] for n in features.NAMES])
        y.append(label)
        groups.append(str(r.get("slug") or r.get("article_id") or len(groups)))
    return x, y, groups


def from_jsonl(path, *, population: str = EVERYTHING
               ) -> Tuple[List[List[float]], List[int], List[str]]:
    return build(list(rows_from_jsonl(path)), population=population)


def split_by_article(x, y, groups, *, holdout: float = 0.25, seed: int = 0):
    """Train/test split with every table of an article on one side."""
    articles = sorted(set(groups))
    rng = random.Random(seed)
    rng.shuffle(articles)
    cut = int(len(articles) * (1 - holdout))
    train_articles = set(articles[:cut])
    idx_tr = [i for i, g in enumerate(groups) if g in train_articles]
    idx_te = [i for i, g in enumerate(groups) if g not in train_articles]
    pick = lambda idx: ([x[i] for i in idx], [y[i] for i in idx])
    return pick(idx_tr), pick(idx_te)


def summarise(y: Sequence[int]) -> Dict[str, float]:
    n = len(y) or 1
    pos = sum(y)
    return {"n": len(y), "positive": pos, "negative": len(y) - pos,
            "positive_rate": pos / n}
