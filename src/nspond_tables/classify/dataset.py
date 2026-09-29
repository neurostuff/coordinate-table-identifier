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


def label_of(row: Dict) -> Optional[int]:
    """1, 0, or None for a row that should not be trained on."""
    if row.get("extractor_found"):
        return 1
    verdict = row.get("label")
    if verdict == Verdict.POSITIVE.value:
        return 1
    if verdict == Verdict.NEGATIVE.value:
        return 0
    return None


def rows_from_jsonl(path) -> Iterator[Dict]:
    for line in Path(path).read_text().split("\n"):
        if line.strip():
            yield json.loads(line)


def is_hard(vec: Dict[str, float], label: int) -> bool:
    """Whether the reader leaves this table in question.

    The reader is wrong in exactly two ways. It reads nothing in a table that
    holds coordinates -- the residual -- and it reads a triple out of one that
    holds none, because `1.5 (0.3-7.8)` and `58 (43-63)` parse as triples. Both
    are the gate's to settle. A table the reader reads correctly settles
    itself, and training on those costs recall where it matters: a gate fitted
    on every table reaches 35.7% recall on the residual at a 95% precision
    floor, against 85.7% for one fitted on the hard cases alone, while both
    stay at 100% on the tables the reader reads.
    """
    return vec["reader_points"] == 0 or label == 0


def build(records: Sequence[Dict], *, hard_only: bool = True
          ) -> Tuple[List[List[float]], List[int], List[str]]:
    """Feature rows, labels, and the article each came from.

    Only the tables the reader leaves in question, unless `hard_only` is off.
    """
    x, y, groups = [], [], []
    for r in records:
        label = label_of(r)
        if label is None:
            continue
        vec = features.vector(r.get("table_serialised") or "",
                              r.get("caption") or "", r.get("footer") or "")
        if hard_only and not is_hard(vec, label):
            continue
        x.append([vec[n] for n in features.NAMES])
        y.append(label)
        groups.append(str(r.get("slug") or r.get("article_id") or len(groups)))
    return x, y, groups


def from_jsonl(path, *, hard_only: bool = True
               ) -> Tuple[List[List[float]], List[int], List[str]]:
    return build(list(rows_from_jsonl(path)), hard_only=hard_only)


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
