"""A high-recall gate over `features.NAMES`.

Not a balanced classifier. `create_analyses` drops a table with no coordinates,
so a false negative loses the article permanently while a false positive costs
one call that returns nothing. The threshold is therefore chosen as *the lowest
score that still meets a precision floor*, and the number reported is recall at
that point -- not accuracy, which a 90%-negative corpus makes meaningless.

Logistic regression, fitted by gradient descent on standardised features. No
sklearn dependency: the problem is near-linearly separable and a linear model is
also readable, which matters more here than the last point of AUC. `sklearn_fit`
is provided for when you want to compare against gradient boosting.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from . import features


def _standardise(rows: Sequence[Sequence[float]]):
    n = len(rows[0])
    mean = [sum(r[i] for r in rows) / len(rows) for i in range(n)]
    var = [sum((r[i] - mean[i]) ** 2 for r in rows) / max(len(rows) - 1, 1)
           for i in range(n)]
    std = [math.sqrt(v) if v > 1e-12 else 1.0 for v in var]
    return mean, std


def _apply(row: Sequence[float], mean, std) -> List[float]:
    return [(row[i] - mean[i]) / std[i] for i in range(len(row))]


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass
class Gate:
    """A fitted gate: weights, the standardisation, and a chosen threshold."""

    weights: List[float] = field(default_factory=list)
    bias: float = 0.0
    mean: List[float] = field(default_factory=list)
    std: List[float] = field(default_factory=list)
    names: List[str] = field(default_factory=lambda: list(features.NAMES))
    threshold: float = 0.5
    precision_floor: float = 0.90
    metrics: Dict[str, float] = field(default_factory=dict)

    # -- use ---------------------------------------------------------------
    def score(self, text_or_grid, caption: str = "", footer: str = "") -> float:
        row = [features.vector(text_or_grid, caption, footer)[n] for n in self.names]
        return self.score_row(row)

    def score_row(self, row: Sequence[float]) -> float:
        z = self.bias + sum(w * v for w, v in
                            zip(self.weights, _apply(row, self.mean, self.std)))
        return _sigmoid(z)

    def predict(self, text_or_grid, caption: str = "", footer: str = "") -> bool:
        return self.score(text_or_grid, caption, footer) >= self.threshold

    def explain(self, text_or_grid, caption: str = "", footer: str = "",
                top: int = 6) -> List[Tuple[str, float]]:
        """The features pushing this decision hardest, signed."""
        row = [features.vector(text_or_grid, caption, footer)[n] for n in self.names]
        z = _apply(row, self.mean, self.std)
        contrib = [(self.names[i], self.weights[i] * z[i]) for i in range(len(z))]
        contrib.sort(key=lambda kv: -abs(kv[1]))
        return contrib[:top]

    # -- persistence -------------------------------------------------------
    def save(self, path) -> None:
        Path(path).write_text(json.dumps({
            "weights": self.weights, "bias": self.bias, "mean": self.mean,
            "std": self.std, "names": self.names, "threshold": self.threshold,
            "precision_floor": self.precision_floor, "metrics": self.metrics,
        }, indent=1))

    @classmethod
    def load(cls, path) -> "Gate":
        d = json.loads(Path(path).read_text())
        return cls(**d)


def fit(rows: Sequence[Sequence[float]], labels: Sequence[int], *,
        epochs: int = 400, lr: float = 0.5, l2: float = 1e-3,
        precision_floor: float = 0.90,
        names: Optional[Sequence[str]] = None) -> Gate:
    """Fit the gate, then pick the threshold that meets the precision floor.

    Class weighting is by inverse frequency, because the corpus is roughly 90%
    negative and an unweighted fit simply predicts "no" and scores well.
    """
    assert rows and len(rows) == len(labels)
    mean, std = _standardise(rows)
    x = [_apply(r, mean, std) for r in rows]
    n_feat = len(x[0])
    pos = sum(labels) or 1
    neg = len(labels) - pos or 1
    w_pos, w_neg = len(labels) / (2.0 * pos), len(labels) / (2.0 * neg)

    w = [0.0] * n_feat
    b = 0.0
    for _ in range(epochs):
        gw = [0.0] * n_feat
        gb = 0.0
        for xi, yi in zip(x, labels):
            p = _sigmoid(b + sum(w[j] * xi[j] for j in range(n_feat)))
            cw = w_pos if yi else w_neg
            err = cw * (p - yi)
            gb += err
            for j in range(n_feat):
                gw[j] += err * xi[j]
        m = len(x)
        b -= lr * gb / m
        for j in range(n_feat):
            w[j] -= lr * (gw[j] / m + l2 * w[j])

    gate = Gate(weights=w, bias=b, mean=mean, std=std,
                names=list(names or features.NAMES),
                precision_floor=precision_floor)
    scores = [gate.score_row(r) for r in rows]
    gate.threshold, gate.metrics = choose_threshold(
        scores, labels, precision_floor=precision_floor)
    return gate


def choose_threshold(scores: Sequence[float], labels: Sequence[int], *,
                     precision_floor: float = 0.90) -> Tuple[float, Dict[str, float]]:
    """The lowest threshold whose precision still clears the floor.

    Lowest, not best: every step down recovers tables that would otherwise be
    dropped for good, and the only thing bounding the descent is how much wasted
    downstream work the precision floor allows.
    """
    order = sorted(set(scores))
    best = (1.0, {"precision": 1.0, "recall": 0.0, "f1": 0.0, "n_pos": sum(labels)})
    for t in order:
        tp = sum(1 for s, y in zip(scores, labels) if s >= t and y)
        fp = sum(1 for s, y in zip(scores, labels) if s >= t and not y)
        fn = sum(1 for s, y in zip(scores, labels) if s < t and y)
        prec = tp / (tp + fp) if tp + fp else 1.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        if prec >= precision_floor and rec > best[1]["recall"]:
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            best = (t, {"precision": prec, "recall": rec, "f1": f1,
                        "tp": tp, "fp": fp, "fn": fn, "n_pos": sum(labels)})
    return best


def evaluate(gate: Gate, rows: Sequence[Sequence[float]],
             labels: Sequence[int]) -> Dict[str, float]:
    """Recall at the gate's threshold, and what it costs downstream."""
    scores = [gate.score_row(r) for r in rows]
    tp = sum(1 for s, y in zip(scores, labels) if s >= gate.threshold and y)
    fp = sum(1 for s, y in zip(scores, labels) if s >= gate.threshold and not y)
    fn = sum(1 for s, y in zip(scores, labels) if s < gate.threshold and y)
    tn = len(labels) - tp - fp - fn
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn,
            "tn": tn, "n": len(labels),
            # what the gate actually buys: how much of the corpus it lets through
            "pass_rate": (tp + fp) / max(len(labels), 1)}


def sklearn_fit(rows, labels, **kwargs):
    """Gradient boosting, when you want to check the linear model is not leaving
    much on the table. Optional import so the package stays dependency-free."""
    from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: PLC0415

    clf = HistGradientBoostingClassifier(**kwargs)
    clf.fit(rows, labels)
    return clf
