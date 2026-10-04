"""A high-recall gate over `features.NAMES`.

Not a balanced classifier. `create_analyses` drops a table with no coordinates,
so a false negative loses the article permanently while a false positive costs
one call that returns nothing. The threshold is therefore chosen as *the lowest
score that still meets a precision floor*, and the number reported is recall at
that point -- not accuracy, which a 90%-negative corpus makes meaningless.

Two model classes, because the two populations are not alike. Where the reader
read a triple, the question is near-linearly separable and `Gate`, a logistic
regression fitted here by gradient descent, reaches 100% recall at a 95%
precision floor with no variance and no dependency. Where it read nothing, the
same model reaches 56.9% with a standard deviation of 30, and a random forest
on identical features reaches 80.3% with a standard deviation of 9. The signal
there is in combinations -- anatomy in the row labels *and* a coordinate word
in a header *and* numbers that fit in a head -- which a sum of weights cannot
express. `Forest` is for that side.

It is not a shortage of labels: the learning curve on the residual is flat from
a quarter of the articles, and 209 hand judgments folded in moved recall by
-2.7 points.
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


@dataclass
class RoutedGate:
    """Two gates, routed on whether the reader read a triple out of the table.

    The two populations want different operating points, and a single gate
    cannot hold both. Where the reader read something, a wrong answer costs one
    model call that comes back empty, so that gate runs for precision. Where it
    read nothing, a wrong answer loses the article for good, because a table
    dropped here is never looked at again, so that gate runs for recall.

    Both reach 100% recall at a 95% precision floor on the tables the reader
    reads; they differ only on the residual, where the pair averages 59.6%
    against 54.0% for one gate over ten article splits. The spread is wide --
    there are 40 residual positives in the whole label set -- so the case for
    the pair rests on the two thresholds, not on that gap.
    """

    candidates: Gate = field(default_factory=Gate)
    residual: object = field(default_factory=Gate)

    def gate_for(self, text_or_grid, caption: str = "", footer: str = "") -> Gate:
        """The one gate that decides this table.

        A partition, not a pipeline. A table the candidate gate rejects is
        rejected, and is never put to the residual gate afterwards. Letting it
        fall through would undo the decision that was just made, and the
        residual model could not be trusted to make it anyway: it is fitted
        only on tables where the reader found nothing, so on a table where the
        reader found something, `reader_points`, `reader_by_packed` and
        `frac_rows_read` all carry values it never saw.
        """
        vec = features.vector(text_or_grid, caption, footer)
        return self.candidates if vec["reader_points"] > 0 else self.residual

    def decide(self, text_or_grid, caption: str = "", footer: str = "") -> Dict:
        """The verdict, and everything needed to know when it goes stale.

        A verdict is not permanent, because the route is not. It is computed
        from the reader's output, and the reader changes: a table that was
        residual last month is a candidate once the reader learns to parse the
        form it is written in, and a different gate then decides it. So the
        verdict is recorded with the route that produced it, and is recomputed
        when the reader or either gate changes -- not on every pass, which
        would let a table's fate drift silently.
        """
        vec = features.vector(text_or_grid, caption, footer)
        route = "candidates" if vec["reader_points"] > 0 else "residual"
        gate = getattr(self, route)
        score = gate.score_row([vec[n] for n in gate.names])
        return {"passes": score >= gate.threshold, "score": score,
                "route": route, "threshold": gate.threshold,
                "reader_points": vec["reader_points"]}

    def score(self, text_or_grid, caption: str = "", footer: str = "") -> float:
        """Careful: two gates, two scales. Compare with `predict`, not across."""
        return self.gate_for(text_or_grid, caption, footer).score(
            text_or_grid, caption, footer)

    def predict(self, text_or_grid, caption: str = "", footer: str = "") -> bool:
        return self.gate_for(text_or_grid, caption, footer).predict(
            text_or_grid, caption, footer)

    def explain(self, text_or_grid, caption: str = "", footer: str = "",
                top: int = 6) -> List[Tuple[str, float]]:
        return self.gate_for(text_or_grid, caption, footer).explain(
            text_or_grid, caption, footer, top=top)

    def save(self, path) -> None:
        """Both halves in one file.

        A forest cannot be written as JSON, so a pair holding one is saved with
        joblib and a pair of logistic gates stays JSON. The file says which it
        is, so `load` does not have to guess from the extension.
        """
        if isinstance(self.residual, Forest) or isinstance(self.candidates, Forest):
            import joblib  # noqa: PLC0415

            joblib.dump({"kind": "routed-forest",
                         "candidates": _part(self.candidates),
                         "residual": _part(self.residual)}, str(path))
            return
        Path(path).write_text(json.dumps({
            "kind": "routed",
            "candidates": json.loads(_dumps(self.candidates)),
            "residual": json.loads(_dumps(self.residual)),
        }, indent=1))

    @classmethod
    def load(cls, path) -> "RoutedGate":
        try:
            d = json.loads(Path(path).read_text())
        except (UnicodeDecodeError, ValueError):
            import joblib  # noqa: PLC0415

            d = joblib.load(str(path))
            return cls(candidates=_unpart(d["candidates"]), residual=_unpart(d["residual"]))
        return cls(candidates=Gate(**d["candidates"]),
                   residual=Gate(**{k: v for k, v in d["residual"].items()}))


def _part(gate) -> Dict:
    """One side of a pair as a joblib payload: a forest keeps its estimator."""
    if isinstance(gate, Forest):
        return {"clf": gate.clf, "names": gate.names, "threshold": gate.threshold,
                "precision_floor": gate.precision_floor, "metrics": gate.metrics}
    return json.loads(_dumps(gate))


def _unpart(d: Dict):
    return Forest(**d) if "clf" in d else Gate(**d)


def _dumps(gate: Gate) -> str:
    return json.dumps({
        "weights": gate.weights, "bias": gate.bias, "mean": gate.mean,
        "std": gate.std, "names": gate.names, "threshold": gate.threshold,
        "precision_floor": gate.precision_floor, "metrics": gate.metrics})


def fit_routed(records, *, candidate_floor: float = 0.95,
               residual_floor: float = 0.0,
               candidate_recall: float = 0.99,
               residual_recall: float = 0.97, epochs: int = 400,
               forest: bool = True, candidate_forest: bool = False) -> RoutedGate:
    """Fit both gates from one set of labelled records.

    The floors differ on purpose, and so do the model classes: see
    `RoutedGate`.

    The residual side has **no precision floor**. Asking it for both 99%
    recall and 90% precision asks for a point that does not exist -- 3.0% of
    the tables it sees hold coordinates, and at 97% recall it runs at 54%
    precision, at 100% recall at 21%. A floor that cannot be met is not a
    safeguard; it silently hands back whatever threshold the search settled
    on, which was 0.694 and 56% recall. The floor that is meant here is the
    recall one, and it is stated as such: keep 97% of them and let the
    extraction model deal with what comes through.

    Pass `forest=False` to fit the residual side with the same logistic
    regression as the candidate side, at a cost of 23 points of recall, when
    sklearn is not available.

    `candidate_forest=True` fits the candidate side as a forest too, its
    threshold chosen out of fold for `candidate_recall` with no precision
    floor. The logistic candidate gate could not hold 99% recall held out by
    article once the packed-cell tables it used to reject were labelled: those
    tables differ from interval tables (`1.06 (1.02-1.11)`) only in how their
    features combine, which a linear model cannot express.
    """
    from . import dataset
    xc, yc, _ = dataset.build(records, population=dataset.CANDIDATES)
    xr, yr, _ = dataset.build(records, population=dataset.RESIDUAL)
    residual = (fit_forest(xr, yr, precision_floor=residual_floor,
                           recall_floor=residual_recall) if forest
                else fit(xr, yr, epochs=epochs, precision_floor=residual_floor,
                         recall_floor=residual_recall))
    candidates = (fit_forest(xc, yc, precision_floor=0.0, recall_floor=candidate_recall)
                  if candidate_forest else
                  fit(xc, yc, epochs=epochs, precision_floor=candidate_floor,
                      recall_floor=candidate_recall))
    return RoutedGate(candidates=candidates, residual=residual)


def fit(rows: Sequence[Sequence[float]], labels: Sequence[int], *,
        epochs: int = 400, lr: float = 0.5, l2: float = 1e-3,
        precision_floor: float = 0.90, recall_floor: float = 0.99,
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
        scores, labels, precision_floor=precision_floor,
        recall_floor=recall_floor)
    return gate


def _midpoints(order: Sequence[float]) -> List[float]:
    """Halfway between each pair of observed scores, and just past the ends.

    A threshold equal to an observed score sits on top of the example that
    produced it, so the margin on that side is nothing: the residual gate's
    lowest positive scores 0.9263 and a threshold of 0.9263 keeps it by
    exactly zero. With 34 positives to go on, the next table to score 0.92
    would be dropped. Halfway between the modes leaves room on both sides.
    """
    if not order:
        return [0.5]
    out = [order[0] * 0.5]
    out += [(a + b) / 2.0 for a, b in zip(order, order[1:])]
    out.append(min(1.0, order[-1] + (1.0 - order[-1]) / 2.0))
    return out


def choose_threshold(scores: Sequence[float], labels: Sequence[int], *,
                     precision_floor: float = 0.90,
                     recall_floor: float = 0.99) -> Tuple[float, Dict[str, float]]:
    """The highest threshold that still keeps `recall_floor` of the positives.

    Not the lowest that clears a precision floor, which is what this used to
    do. That reads as the generous choice and is not, because precision is
    measured on the population the gate was fitted on and the threshold is
    used on a different one.

    The candidate half of the label set is 83.5% positive. A threshold low
    enough to admit most of the known negatives still scores 91% precision
    there, so the floor is met at 0.0009 -- below the negatives' own median.
    In the corpus, where far fewer of the tables a reader reads hold
    coordinates, that threshold lets nearly everything through: of eight
    tables it newly admitted, seven were miRNA chromosome locations, sample
    sizes, quality-of-life ranges and Stata command options.

    The scores are strongly bimodal and that is what to use. Tables that hold
    coordinates score above 0.95 at the tenth percentile; tables that do not
    score below 0.04 at the ninetieth. Nothing lives in between, so the
    threshold belongs in that gap, at the highest point that keeps essentially
    every positive, and halfway between observed scores rather than on one of
    them so that neither side is decided by a single example. The precision
    floor stays as a bound in case the two modes ever overlap.
    """
    order = sorted(set(scores))
    n_pos = sum(labels)
    best = (order[0] if order else 0.5,
            {"precision": 0.0, "recall": 1.0, "f1": 0.0, "n_pos": n_pos})
    for t in _midpoints(order):
        tp = sum(1 for s, y in zip(scores, labels) if s >= t and y)
        fp = sum(1 for s, y in zip(scores, labels) if s >= t and not y)
        fn = sum(1 for s, y in zip(scores, labels) if s < t and y)
        prec = tp / (tp + fp) if tp + fp else 1.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        if rec >= recall_floor and prec >= precision_floor:
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            best = (t, {"precision": prec, "recall": rec, "f1": f1,
                        "tp": tp, "fp": fp, "fn": fn, "n_pos": n_pos})
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


@dataclass
class Forest:
    """A random forest with the same surface as `Gate`.

    Kept behind an optional import: only the residual side needs it.
    """

    clf: object = None
    names: List[str] = field(default_factory=lambda: list(features.NAMES))
    threshold: float = 0.5
    precision_floor: float = 0.90
    metrics: Dict[str, float] = field(default_factory=dict)

    def score(self, text_or_grid, caption: str = "", footer: str = "") -> float:
        return self.score_row(
            [features.vector(text_or_grid, caption, footer)[n] for n in self.names])

    def score_row(self, row: Sequence[float]) -> float:
        return float(self.clf.predict_proba([list(row)])[0][1])

    def predict(self, text_or_grid, caption: str = "", footer: str = "") -> bool:
        return self.score(text_or_grid, caption, footer) >= self.threshold

    def explain(self, text_or_grid, caption: str = "", footer: str = "",
                top: int = 6) -> List[Tuple[str, float]]:
        """A forest has no per-table weights, so this reports the features it
        splits on most across the whole model, not this table's own reasons."""
        imp = sorted(zip(self.names, self.clf.feature_importances_),
                     key=lambda kv: -kv[1])
        return [(n, float(v)) for n, v in imp[:top]]

    def save(self, path) -> None:
        import joblib  # noqa: PLC0415
        joblib.dump({"clf": self.clf, "names": self.names,
                     "threshold": self.threshold,
                     "precision_floor": self.precision_floor,
                     "metrics": self.metrics}, str(path))

    @classmethod
    def load(cls, path) -> "Forest":
        import joblib  # noqa: PLC0415
        return cls(**joblib.load(str(path)))


def fit_forest(rows: Sequence[Sequence[float]], labels: Sequence[int], *,
               precision_floor: float = 0.90, recall_floor: float = 0.99,
               n_estimators: int = 400,
               min_samples_leaf: int = 2, seed: int = 0, n_jobs: int = -1,
               names: Optional[Sequence[str]] = None) -> Forest:
    """Fit the forest on everything, and pick the threshold out of fold.

    A forest scores the rows it was grown on at almost 0 or almost 1, because
    it has memorised them. A threshold read off those scores describes nothing
    and is met by any cut in the gap: 0.694 looked like 99% recall and 100%
    precision that way, and delivered **55.9%** recall on tables the forest had
    not seen -- 15 of 34 positives dropped, on the side of the gate where a
    dropped table is never looked at again.

    So the threshold is chosen from cross-validated scores, which are the only
    ones that say what this forest does to a table it has not met. The forest
    that ships is still fitted on everything; only the cut comes from the folds.

    `n_jobs` is exposed so a caller fitting many forests at once can give each
    one core: sixty forests each grabbing every core is slower than sixty
    forests on one apiece.
    """
    from sklearn.ensemble import RandomForestClassifier  # noqa: PLC0415
    from sklearn.model_selection import StratifiedKFold  # noqa: PLC0415

    rows = [list(r) for r in rows]
    labels = list(labels)

    def _new():
        return RandomForestClassifier(n_estimators=n_estimators,
                                      min_samples_leaf=min_samples_leaf,
                                      class_weight="balanced",
                                      random_state=seed, n_jobs=n_jobs)

    clf = _new().fit(rows, labels)
    forest = Forest(clf=clf, names=list(names or features.NAMES),
                    precision_floor=precision_floor)

    folds = min(5, sum(labels), len(labels) - sum(labels))
    if folds < 2:
        # Too few of one class to hold any out. Fall back to the in-sample
        # scores and say so in the metrics, rather than pretending.
        scores = [forest.score_row(r) for r in rows]
        out_of_fold = False
    else:
        scores = [0.0] * len(rows)
        splitter = StratifiedKFold(folds, shuffle=True, random_state=seed)
        for train, test in splitter.split(rows, labels):
            held = _new().fit([rows[i] for i in train], [labels[i] for i in train])
            for i, p in zip(test, held.predict_proba([rows[i] for i in test])):
                scores[i] = float(p[1])
        out_of_fold = True

    forest.threshold, forest.metrics = choose_threshold(
        scores, labels, precision_floor=precision_floor,
        recall_floor=recall_floor)
    forest.metrics["out_of_fold"] = out_of_fold
    return forest


def sklearn_fit(rows, labels, **kwargs):
    """Gradient boosting, when you want to check the linear model is not leaving
    much on the table. Optional import so the package stays dependency-free."""
    from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: PLC0415

    clf = HistGradientBoostingClassifier(**kwargs)
    clf.fit(rows, labels)
    return clf
