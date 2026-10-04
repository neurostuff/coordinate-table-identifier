"""Fit a routed gate and report recall where it matters -- features computed once, in parallel.

    harness2.py LABELS.jsonl --hand HAND.json [--residual-recall 0.99] [--candidate-recall 0.99]
                [--candidate-logistic] [--save OUT] [--dump OOF.json] [--baseline v5.joblib]

Every table's feature vector is computed once, across all cores, and cached by
content (the cache is per feature list, so a changed feature set misses it).
Folds, the final fit and every evaluation then reuse the same matrix.

Readings, all on tables the fitted gate never trained on:
  * out of fold, held out by ARTICLE, at the thresholds each fold's own fit chose;
  * the hand-labelled test half, weighted back to the corpus by stratum;
  * tables an independent parser read 3+ coordinates out of (recall only).
"""
import argparse, collections, hashlib, json, os, pickle
from multiprocessing import Pool
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from nspond_tables.classify import RoutedGate, dataset, features, model

ap = argparse.ArgumentParser()
ap.add_argument("labels")
ap.add_argument("--hand", required=True)
ap.add_argument("--parser", default="/data/alejandro/jk-gate/parser_pos.json")
ap.add_argument("--residual-recall", type=float, default=0.99)
ap.add_argument("--candidate-recall", type=float, default=0.99)
ap.add_argument("--candidate-floor", type=float, default=0.95)
ap.add_argument("--candidate-logistic", action="store_true")
ap.add_argument("--save"); ap.add_argument("--dump"); ap.add_argument("--baseline")
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--workers", type=int, default=40)
a = ap.parse_args()

NAMES = features.NAMES
CACHE = "/data/alejandro/jk-gate/featcache-%s.pkl" % hashlib.md5(",".join(NAMES).encode()).hexdigest()[:10]


def key(t, c, f):
    return hashlib.md5(("\x00".join([t or "", c or "", f or ""])).encode()).hexdigest()


def vec(args):
    t, c, f = args
    return key(t, c, f), features.vector(t or "", c or "", f or "")


cache = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}


def vectors(items):
    todo = list({key(*i): i for i in items if key(*i) not in cache}.values())
    if todo:
        with Pool(a.workers) as pool:
            for k, v in pool.imap_unordered(vec, todo, chunksize=8):
                cache[k] = v
        pickle.dump(cache, open(CACHE, "wb"))
    return [cache[key(*i)] for i in items]


# ---- the label set
recs = [json.loads(l) for l in open(a.labels) if l.strip()]
hand = json.load(open(a.hand))
for h in hand:
    if h["split"] == "train" and h["verdict"] in ("positive", "negative"):
        recs.append({"article_id": h["article_id"], "source": h["source"], "table_id": h["table_id"],
                     "table_serialised": h["text"], "caption": h["caption"], "footer": h["footer"],
                     "label": h["verdict"], "hand_judged": True})
V = vectors([(r.get("table_serialised"), r.get("caption"), r.get("footer")) for r in recs])
keep = [i for i in range(len(recs)) if dataset.label_of(recs[i], V[i]) is not None]
X = np.array([[V[i][n] for n in NAMES] for i in keep])
y = np.array([dataset.label_of(recs[i], V[i]) for i in keep])
route = np.array([int(V[i]["reader_points"] > 0) for i in keep])
groups = np.array([str(recs[i].get("article_id") or recs[i].get("slug")) for i in keep])
R = [recs[i] for i in keep]
print("label set: %d tables, %d articles | candidates %d (%d pos) | residual %d (%d pos)"
      % (len(y), len(set(groups)), (route == 1).sum(), y[route == 1].sum(), (route == 0).sum(), y[route == 0].sum()), flush=True)


def fit_pair(idx):
    c, r = idx[route[idx] == 1], idx[route[idx] == 0]
    cand = (model.fit(X[c].tolist(), y[c].tolist(), precision_floor=a.candidate_floor, recall_floor=a.candidate_recall)
            if a.candidate_logistic else
            model.fit_forest(X[c].tolist(), y[c].tolist(), precision_floor=0.0, recall_floor=a.candidate_recall))
    res = model.fit_forest(X[r].tolist(), y[r].tolist(), precision_floor=0.0, recall_floor=a.residual_recall)
    return RoutedGate(candidates=cand, residual=res)


def scores(sub, rows):
    """All rows in one call: a forest scored row by row spins up its thread pool every time."""
    if hasattr(sub, "clf"):
        return sub.clf.predict_proba(rows)[:, 1]
    return np.array([sub.score_row(r) for r in rows.tolist()])


def passes(g, rows, routes):
    out = np.zeros(len(rows), bool)
    for m, sub in ((1, g.candidates), (0, g.residual)):
        sel = routes == m
        if sel.any():
            out[sel] = scores(sub, np.asarray(rows)[sel]) >= sub.threshold
    return out


def report(name, keepv, truth, mask=None, w=None):
    mask = np.ones(len(truth), bool) if mask is None else mask
    w = np.ones(len(truth)) if w is None else w
    k, t, w = keepv[mask], truth[mask], w[mask]
    tp = (w * (k & (t == 1))).sum(); pos = (w * (t == 1)).sum(); kept = (w * k).sum()
    print("  %-30s recall %7.2f%%  precision %6.2f%%   (%d pos, %d missed, %d kept)"
          % (name, 100 * tp / max(pos, 1e-9), 100 * tp / max(kept, 1e-9),
             int((t == 1).sum()), int(((t == 1) & ~k).sum()), int(k.sum())), flush=True)


oof = np.zeros(len(y), bool)
for tr, te in StratifiedGroupKFold(a.folds, shuffle=True, random_state=0).split(X, route * 2 + y, groups):
    g = fit_pair(tr)
    oof[te] = passes(g, X[te], route[te])
print("out of fold, held out by article:")
report("both routes", oof, y)
report("candidates", oof, y, route == 1)
report("residual", oof, y, route == 0)
if a.dump:
    json.dump([{"kind": "missed" if y[i] else "false positive", "route": int(route[i]), "article_id": groups[i],
                "table_id": R[i].get("table_id"), "source": R[i].get("source"), "hand": bool(R[i].get("hand_judged")),
                "label": R[i].get("label"), "reasons": R[i].get("reasons"), "caption": R[i].get("caption"),
                "text": (R[i].get("table_serialised") or "")[:3000]}
               for i in range(len(y)) if oof[i] != bool(y[i])], open(a.dump, "w"))

gate = fit_pair(np.arange(len(y)))
print("thresholds: candidates %.4f  residual %.4f" % (gate.candidates.threshold, gate.residual.threshold), flush=True)
gates = [("new", gate)] + ([("baseline", RoutedGate.load(a.baseline))] if a.baseline else [])

test = [h for h in hand if h["split"] == "test" and h["verdict"] in ("positive", "negative")]
TV = vectors([(h["text"], h["caption"], h["footer"]) for h in test])
TX = np.array([[v[n] for n in NAMES] for v in TV]); TR = np.array([int(v["reader_points"] > 0) for v in TV])
truth = np.array([h["verdict"] == "positive" for h in test], int)
base = lambda h: h["stratum"].replace("-2", "")
n_in = collections.Counter(base(h) for h in test)
size = {base(h): h["stratum_size"] for h in test}
wts = np.array([0.0 if base(h) in ("mined", "fail-residual-words") else size[base(h)] / n_in[base(h)] for h in test])
PP = json.load(open(a.parser))
PV = vectors([(p["text"], p["caption"], p["footer"]) for p in PP])
PX = np.array([[v[n] for n in NAMES] for v in PV]); PR = np.array([int(v["reader_points"] > 0) for v in PV])
for name, g in gates:
    k = passes(g, TX, TR) if name == "new" else np.array([bool(g.predict(h["text"], h["caption"], h["footer"])) for h in test])
    print("hand test half [%s]:" % name)
    report("all", k, truth)
    report("random strata, corpus-weighted", k, truth, wts > 0, wts)
    for s in sorted({h["stratum"] for h in test}):
        report("  " + s, k, truth, np.array([h["stratum"] == s for h in test]))
    for h, kk, t in zip(test, k, truth):
        if t and not kk:
            print("      MISSED", h["stratum"], h["article_id"], h["table_id"], "|", h.get("note", "")[:90])
    pk = passes(g, PX, PR) if name == "new" else np.array([bool(g.predict(p["text"], p["caption"], p["footer"])) for p in PP])
    print("parser-read tables [%s]: recall %.2f%% (%d of %d missed)" % (name, 100 * pk.mean(), (~pk).sum(), len(pk)), flush=True)
if a.save:
    import joblib
    joblib.dump(gate, a.save)
    print("saved", a.save)
