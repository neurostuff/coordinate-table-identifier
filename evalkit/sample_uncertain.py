"""50 UNCERTAIN tables, stratified by gate score, for hand review.

Stratified rather than random: a random sample of a band that is 87% pass tells
you least about the boundary, which is where a gate is worth checking.
"""
import sys, json, random, collections
sys.path.insert(0, "/tmp/nspond/pkg/src")
from nspond_tables.classify import Gate, dataset

gate = Gate.load("/home/james/train-data/coord_gate.json")
records = [r for r in dataset.rows_from_jsonl("/home/james/train-data/coord_labels.jsonl")
           if r.get("label") == "uncertain"]
scored = []
for r in records:
    s = gate.score(r.get("table_serialised") or "", r.get("caption") or "",
                   r.get("footer") or "")
    scored.append((s, r))
scored.sort(key=lambda t: t[0])
# ten bands of five
picked, rng = [], random.Random(11)
n = len(scored)
for b in range(10):
    lo, hi = int(n * b / 10), int(n * (b + 1) / 10)
    band = scored[lo:hi]
    picked.extend(rng.sample(band, min(5, len(band))))
picked.sort(key=lambda t: t[0])
out = []
for i, (s, r) in enumerate(picked, 1):
    out.append({"n": i, "score": round(s, 4), "source": r.get("source"),
                "slug": r.get("slug"), "table_id": r.get("table_id"),
                "caption": (r.get("caption") or "")[:150],
                "rows": (r.get("table_serialised") or "").split("\n")[:7],
                "n_rows": len((r.get("table_serialised") or "").split("\n"))})
json.dump(out, open("/tmp/nspond/review50.json", "w"), indent=1)
print("  %d uncertain tables, sampled %d across 10 score bands" % (n, len(picked)))
print("  score range %.4f .. %.4f" % (picked[0][0], picked[-1][0]))
