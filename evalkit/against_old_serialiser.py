"""The new repo against the old serialiser, on real tables. Read-only."""
import sqlite3, json, gzip, re, sys, os, random, collections
sys.path.insert(0, "/tmp/nspond/pkg/src")
sys.path.insert(0, "/home/james/scans")
import serialize2
from nspond_tables import read as nread, serialize as nser
B = "/data/alejandro/projects/ns-pond/ingestion-cache-indices/extract"
PER = int(sys.argv[1]) if len(sys.argv) > 1 else 80

def old_header_read(text, coords):
    """The same header-directed read, spelled for the old format."""
    def cells(l): return [c.strip() for c in l.split("|")]
    def parse(c):
        s = c.strip(); h = s.startswith("#")
        if h: s = s[1:]
        cols = 1
        m = re.match(r"^<(\d+)", s)
        if m: cols = int(m.group(1)); s = s[m.end():]
        m = re.match(r"^\^(\d+)", s)
        if m: s = s[m.end():]
        return h, cols, s
    lines = [l for l in text.split("\n") if l.strip()]
    xi = None
    for l in lines:
        col = 0
        for c in cells(l):
            h, cs_, t = parse(c)
            if h and re.fullmatch(r"x", t, re.I): xi = col; break
            col += cs_
        if xi is not None: break
    if xi is None: return None
    want = collections.Counter(float(c["x"]) for c in coords); got = collections.Counter()
    for l in lines:
        col = 0
        for c in cells(l):
            h, cs_, t = parse(c)
            if col == xi and not h:
                try: got[float(re.sub(r"^[<>~]\s*", "", t))] += 1
                except Exception: pass
            col += cs_
    return sum((want & got).values()) / max(sum(want.values()), 1)

agg = collections.defaultdict(collections.Counter)
for src in ("pubget", "ace", "elsevier", "pdf"):
    p = os.path.join(B, src, "index.sqlite")
    if not os.path.exists(p): continue
    con = sqlite3.connect("file:%s?mode=ro" % p, uri=True); con.execute("PRAGMA query_only=ON")
    n = con.execute("SELECT COUNT(*) FROM extractions").fetchone()[0]
    random.seed(17); got = 0
    for off in random.sample(range(n), min(n, PER * 12)):
        if got >= PER: break
        row = con.execute("SELECT payload_json FROM extractions LIMIT 1 OFFSET ?", (off,)).fetchone()
        if not row: continue
        raw = row[0]
        if isinstance(raw, (bytes, bytearray)) and raw[:2] == b"\x1f\x8b": raw = gzip.decompress(raw)
        try: d = json.loads(raw)
        except Exception: continue
        for t in (d.get("tables") or []):
            if got >= PER: break
            cds = t.get("coordinates") or []
            if len(cds) < 3: continue
            fp = t.get("raw_content_path")
            if not fp or not os.path.exists(fp): continue
            try: content = open(fp, encoding="utf-8", errors="replace").read()
            except Exception: continue
            try:
                o = serialize2.serialize(content)
                w = nser.serialize(content)
            except Exception: continue
            if not o.strip() or not w.strip(): continue
            a = agg[src]; a["tables"] += 1
            hr = old_header_read(o, cds)
            if hr is not None:
                a["old_scored"] += 1; a["old_read"] += hr; a["old_perfect"] += int(hr > 0.999)
            r = nread.extract(w)
            if r.located_by == "header":
                want = collections.Counter(float(c["x"]) for c in cds)
                have = collections.Counter(p_.x for p_ in r.points)
                hit = sum((want & have).values()) / max(sum(want.values()), 1)
                a["new_scored"] += 1; a["new_read"] += hit; a["new_perfect"] += int(hit > 0.999)
            elif r.located_by == "packed cell":
                a["new_packed"] += 1
            got += 1
    con.close()
print("  %-9s %7s | %11s %11s | %11s %11s %8s" % (
    "source","tables","old x-read","old 100%","new x-read","new 100%","packed"))
for src in ("pubget","ace","elsevier","pdf"):
    a = agg[src]
    os_, ns = max(a["old_scored"],1), max(a["new_scored"],1)
    print("  %-9s %7d | %10.1f%% %10.1f%% | %10.1f%% %10.1f%% %8d"
          % (src, a["tables"], 100*a["old_read"]/os_, 100*a["old_perfect"]/os_,
             100*a["new_read"]/ns, 100*a["new_perfect"]/ns, a["new_packed"]))
