"""Command line: serialise a source table, extract from it, or generate one."""

from __future__ import annotations

import argparse
import json
import sys

from . import read, serialize
from .synth import build


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="nspond-tables", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serialize", help="render a source table")
    s.add_argument("path", help="file holding the raw table, or - for stdin")

    e = sub.add_parser("extract", help="render, then read what is determinable")
    e.add_argument("path")
    e.add_argument("--caption", default="")
    e.add_argument("--footer", default="")

    g = sub.add_parser("generate", help="synthetic tables with their targets")
    g.add_argument("-n", type=int, default=1)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--jsonl", action="store_true", help="one training row per line")

    a = ap.parse_args(argv)
    if a.cmd == "generate":
        for i in range(a.n):
            t = build(seed=a.seed + i)
            if a.jsonl:
                print(json.dumps({
                    "origin": "synthetic",
                    "caption": t.caption,
                    "footer": t.footer,
                    "table_serialised": t.grid.render(),
                    "target_json": json.dumps(t.truth.as_target(), separators=(",", ":")),
                }, ensure_ascii=False))
            else:
                print("=" * 76)
                print("caption:", t.caption)
                print("footer :", t.footer)
                print(t.grid.render())
                print(json.dumps(t.truth.as_target(), indent=2)[:600])
        return 0

    raw = sys.stdin.read() if a.path == "-" else open(a.path, encoding="utf-8",
                                                      errors="replace").read()
    text = serialize.serialize(raw)
    if a.cmd == "serialize":
        print(text)
        return 0
    got = read.extract(text, caption=a.caption, footer=a.footer)
    print(json.dumps({
        "located_by": got.located_by,
        "space": got.space,
        "measure": got.measure,
        "axis_columns": got.axis_columns,
        "sections": got.sections,
        "sign_disagreements": got.sign_disagreements,
        "points": [list(p.as_tuple()) + [p.label] for p in got.points],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
