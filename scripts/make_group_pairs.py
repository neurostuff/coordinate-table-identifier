"""Column-group twins from nspond-tables (branch synth/column-group-pairs) as train rows.

usage: PYTHONPATH=<nspond-tables worktree>/src python make_group_pairs.py N OUT.jsonl
Each seed gives two tables with the same rows and conditions: one whose column groups name groups,
sessions or arms (split: one analysis per condition and group) and one whose groups are hemispheres or the
same header printed twice (kept: one analysis per condition). Rows have every train_v23 field except
n_tokens (added on beast). table_id is grouppair-<seed>-<split|hemispheres|halves>, source
synth-group-pair, origin generated.
"""
import json, sys
from nspond_tables.synth import build_group_pair
from nspond_tables.synth.trainset import Example, add_context, set_space_from_what_is_visible

n, out = int(sys.argv[1]), sys.argv[2]
examples = []
for s in range(n):
    for t in build_group_pair(seed=s):
        twin = t.notes["column_groups"]
        notes = dict(t.notes, source="synth-group-pair", table_id="grouppair-%d-%s" % (s, twin),
                     kind="coordinates")
        examples.append(Example(table=t.grid.render(), target=t.truth.as_target(),
                                caption=t.caption, footer=t.footer, origin="generated",
                                notes=notes))
examples = set_space_from_what_is_visible(add_context(examples, seed=24))
with open(out, "w") as f:
    for ex in examples:
        f.write(json.dumps(ex.as_training_row(), ensure_ascii=False) + "\n")
print(len(examples))
