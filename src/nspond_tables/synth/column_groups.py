"""Column blocks side by side: groups to split, and halves or hemispheres to keep together.

A table can print two or three blocks of coordinate columns side by side, each with its
own x, y and z. What the blocks are decides the grouping, and only the header says:

* **groups or sessions** (`Patients | Comparison subjects`, `Session 1 | Session 2`,
  `AUD (n = 18) | HC (n = 18)`): each block is a separate result, so every
  condition x block is its own analysis, named `condition — block`;
* **hemispheres** (`Left hemisphere | Right hemisphere`): one result's peaks, sorted by
  side, so a condition is one analysis holding both blocks' points;
* **halves**: one list printed in two runs to save space, each run under the same header
  (`Region | x | y | z | …` twice), so a condition is one analysis read down the left
  run and then the right.

nu-v23 read the first kind as the third: Schneider et al. 2001, Tables 2 and 4, print
patients and healthy subjects side by side under condition banners, and each condition
became one analysis holding both groups' peaks (FDCR evaluation, 2026-11-25). The
generator's only side-by-side layout (`build._column_grouped`) heads every block with a
contrast name and has no condition banners, so it never taught the difference.

`build_group_pair(seed)` returns two readings of one plan: the same conditions, regions,
columns and statistic. The split twin heads its blocks with groups; the other twin
heads them as hemispheres (half the pairs) or repeats one header as halves (the other
half). Only the header and the caption tell them apart.
"""

from __future__ import annotations

import copy
import random
from typing import List, Optional, Tuple

from ..grid import Cell, Grid
from . import vocab
from .build import (Table, Truth, TruthAnalysis, TruthPoint, _axis_names, _coord,
                    _divider, _fmt, _names_space, _stat_value)
from .weights import DEFAULT, Weights

#: Labels heading side-by-side blocks that are separate results.
#: Each with its kind, which picks the caption: a table of sessions is not "in each group".
GROUP_LABELS = [
    ("group", ("Patients", "Comparison subjects")), ("group", ("Patients", "Healthy controls")),
    ("group", ("Alcohol-dependent patients", "Healthy controls")), ("group", ("Smokers", "Non-smokers")),
    ("group", ("Cocaine users", "Controls")), ("group", ("AUD (n = {n})", "HC (n = {m})")),
    ("group", ("Relapsers", "Abstainers")), ("group", ("Group A", "Group B")),
    ("group", ("Adolescents", "Adults")),
    ("session", ("Session 1", "Session 2")), ("session", ("Pre-treatment", "Post-treatment")),
    ("session", ("Baseline", "Follow-up")), ("session", ("Week 0", "Week 12")),
    ("arm", ("Placebo", "Active drug")), ("arm", ("Sham", "Active stimulation")),
]
GROUP_TRIPLES = [
    ("group", ("Patients", "First-degree relatives", "Controls")),
    ("session", ("Session 1", "Session 2", "Session 3")),
    ("arm", ("Placebo", "Low dose", "High dose")),
]
HEMISPHERE_LABELS = [("Left hemisphere", "Right hemisphere"), ("Left", "Right"),
                     ("LH", "RH"), ("Left side", "Right side")]

#: The conditions a banner names, and the contrast a caption names when there is no banner.
CONDITIONS = [
    "Ethanol odor", "Neutral room air", "Alcohol cues > neutral cues", "Neutral cues > alcohol cues",
    "Smoking cues > neutral cues", "Drug cues", "Neutral pictures", "Cue reactivity",
    "Craving > rest", "Reward anticipation", "Loss anticipation", "Working memory load",
    "Stop > go", "Faces > houses", "Negative > neutral pictures", "Task > baseline",
]
GROUP_CAPTIONS = {
    "group": ["Brain regions with activation in {a} and in {b}", "Peak activations in each group",
              "Regions activated in {a} and {b} separately", "Within-group activation maps"],
    "session": ["Activation clusters at each session", "Peak activations at {a} and at {b}",
                "Regions activated in each scanning session"],
    "arm": ["Activation clusters in each treatment arm", "Peak activations under {a} and {b}",
            "Regions activated in each condition of the trial"],
}
HEMISPHERE_CAPTIONS = [
    "Peak activations in the left and right hemispheres",
    "Regions activated bilaterally",
    "Local maxima by hemisphere",
]
HALVES_CAPTIONS = [
    "Regions showing significant activation",
    "Local maxima of significant clusters",
    "Peak activations",
]
EXTENT_HEADERS = [("Voxels", "voxels"), ("k", "voxels"), ("Cluster size", "voxels"),
                  ("Volume (mm3)", "mm3")]


def _plan(seed: int, w: Weights) -> dict:
    """Everything the twins share."""
    rng = random.Random("group-pair-plan-%d" % seed)
    n_conditions = rng.choice([1, 1, 2, 2, 3])
    conditions = rng.sample(CONDITIONS, n_conditions)
    stat_kind = "T" if rng.random() < w.statistic_is_t else rng.choice(["Z", "F"])
    space = rng.choice(["MNI", "TAL"]) if rng.random() < (
        w.space_in_table + w.space_in_context_only) else None
    axes = _axis_names(rng, w, space)
    space = _names_space(axes) or space
    roles = (["side"] if rng.random() < 0.4 else []) + ["x", "y", "z"]
    if rng.random() < 0.8:
        roles.append("stat")
    extent = rng.choice(EXTENT_HEADERS) if rng.random() < 0.5 else None
    if extent:
        roles.append("extent")
    pool = [r for r in vocab.REGIONS if not r.midline] or list(vocab.REGIONS)
    rows = []
    for _ in conditions:
        regions = rng.sample(pool, min(len(pool), rng.randint(2, 6)))
        rows.append([(r, rng.choice(["L", "R"])) for r in regions])
    return {"conditions": conditions, "bannered": n_conditions > 1 or rng.random() < 0.5,
            "stat_kind": stat_kind, "stat_header": rng.choice(vocab.STAT_HEADERS[stat_kind]),
            "space": space, "axes": axes, "roles": roles, "extent": extent,
            "region_header": rng.choice(vocab.REGION_HEADERS),
            "coord_header": rng.choice(vocab.BARE_COORD_HEADERS) if rng.random() < 0.5 else None,
            "rows": rows, "three": rng.random() < 0.2}


def _role_header(plan: dict, role: str) -> str:
    if role in "xyz":
        return plan["axes"]["xyz".index(role)]
    if role == "side":
        return "Side"
    if role == "stat":
        return plan["stat_header"]
    return plan["extent"][0]


def _block_cells(rng: random.Random, plan: dict, region: vocab.Region, side: Optional[str],
                 w: Weights) -> Tuple[List[Cell], TruthPoint]:
    """One block's cells for a region on a side, and the point they print."""
    x, y, z = _coord(rng, region.sided(side), w)
    value = _stat_value(rng, plan["stat_kind"])
    extent = float(rng.randint(3, 900)) if plan["extent"] else None
    cells = []
    for role in plan["roles"]:
        if role == "side":
            cells.append(Cell({"L": "L", "R": "R"}[side] if side else ""))
        elif role in "xyz":
            cells.append(Cell(_fmt({"x": x, "y": y, "z": z}[role])))
        elif role == "stat":
            cells.append(Cell(_fmt(value)))
        else:
            cells.append(Cell(_fmt(extent)))
    point = TruthPoint(x, y, z,
                       statistic_type=plan["stat_kind"] if "stat" in plan["roles"] else None,
                       statistic_value=value if "stat" in plan["roles"] else None,
                       extent=extent)
    return cells, point


def _header(plan: dict, labels: List[str], region_column_per_block: bool) -> List[List[Cell]]:
    """Header rows: block labels over the blocks (none for halves), the coordinate span, the axes."""
    width = len(plan["roles"])
    rows: List[List[Cell]] = []
    has_coord_row = plan["coord_header"] is not None
    depth = (1 if labels else 0) + (1 if has_coord_row else 0) + 1
    if region_column_per_block:
        # Halves: each run repeats the whole header, region column included.
        axes_row = []
        for _ in range(2):
            axes_row.append(Cell(plan["region_header"], header=True))
            axes_row += [Cell(_role_header(plan, r), header=True) for r in plan["roles"]]
        return [axes_row]
    top = [Cell(plan["region_header"], header=True, rowspan=depth)]
    if labels:
        top += [Cell(label, header=True, colspan=width) for label in labels]
        rows.append(top)
    coord_row: List[Cell] = [] if labels else list(top)
    axes_row: List[Cell] = [] if (labels or has_coord_row) else list(top)
    for _ in (labels or [None]):
        if has_coord_row:
            for role in plan["roles"]:
                if role == "x":
                    coord_row.append(Cell(plan["coord_header"], header=True, colspan=3))
                elif role not in "yz":
                    coord_row.append(Cell(_role_header(plan, role), header=True,
                                          rowspan=2 if role != "side" else 2))
        for role in plan["roles"]:
            if not has_coord_row or role in "xyz":
                axes_row.append(Cell(_role_header(plan, role), header=True))
    if has_coord_row:
        rows.append(coord_row)
    rows.append(axes_row)
    return rows


def _group_labels(rng: random.Random, plan: dict) -> Tuple[str, List[str]]:
    """(kind, labels) of the blocks."""
    if plan["three"]:
        kind, labels = rng.choice(GROUP_TRIPLES)
        return kind, list(labels)
    kind, (a, b) = rng.choice(GROUP_LABELS)
    n = rng.randint(10, 40)
    return kind, [a.format(n=n, m=n + rng.randint(-3, 3)), b.format(n=n, m=n + rng.randint(-3, 3))]


def _caption(rng: random.Random, plan: dict, heads: List[str], labels: List[str]) -> str:
    head = rng.choice(heads).format(a=labels[0].lower() if labels else "",
                                    b=labels[-1].lower() if labels else "")
    caption = "Table %d. %s" % (rng.randint(1, 6), head)
    if not plan["bannered"]:
        caption += " for %s." % plan["conditions"][0].lower()
    else:
        caption += "."
    if plan["space"]:
        caption += " Coordinates are in %s space." % (
            "Talairach" if plan["space"] == "TAL" else "MNI")
    return caption


def _split_twin(seed: int, plan: dict, w: Weights) -> Table:
    """Blocks are groups or sessions: one analysis per condition x group."""
    rng = random.Random("group-pair-%d-split" % seed)
    kind, labels = _group_labels(rng, plan)
    grid = Grid()
    header = _header(plan, labels, False)
    for row in header:
        grid.add(row)
    width = sum(c.colspan for c in header[0])
    truth = Truth(space=plan["space"])
    for ci, condition in enumerate(plan["conditions"]):
        if plan["bannered"]:
            grid.add(_divider(rng, condition, width, w))
        per_group = [TruthAnalysis(name=condition + " — " + label,
                                   measure=plan["extent"][1] if plan["extent"] else None)
                     for label in labels]
        for region, side in plan["rows"][ci]:
            found = [rng.random() < 0.7 for _ in labels]
            if not any(found):
                found[rng.randrange(len(found))] = True
            name = region.name[0].upper() + region.name[1:]
            cells = [Cell(name if "side" in plan["roles"] else "%s %s" % (side, name))]
            for gi, present in enumerate(found):
                if not present:
                    cells += [Cell("") for _ in plan["roles"]]
                    continue
                block, point = _block_cells(rng, plan, region, side, w)
                cells += block
                per_group[gi].points.append(point)
            grid.add(cells)
        truth.analyses += [a for a in per_group if a.points]
    caption = _caption(rng, plan, GROUP_CAPTIONS[kind], labels)
    return Table(grid=grid, truth=truth, caption=caption, footer="",
                 notes={"column_groups": "split", "group_kind": kind, "groups": len(labels),
                        "group_labels": labels, "dividers": len(plan["conditions"]) if plan["bannered"] else 0,
                        "layout": ["region"] + plan["roles"] * len(labels)})


def _hemispheres_twin(seed: int, plan: dict, w: Weights) -> Table:
    """Blocks are the left and right hemisphere of one result: one analysis per condition."""
    rng = random.Random("group-pair-%d-hemispheres" % seed)
    labels = list(rng.choice(HEMISPHERE_LABELS))
    plan = dict(plan, roles=[r for r in plan["roles"] if r != "side"])
    grid = Grid()
    header = _header(plan, labels, False)
    for row in header:
        grid.add(row)
    width = sum(c.colspan for c in header[0])
    truth = Truth(space=plan["space"])
    for ci, condition in enumerate(plan["conditions"]):
        if plan["bannered"]:
            grid.add(_divider(rng, condition, width, w))
        analysis = TruthAnalysis(name=condition,
                                 measure=plan["extent"][1] if plan["extent"] else None)
        for region, _ in plan["rows"][ci]:
            found = [rng.random() < 0.75, rng.random() < 0.75]
            if not any(found):
                found[rng.randrange(2)] = True
            cells = [Cell(region.name[0].upper() + region.name[1:])]
            for side, present in zip(("L", "R"), found):
                if not present:
                    cells += [Cell("") for _ in plan["roles"]]
                    continue
                block, point = _block_cells(rng, plan, region, side, w)
                cells += block
                analysis.points.append(point)
            grid.add(cells)
        truth.analyses.append(analysis)
    caption = _caption(rng, plan, HEMISPHERE_CAPTIONS, [])
    return Table(grid=grid, truth=truth, caption=caption, footer="",
                 notes={"column_groups": "hemispheres", "groups": 2, "group_labels": labels,
                        "dividers": len(plan["conditions"]) if plan["bannered"] else 0,
                        "layout": ["region"] + plan["roles"] * 2})


def _halves_twin(seed: int, plan: dict, w: Weights) -> Table:
    """One list printed in two runs under one repeated header: one analysis per condition,
    read down the left run and then the right."""
    rng = random.Random("group-pair-%d-halves" % seed)
    grid = Grid()
    header = _header(plan, [], True)
    for row in header:
        grid.add(row)
    half = 1 + len(plan["roles"])
    width = 2 * half
    truth = Truth(space=plan["space"])
    for ci, condition in enumerate(plan["conditions"]):
        if plan["bannered"]:
            grid.add(_divider(rng, condition, width, w))
        entries = []
        for region, side in plan["rows"][ci]:
            block, point = _block_cells(rng, plan, region, side, w)
            name = region.name[0].upper() + region.name[1:]
            label = name if "side" in plan["roles"] else "%s %s" % (side, name)
            entries.append(([Cell(label)] + block, point))
        if len(entries) < 2:            # a run of one would not be printed in halves
            region, side = plan["rows"][ci][0]
            block, point = _block_cells(rng, plan, region, "R" if side == "L" else "L", w)
            entries.append(([Cell(region.name[0].upper() + region.name[1:])] + block, point))
        left_n = (len(entries) + 1) // 2
        left, right = entries[:left_n], entries[left_n:]
        analysis = TruthAnalysis(name=condition,
                                 measure=plan["extent"][1] if plan["extent"] else None)
        analysis.points = [p for _, p in left] + [p for _, p in right]
        for i in range(left_n):
            cells = list(left[i][0])
            cells += list(right[i][0]) if i < len(right) else [Cell("") for _ in range(half)]
            grid.add(cells)
        truth.analyses.append(analysis)
    caption = _caption(rng, plan, HALVES_CAPTIONS, [])
    return Table(grid=grid, truth=truth, caption=caption, footer="",
                 notes={"column_groups": "halves", "groups": 2, "group_labels": [],
                        "dividers": len(plan["conditions"]) if plan["bannered"] else 0,
                        "layout": (["region"] + plan["roles"]) * 2})


def build_group_pair(seed: int = 0, weights: Weights = DEFAULT) -> Tuple[Table, Table]:
    """(split twin, keep-together twin) of one plan. The second is the hemisphere reading for
    half the seeds and the halves reading for the other half."""
    plan = _plan(seed, weights)
    keep = random.Random("group-pair-kind-%d" % seed).random() < 0.5
    other = (_hemispheres_twin if keep else _halves_twin)(seed, copy.deepcopy(plan), weights)
    return _split_twin(seed, plan, weights), other
