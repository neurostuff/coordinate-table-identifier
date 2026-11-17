"""Seed blocks and their anatomical twin: one layout, two answers.

A region label heading a block of rows means one of two things. In a
seed-based connectivity (or PPI) table it names a **seed**, an independent
variable, so every seed is its own analysis. In an activation table it is an
**anatomical cue** that only sorts one contrast's peaks by place, so the block
stays inside that contrast's analysis. Nothing in the label decides which: the
caption, a `Seed` header, the word "seed" in the label, and where the rows
under it lie do.

Before this module the generator only ever made the anatomical kind (676
v22 tables with two or more region-named banners, every one merged into one
analysis) and no seed table at all, so the only lesson it taught was "a region
banner never starts an analysis". The curated seed tables taught the other
half and were themselves often merged: of the 872 curated train_v22 examples
that mention a seed, Codex split 92 by seed under the corrected brief
(coordinate-fixes `v22seed/`, 2026-11-17).

`build_seed_pair(seed)` returns both readings of one plan: the same layout,
header, contrasts and, in the shared mode, the same banner text. The seed twin
splits by seed; the anatomical twin keeps one analysis per contrast. The
difference between them is only the context a reader has to use.

Three ways a table says its blocks are seeds, each generated:

* ``caption``: the banners are bare region names and only the caption says
  the table reports connectivity with seeds -- the hard case;
* ``banner``: the banner itself says so (`Left amygdala seed`,
  `Seed: Cerebellum`), sometimes with the seed's own centre after it, which is
  never a point of the analysis;
* ``column``: a `Seed` column names the seed on every row or spans its block.
  Its anatomical twin heads the same column `Lobe`.
"""

from __future__ import annotations

import copy
import random
import re
from typing import List, Optional, Tuple

from ..grid import Cell, Grid
from . import vocab
from .build import (Table, Truth, TruthAnalysis, _analysis_name, _context,
                    _coord, _data_row, _fmt, _header, _layout, _rough_up,
                    _divider)
from .weights import DEFAULT, Weights

#: Labels that head a block in papers of both kinds. As a seed: a cerebellar,
#: visual or cingulate seed map. As anatomy: the section the peaks sit in.
#: Each maps to the lobe its anatomical rows are drawn from.
SHARED_LABELS = {
    "Frontal cortex": "frontal", "Temporal cortex": "temporal",
    "Parietal cortex": "parietal", "Visual cortex": "occipital",
    "Basal ganglia": "subcortical", "Cingulate cortex": "limbic",
    "Cerebellum": "cerebellum",
}

#: Regions papers seed from. Their anatomical twin heads the block with the
#: region's lobe section instead, since a paper does not head an activation
#: block `Left amygdala` and list peaks all over the brain beneath it.
SEED_REGIONS = ("amygdala", "hippocampus", "insula", "thalamus",
                "posterior cingulate", "anterior cingulate", "precuneus",
                "caudate", "putamen", "medial frontal gyrus",
                "middle frontal gyrus", "inferior parietal lobule",
                "cerebellum")

SEED_COLUMN_HEADERS = ["Seed", "Seed region", "Seed ROI", "ROI", "Seed regions"]
LOBE_COLUMN_HEADERS = ["Lobe", "Lobe/region", "Division", "Anatomical division"]

#: Seed captions that say "seed": the only cue when the banners are bare.
SEED_CAPTIONS = [
    "Brain regions showing significant functional connectivity with each seed region",
    "Seed-based resting-state functional connectivity",
    "Psychophysiological interaction (PPI) results for each seed",
    "Clusters functionally connected to the seed regions",
    "Whole-brain seed-to-voxel connectivity results",
    "Regions whose connectivity with each seed ROI differed",
]
#: Connectivity captions that do not say "seed": the header or banner must.
CONNECTIVITY_CAPTIONS = [
    "Resting-state functional connectivity results",
    "Functional connectivity differences",
    "Regions showing altered functional connectivity",
]
#: Activation captions for the anatomical twin. Some speak of connectivity
#: too -- a voxel-wise connectivity contrast is sorted by lobe like any other
#: map -- so that the word "connectivity" alone never decides the split.
ANATOMY_CAPTIONS = [
    "Regions showing significant activation",
    "Peak activations",
    "Clusters surviving whole-brain correction",
    "Local maxima of significant clusters",
]
ANATOMY_CONNECTIVITY_CAPTIONS = [
    "Regions showing altered global functional connectivity",
    "Regions showing increased degree centrality",
    "Clusters of altered regional homogeneity",
]


def _plain_layout(rng: random.Random, w: Weights):
    """A row-block layout opening on the region column.

    The shapes no reader can follow are generated elsewhere; this one is about
    grouping, so it keeps to layouts whose rows a reader can locate.
    """
    for _ in range(64):
        lay = _layout(rng, w)
        if (lay.columns[0] == "region" and not lay.voxel_indices
                and not lay.signs_spaced and not lay.coords_in_name
                and not lay.coords_first and not lay.unmarked_span):
            return lay
    return lay


def _side_name(rng: random.Random, region: vocab.Region,
               side: Optional[str]) -> str:
    name = region.name[0].upper() + region.name[1:]
    if side and not region.midline:
        name = "%s %s" % (rng.choice([{"L": "Left", "R": "Right"}[side], side]),
                          region.name)
    return name


def _caption(rng: random.Random, head: str, contrast: Optional[str],
             spoken: str) -> str:
    """`Table N. head[ for contrast].` plus the space sentence `_context` chose."""
    caption = "Table %d. %s" % (rng.randint(1, 6), head)
    caption += (" for %s." % contrast) if contrast else "."
    tail = re.search(r" Coordinates are in \w+ space\.", spoken)
    return caption + (tail.group(0) if tail else "")


def _plan(seed: int, w: Weights) -> dict:
    """Everything the two twins share."""
    rng = random.Random("seed-pair-plan-%d" % seed)
    lay = _plain_layout(rng, w)
    header = _header(rng, lay, w)
    mode = "shared" if rng.random() < 0.5 else "region"
    cue = rng.choice(["caption", "banner", "column"])
    n_blocks = rng.randint(2, 4)
    if mode == "shared":
        names = rng.sample(sorted(SHARED_LABELS), n_blocks)
        blocks = [{"label": n, "lobe": SHARED_LABELS[n], "region": None,
                   "side": None} for n in names]
    else:
        pool = [r for r in vocab.REGIONS if r.name in SEED_REGIONS]
        rng.shuffle(pool)
        blocks, lobes = [], set()
        for region in pool:                 # distinct lobes: the twin's sections
            if region.lobe in lobes:
                continue
            lobes.add(region.lobe)
            side = None if region.midline else rng.choice(["L", "R"])
            blocks.append({"label": _side_name(rng, region, side),
                           "lobe": region.lobe, "region": region, "side": side})
            if len(blocks) == n_blocks:
                break
    n_contrasts = 2 if rng.random() < 0.4 else 1
    contrasts = []
    while len(contrasts) < n_contrasts:
        name = _analysis_name(rng)
        if name not in contrasts:
            contrasts.append(name)
    bannered = n_contrasts > 1 or rng.random() < 0.5
    return {"lay": lay, "header": header, "mode": mode, "cue": cue,
            "blocks": blocks, "contrasts": contrasts, "bannered": bannered,
            "rows": [[rng.randint(1, 4) for _ in blocks] for _ in contrasts],
            "seed_centre": rng.random() < 0.2,
            "span_column": rng.random() < 0.6}


def _seed_label(rng: random.Random, block: dict, cue: str,
                centre: bool, w: Weights) -> Tuple[str, str]:
    """(text printed, analysis name) for a seed block."""
    label = block["label"]
    if cue == "caption":
        return label, label
    if cue == "banner":
        form = rng.choice(["%s seed", "Seed: %s", "Seed region: %s"])
        printed = form % label
    else:
        printed = label
    name = printed
    if centre and block["region"] is not None:
        # The seed's own centre, printed with it. It says where the seed is,
        # not what the seed found, so it is never a point of the analysis.
        x, y, z = _coord(rng, block["region"].sided(block["side"]), w)
        printed = "%s (%s, %s, %s)" % (printed, _fmt(x), _fmt(y), _fmt(z))
    return printed, name


def _anatomy_label(rng: random.Random, block: dict) -> str:
    if block["region"] is None:
        return block["label"]
    return rng.choice(vocab.LOBE_SECTIONS[block["lobe"]])


def _twin(seed: int, plan: dict, as_seeds: bool, w: Weights) -> Table:
    rng = random.Random("seed-pair-%d-%s" % (seed, "seed" if as_seeds else "anatomy"))
    lay, cue = plan["lay"], plan["cue"]
    header = copy.deepcopy(plan["header"])
    column = cue == "column"
    if column:
        head = rng.choice(SEED_COLUMN_HEADERS if as_seeds else LOBE_COLUMN_HEADERS)
        header[0].insert(0, Cell(head, header=lay.header_marked,
                                 rowspan=len(header)))
    width = sum(c.colspan for c in header[0])
    grid = Grid()
    for row in header:
        grid.add(row)

    truth = Truth(space=lay.space)
    dividers = 0
    labels = [(_seed_label(rng, b, cue, plan["seed_centre"], w) if as_seeds
               else (_anatomy_label(rng, b),) * 2) for b in plan["blocks"]]
    for ci, contrast in enumerate(plan["contrasts"]):
        if plan["bannered"]:
            grid.add(_divider(rng, contrast, width, w))
            dividers += 1
        merged = TruthAnalysis(name=contrast, measure=lay.measure)
        for bi, block in enumerate(plan["blocks"]):
            printed, seed_name = labels[bi]
            if as_seeds:
                # Outermost label first: the contrast (banner or caption),
                # then the seed it was tested on.
                analysis = TruthAnalysis(name=contrast + " — " + seed_name,
                                         measure=lay.measure)
            else:
                analysis = merged
            if not column:
                grid.add(_divider(rng, printed, width, w))
                dividers += 1
            if as_seeds:                    # connected peaks lie anywhere
                pool = list(vocab.REGIONS)
            else:                           # an anatomical section's peaks lie in it
                pool = [r for r in vocab.REGIONS if r.lobe == block["lobe"]]
            rng.shuffle(pool)
            n = min(plan["rows"][ci][bi], len(pool))
            spanned = column and plan["span_column"] and n > 1
            for pi in range(n):
                region = pool[pi]
                side = None if region.midline else rng.choice(["L", "R"])
                cells, point = _data_row(rng, lay, region, side, w, False)
                _rough_up(rng, cells, lay, w, point)
                if column:
                    if not spanned:
                        cells.insert(0, Cell(printed))
                    elif pi == 0:
                        cells.insert(0, Cell(printed, rowspan=n))
                grid.add(cells)
                analysis.points.append(point)
            if as_seeds:
                truth.analyses.append(analysis)
        if not as_seeds:
            truth.analyses.append(merged)

    spoken_caption, footer = _context(rng, lay, w, names_in_caption=None)
    contrast_in_caption = None if plan["bannered"] else plan["contrasts"][0]
    if as_seeds:
        heads = SEED_CAPTIONS if cue == "caption" else (
            SEED_CAPTIONS + CONNECTIVITY_CAPTIONS)
    else:
        heads = (ANATOMY_CONNECTIVITY_CAPTIONS if rng.random() < 0.3
                 else ANATOMY_CAPTIONS)
    caption = _caption(rng, rng.choice(heads), contrast_in_caption, spoken_caption)
    notes = {"seed_blocks": "seed" if as_seeds else "anatomy",
             "seed_cue": cue, "seed_mode": plan["mode"],
             "layout": (["seed" if as_seeds else "lobe"] if column else [])
             + list(lay.columns),
             "dividers": dividers, "axes_named": lay.axes_named,
             "packed_form": lay.packed_form, "reader_cannot": False,
             "space_in_table": lay.space_in_table,
             "identical_neighbours": False, "zero_x_rows": 0}
    return Table(grid=grid, truth=truth, caption=caption, footer=footer, notes=notes)


def build_seed_pair(seed: int = 0, weights: Weights = DEFAULT) -> Tuple[Table, Table]:
    """(seed twin, anatomical twin) of one plan."""
    plan = _plan(seed, weights)
    return _twin(seed, plan, True, weights), _twin(seed, plan, False, weights)
