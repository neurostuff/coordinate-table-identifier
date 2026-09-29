"""The generator's contract: the target is honest, and the hard cases are common.

These are statistical tests over a fixed seed range, so they are deterministic
but they assert bands rather than points. A band that fails because a weight
changed on purpose should be updated with the weight, in the same commit.
"""

from nspond_tables import synth
import collections
import re

import pytest

from nspond_tables import read
from nspond_tables.fields import visible_space
from nspond_tables.synth import DEFAULT, build

N = 600
TABLES = [build(seed=s) for s in range(N)]


def _rate(predicate):
    return sum(1 for t in TABLES if predicate(t)) / len(TABLES)


# -- the target may not claim what the document does not say ---------------

def test_the_space_label_always_matches_the_document():
    """A target asserting an unstated space teaches a model to invent one."""
    for t in TABLES:
        stated = visible_space(t.caption, t.footer) or visible_space(t.grid.render())
        assert t.truth.space == stated, t.grid.render()[:200]


def test_a_statistic_type_is_claimed_only_where_the_document_names_it():
    """A name counts -- "Coefficient" names a Beta -- so ask the field reader.

    Which columns are axes is decided by the reader, not by matching text: a
    Z-statistic column is headed `Z`, exactly like the z axis, and only the
    adjacency rule tells them apart.
    """
    from nspond_tables.fields import statistic_type
    for t in TABLES:
        claimed = {p.statistic_type for a in t.truth.analyses
                   for p in a.points if p.statistic_type}
        if not claimed:
            continue
        grid = read.parse(t.grid.render())
        axes = set((read.axis_columns(grid) or {}).values())
        derivable = {statistic_type(read._column_header_text(grid, col))
                     for col in range(grid.width()) if col not in axes}
        derivable |= {read.claimed_statistic(t.footer),
                      read.claimed_statistic(t.caption)}
        for kind in claimed:
            assert kind in derivable, (kind, derivable, t.grid.render()[:120])


def test_a_statistic_value_is_claimed_only_where_a_cell_holds_it():
    for t in TABLES:
        cells = {c.text for row in t.grid.rows for c in row}
        for a in t.truth.analyses:
            for p in a.points:
                if p.statistic_value is None:
                    continue
                assert any(str(p.statistic_value) in c for c in cells) or \
                    any(str(p.statistic_value).rstrip("0").rstrip(".") in c for c in cells)


def test_every_coordinate_is_inside_a_head():
    w = DEFAULT
    for t in TABLES:
        for a in t.truth.analyses:
            for p in a.points:
                assert abs(p.x) <= w.x_limit and abs(p.y) <= w.y_limit
                assert abs(p.z) <= w.z_limit


# -- a faithful reader recovers what the target claims --------------------

def test_a_reader_recovers_every_coordinate_it_can_locate():
    hit = total = 0
    for t in TABLES:
        got = read.extract(t.grid.render(), caption=t.caption, footer=t.footer)
        if got.located_by != "header":
            continue
        truth = {(p.x, p.y, p.z) for a in t.truth.analyses for p in a.points}
        found = {(p.x, p.y, p.z) for p in got.points}
        hit += len(truth & found)
        total += len(truth)
        assert not (found - truth), "reader invented %s" % (found - truth)
    assert total > 0
    assert hit / total == 1.0


# -- the difficulty levers fire ------------------------------------------

def test_a_numeric_column_sits_before_x_far_more_often_than_in_papers():
    """The lever that broke v14's minus-sign rule. Real rate is ~28%."""
    def _leads(t):
        cols = t.notes["layout"]
        axis = "xyz" if "xyz" in cols else "x"
        return "extent" in cols and cols.index("extent") < cols.index(axis)

    rate = _rate(_leads)
    assert 0.62 <= rate <= 0.85, rate


def test_the_axes_are_named_less_often_than_in_papers():
    """Handing over the locating cue more than reality does teaches dependence."""
    rate = _rate(lambda t: t.notes["axes_named"])
    assert 0.35 <= rate <= 0.55, rate


def test_the_space_is_sometimes_in_the_table_and_sometimes_only_in_the_context():
    """The rates track the re-extracted corpus, where the space is stated in
    the table on 45.2% of real coordinate tables -- down from 84.1% before ACE
    began scanning source html, because the rescued tables state it far less
    often than the ones it used to parse."""
    in_table = _rate(lambda t: t.notes["space_in_table"] and t.truth.space)
    context = _rate(lambda t: not t.notes["space_in_table"] and t.truth.space)
    nowhere = _rate(lambda t: t.truth.space is None)
    assert 0.35 < in_table < 0.55, in_table
    assert context > 0.05, context
    assert nowhere > 0.2, nowhere


def test_every_table_carries_at_least_one_section_divider():
    assert _rate(lambda t: t.notes["dividers"] >= 1) > 0.95


def test_some_dividers_are_partial_spans_with_the_row_left_empty():
    """The form the naturalistic table uses; the old generator never made it."""
    found = 0
    for t in TABLES:
        for row in t.grid.resolve():
            texts = [p for p in row if not p.cell.is_filler and p.cell.text.strip()]
            if len(texts) == 1 and texts[0].cell.colspan > 1 and len(row) > 1:
                found += 1
                break
    assert found / len(TABLES) > 0.15, found / len(TABLES)


def test_midline_coordinates_appear_at_about_the_real_rate():
    zero = sum(t.notes["zero_x_rows"] for t in TABLES)
    points = sum(len(a.points) for t in TABLES for a in t.truth.analyses)
    assert 0.005 <= zero / points <= 0.05, zero / points


def test_laterality_and_x_sign_agree_about_as_often_as_in_papers():
    from nspond_tables.fields import sign_agrees
    agree = dis = 0
    for t in TABLES:
        got = read.extract(t.grid.render(), caption=t.caption, footer=t.footer)
        for p in got.points:
            verdict = sign_agrees(p.x, p.label)
            if verdict is True:
                agree += 1
            elif verdict is False:
                dis += 1
    assert agree + dis > 100
    rate = agree / (agree + dis)
    assert 0.94 <= rate <= 1.0, rate


def test_a_spanning_row_is_not_always_an_analysis_boundary():
    """An anatomical sub-heading inside one analysis. A model that splits on
    every banner gets these wrong, so they have to be present."""
    more_banners_than_analyses = 0
    for t in TABLES:
        if t.notes["dividers"] > len(t.truth.analyses):
            more_banners_than_analyses += 1
    assert more_banners_than_analyses / len(TABLES) > 0.10


def test_the_generator_is_deterministic():
    assert build(seed=42).grid.render() == build(seed=42).grid.render()
    assert build(seed=42).grid.render() != build(seed=43).grid.render()


# -- tables that hold nothing ---------------------------------------------

def test_a_table_with_no_coordinates_gets_the_empty_structure():
    """Not a missing field, not a refusal, and not an analysis with an empty
    point list. No version before v19 was shown one at all, which is why the
    model invents an analysis when handed a demographics table."""
    for kind in synth.empty.KINDS:
        t = synth.build_empty(seed=7, kind=kind)
        assert t.truth.as_target() == {"space": None, "analyses": []}
        assert t.grid.render().strip()


def test_the_hard_negatives_are_hard_by_construction():
    """A `triple` table prints estimates beside their intervals, which parse as
    coordinate triples; that shape is behind 75 of the 87 real tables the
    reader misreads."""
    from nspond_tables import read
    got = [len(read.extract(synth.build_empty(seed=s, kind="triple").grid.render()).points)
           for s in range(30)]
    assert sum(1 for g in got if g >= 3) > 25, got
    plain = [len(read.extract(synth.build_empty(seed=s, kind="ordinary").grid.render()).points)
             for s in range(30)]
    assert sum(plain) == 0, plain


def test_an_interval_belongs_beside_an_estimate_not_a_flip_angle():
    """An MRI acquisition table does not print confidence intervals, and one
    that does teaches the model only that the generator is careless."""
    flavours = {synth.build_empty(seed=s, kind="triple").notes["flavour"]
                for s in range(60)}
    assert flavours <= set(synth.empty._TAKES_AN_INTERVAL), flavours


def test_the_mix_holds_both_kinds_at_the_weighted_rate():
    empty = sum(1 for s in range(600) if synth.build_mixed(seed=s).notes.get("empty"))
    assert 0.18 < empty / 600 < 0.32, empty / 600


def test_real_negatives_are_used_before_generated_ones():
    """A synthetic negative is a guess about what a non-coordinate table looks
    like; a real one is not."""
    rows = [{"label": "negative", "table_serialised": "#Measure | #p\nAge | 0.4",
             "caption": "Participant characteristics", "hand_judged": True}]
    got = synth.build_trainset(n=40, records=rows, seed=1)
    empties = [e for e in got if not e.target["analyses"]]
    assert any(e.origin == "hand-judged" for e in empties)
    assert all(e.target == {"space": None, "analyses": []} for e in empties)


def test_a_real_positive_carries_its_own_points_and_banners():
    """The target is read off the table, not invented: the points are the
    reader's and the grouping is the banner rows it can see."""
    rows = [{"label": "positive", "caption": "Peak activations, MNI",
             "table_serialised": "\n".join([
                 "#Region | #x | #y | #z | #t",
                 "<5:Faces > houses",
                 "L fusiform | -42 | -55 | -18 | 5.01",
                 "R IFG | 44 | 16 | 2 | 3.90",
                 "<5:Houses > faces",
                 "L PHG | -24 | -40 | -12 | 4.22",
                 "R PHG | 26 | -38 | -10 | 4.10"])}]
    got = synth.trainset.real_positives(rows)
    assert len(got) == 1
    target = got[0].target
    assert [a["name"] for a in target["analyses"]] == ["Faces > houses", "Houses > faces"]
    assert [len(a["points"]) for a in target["analyses"]] == [2, 2]
    assert target["analyses"][0]["points"][0][:3] == [-42.0, -55.0, -18.0]


def test_a_table_the_reader_cannot_read_is_not_given_a_target():
    """A guessed target teaches the model to guess."""
    rows = [{"label": "positive", "caption": "Demographics",
             "table_serialised": "#Measure | #p\nAge | 0.4"}]
    assert synth.trainset.real_positives(rows) == []


def test_the_whole_triple_sometimes_sits_in_one_cell():
    """Nearly a quarter of real coordinate tables write it that way, and the
    generator never did -- so a model trained on it had no reason to look
    inside a cell for three numbers."""
    packed = _rate(lambda t: "xyz" in t.notes["layout"])
    assert 0.15 < packed < 0.30, packed


def test_a_packed_triple_still_yields_its_points_to_the_reader():
    from nspond_tables import read
    for t in TABLES:
        if "xyz" not in t.notes["layout"]:
            continue
        got = read.extract(t.grid.render(), caption=t.caption, footer=t.footer)
        want = [p for a in t.truth.analyses for p in a.points]
        assert len(got.points) == len(want), t.grid.render()[:160]
        break


def test_some_tables_mark_no_header_at_all():
    """7% of real ones write it in <td>; the generator is over that on purpose
    because a reader that gives up there loses a table it could have read."""
    unmarked = _rate(lambda t: not t.grid.render().split("\n")[0].lstrip().startswith("#"))
    assert 0.08 < unmarked < 0.25, unmarked


def test_a_footnote_is_not_always_there():
    """47% of real tables carry one. Always writing one teaches the model to
    expect it -- and to have nowhere to look when it is missing."""
    with_footer = _rate(lambda t: bool(t.footer.strip()))
    assert 0.40 < with_footer < 0.75, with_footer


def test_a_generated_table_is_at_least_as_big_as_a_real_one():
    """Real coordinate tables carry 14.5 points. A generator that makes
    smaller ones is training on the easy end of the corpus."""
    from nspond_tables import read
    total = sum(len(read.extract(t.grid.render(), caption=t.caption,
                                 footer=t.footer).points) for t in TABLES)
    read_any = sum(1 for t in TABLES
                   if read.extract(t.grid.render(), caption=t.caption,
                                   footer=t.footer).points)
    assert total / max(read_any, 1) >= 13.0, total / max(read_any, 1)


def test_a_fractional_coordinate_is_a_property_of_the_point():
    """Rolled per axis it compounded: 1 - (1 - 0.11)^3 is 29.5% of points
    fractional where the weight says 11%."""
    from nspond_tables import read
    pts = [p for t in TABLES
           for p in read.extract(t.grid.render(), caption=t.caption,
                                 footer=t.footer).points]
    frac = sum(1 for p in pts if not all(float(v).is_integer()
                                         for v in (p.x, p.y, p.z)))
    assert 0.06 < frac / max(len(pts), 1) < 0.17, frac / max(len(pts), 1)
