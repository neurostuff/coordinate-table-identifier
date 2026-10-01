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

#: Tables whose analyses are row blocks under banners. A column-grouped table
#: is a different shape with its own contract -- no banners, no extent column,
#: its grouping sideways -- so the rates written for this one do not describe
#: it, and asserting them over both would only loosen them.
ROW_BLOCK = [t for t in TABLES if not t.notes.get("column_grouped")
             and not t.notes.get("reader_cannot")]


def _rate(predicate, tables=None):
    tables = TABLES if tables is None else tables
    return sum(1 for t in tables if predicate(t)) / len(tables)


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
    """Except where the table reports voxel indices, which are counts from a
    corner of the volume rather than millimetres from the anterior commissure
    -- `92 | 132 | 96` under a header reading `Peak MNI`. Every one is
    positive and outside a head, which is why the reader rejects them and why
    they arrive on the residual route."""
    w = DEFAULT
    for t in TABLES:
        if t.notes.get("voxel_indices"):
            for a in t.truth.analyses:
                for p in a.points:
                    assert 0 <= p.x <= 200 and 0 <= p.y <= 240 and 0 <= p.z <= 180
            continue
        for a in t.truth.analyses:
            for p in a.points:
                assert abs(p.x) <= w.x_limit and abs(p.y) <= w.y_limit
                assert abs(p.z) <= w.z_limit


# -- a faithful reader recovers what the target claims --------------------

def test_a_reader_recovers_every_coordinate_it_can_locate():
    hit = total = 0
    for t in ROW_BLOCK:
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


def test_the_reader_invents_nothing_on_a_table_it_cannot_group():
    """It takes the first adjacent x, y, z and stops, so on three coordinate
    blocks side by side it reads one of the three. What it must never do is
    return a point that is not in the table."""
    seen = 0
    for t in TABLES:
        if not t.notes.get("column_grouped"):
            continue
        seen += 1
        got = read.extract(t.grid.render(), caption=t.caption, footer=t.footer)
        truth = {(p.x, p.y, p.z) for a in t.truth.analyses for p in a.points}
        found = {(p.x, p.y, p.z) for p in got.points}
        assert not (found - truth), "reader invented %s" % (found - truth)
    assert seen > 0


# -- the difficulty levers fire ------------------------------------------

def test_a_numeric_column_sits_before_x_far_more_often_than_in_papers():
    """The lever that broke v14's minus-sign rule. Real rate is ~28%."""
    def _leads(t):
        cols = t.notes["layout"]
        axis = "xyz" if "xyz" in cols else "x"
        return "extent" in cols and cols.index("extent") < cols.index(axis)

    rate = _rate(_leads, ROW_BLOCK)
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
    # The lower bound was 0.35 and the measured rate is 0.352, two tables out
    # of 600 above it. At this sample size the standard error is about 0.02, so
    # any change that reshuffles the stream crosses it without the generator
    # having moved: adding one column took it to 0.348 while `nowhere` stayed
    # identical to four places.
    assert 0.32 < in_table < 0.55, in_table
    assert context > 0.05, context
    assert nowhere > 0.2, nowhere


def test_every_table_carries_at_least_one_section_divider():
    assert _rate(lambda t: t.notes["dividers"] >= 1, ROW_BLOCK) > 0.95


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
    reader misreads.

    `ordinary` is not the same as "the reader finds nothing". An ROC table
    prints `AUC (95% CI)` because that is what an ROC table prints, so it
    trips the reader whatever kind it is filed under -- which is exactly why
    real ones do. The claim is that the triple kind trips it far more often,
    not that the others never do.
    """
    from nspond_tables import read
    hard = [len(read.extract(synth.build_empty(seed=s, kind="triple").grid.render()).points)
            for s in range(120)]
    plain = [len(read.extract(synth.build_empty(seed=s, kind="ordinary").grid.render()).points)
             for s in range(120)]
    # Not all of them, and demographics is why: a count triple and `35.56,
    # 8.11, 21-49` were both read off real tables the reader misreads, and on
    # a demographics table with nothing else coordinate-like about it the
    # reader passes over them. That is a gap on a negative, which costs
    # nothing -- a false positive it does not raise is one the gate is not
    # asked about -- and the tables are still worth training the empty target
    # on.
    assert sum(1 for g in hard if g >= 3) > 0.75 * len(hard), hard
    assert sum(1 for g in plain if g >= 3) < 0.35 * len(plain), plain
    assert sum(hard) > 3 * sum(plain), (sum(hard), sum(plain))


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
    smaller ones is training on the easy end of the corpus.

    Counted on the target, which is what the model is asked for. Counting
    what the READER recovers measures the reader instead, and now that a
    tenth of the tables are built so the reader recovers nothing, that number
    says nothing about how big the tables are."""
    sizes = [sum(len(a.points) for a in t.truth.analyses) for t in TABLES]
    assert sum(sizes) / len(sizes) >= 12.0, sum(sizes) / len(sizes)

    from nspond_tables import read
    readable = [t for t in TABLES if not t.notes.get("reader_cannot")]
    total = sum(len(read.extract(t.grid.render(), caption=t.caption,
                                 footer=t.footer).points) for t in readable)
    read_any = sum(1 for t in readable
                   if read.extract(t.grid.render(), caption=t.caption,
                                   footer=t.footer).points)
    assert total / max(read_any, 1) >= 12.5, total / max(read_any, 1)


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


# -- the mess ------------------------------------------------------------

def test_real_tables_are_untidy_and_so_are_these():
    """93.3% of real coordinate tables hang a marker off a value -- `4.14*`,
    `Cuneusa` -- and every version before v19 was trained as if none did."""
    import re
    marker = re.compile(r"(?<=[\dA-Za-z)])(?:\*{1,3}|†|‡|[a-c](?![A-Za-z]))\s*$")
    marked = _rate(lambda t: any(marker.search(c.cell.text)
                                 for row in read.parse(t.grid.render()).resolve()
                                 for c in row if not c.cell.is_filler), ROW_BLOCK)
    assert marked > 0.80, marked


def test_a_blanked_cell_leaves_the_target_saying_nothing_about_it():
    """A marker is cosmetic and the number under it is unchanged. A blank is
    not: whatever it held is no longer stated, and a target still asserting it
    would teach the model to read a value that is not there."""
    for t in TABLES:
        body = t.grid.render()
        for a in t.truth.as_target()["analyses"]:
            for p in a["points"]:
                if p[4] is not None:
                    assert ("%g" % p[4]) in body, (p, body[:200])
                if p[5] is not None:
                    assert ("%g" % p[5]) in body, (p, body[:200])


def test_the_reader_recovers_every_point_from_a_messy_table():
    """The mess is there to be read through, not to hide the answer. Any gap
    here is a reader defect, and finding them this way is the point.

    Column-grouped tables are excluded, and not because they are awkward: the
    reader takes the first adjacent x, y, z it finds, so on a table carrying
    three coordinate blocks side by side it reads one and stops. Doing better
    means deciding which contrast a column belongs to, which is the judgement
    this pipeline keeps a model for.
    """
    missed = 0
    for t in TABLES:
        if t.notes.get("column_grouped") or t.notes.get("reader_cannot"):
            continue
        got = read.extract(t.grid.render(), caption=t.caption, footer=t.footer)
        want = sum(len(a["points"]) for a in t.truth.as_target()["analyses"])
        missed += len(got.points) != want
    assert missed == 0, missed


def test_an_analysis_is_sometimes_a_column_block_not_a_row_block():
    """8.4% of multi-analysis curated tables are built this way: two contrasts
    share every row and only the spanning header says which column belongs to
    which. A banner cannot help, and neither can reading down."""
    rate = _rate(lambda t: bool(t.notes.get("column_grouped")))
    assert 0.10 < rate < 0.25, rate
    for t in TABLES:
        if not t.notes.get("column_grouped"):
            continue
        target = t.truth.as_target()
        assert len(target["analyses"]) >= 2
        body = t.grid.render()
        for a in target["analyses"]:
            assert a["name"] in body, a["name"]
            assert a["points"]
        break


# -- the shapes no reader can follow --------------------------------------

def test_the_shapes_only_a_model_can_read_are_generated():
    """Three layouts the reader cannot serve, so the corpus supplies the model
    with none of them unless the generator does: the axes written down the
    side, a centroid stated in a column header, and `+/-30` standing for a peak
    in each hemisphere."""
    assert 0.01 < _rate(lambda t: t.notes.get("transposed", False)) < 0.06
    assert 0.01 < _rate(lambda t: t.notes.get("roi_centroids", False)) < 0.07
    assert 0.005 < _rate(lambda t: bool(t.notes.get("bilateral_rows"))) < 0.06


def test_a_bilateral_row_states_two_peaks_and_prints_one_number():
    """The target has to carry both, or the form teaches the model to read
    `+/-30` as a single positive x."""
    for t in TABLES:
        if not t.notes.get("bilateral_rows"):
            continue
        rendered = t.grid.render()
        assert "±" in rendered
        xs = {p.x for a in t.truth.analyses for p in a.points}
        assert any(-v in xs for v in xs if v), t.notes
        return
    raise AssertionError("no bilateral table in the sample")


def test_a_transposed_table_names_its_axes_down_the_side():
    for t in TABLES:
        if not t.notes.get("transposed"):
            continue
        rows = t.grid.render().split("\n")
        assert rows[1].split(" | ")[0].strip("#").lower() in (
            "x", "y", "z", "right", "anterior", "left", "posterior")
        assert len(t.truth.analyses) == 1 and len(t.truth.analyses[0].points) >= 3
        return
    raise AssertionError("no transposed table in the sample")


def test_the_negative_shapes_read_off_real_tables_are_all_generated():
    """Each was found by reading a table the gate got wrong. A shape the
    generator does not make is one the model only meets in deployment."""
    import re

    from nspond_tables.synth.empty import build_empty

    tables = [build_empty(seed=s) for s in range(1500)]
    text = "\n".join(t.grid.render() for t in tables)
    wanted = {
        "an F with its degrees of freedom": r"F\(\d+,\d+\) = ",
        "a count triple": r"\| \d+, \d+, \d+ \|",
        "mean, SD and a range": r"\d+\.\d+, \d+\.\d+, \d+-\d+",
        "scientific notation": r"\d\.\d x 10-\d+",
        "voxel dimensions": r"\d+ x \d+ x \d+",
        "a genomic position": r"Chr\d+ \| [\d,]{7,}",
        "an equilibrium point": r"E\d \| \(\d, \d, \d\)",
        "a figure legend": r"View larger version\(\d+K\)",
    }
    missing = [name for name, pattern in wanted.items()
               if not re.search(pattern, text)]
    assert not missing, missing


def test_a_table_that_holds_nothing_says_so_rather_than_refusing():
    from nspond_tables.synth.empty import build_empty

    for s in range(200):
        assert build_empty(seed=s).truth.as_target() == {"space": None,
                                                         "analyses": []}


def test_a_negative_is_not_separable_on_nonsense():
    """A `Side` column holding 4.10, a `Voxels` column holding `29.3 +/- 9.3`,
    a `T-value` holding a mean and an SD -- each makes the table answerable
    without reading it, which is the same shortcut as any other. The near-miss
    tables have to be wrong only in holding no coordinates."""
    import re

    from nspond_tables.synth.empty import build_empty

    checks = (
        (re.compile(r"^side$|hemisphere", re.I), re.compile(r"[LRB]|Left|Right")),
        (re.compile(r"voxels|^k$", re.I), re.compile(r"\d+")),
    )
    bad, checked = [], 0
    for seed in range(600):
        lines = build_empty(seed=seed).grid.render().split("\n")
        if not lines:
            continue
        heads = [h.strip("#<>0123456789:^ ") for h in lines[0].split(" | ")]
        for row in lines[1:]:
            cells = row.split(" | ")
            if len(cells) != len(heads):
                continue
            for head, cell in zip(heads, cells):
                checked += 1
                for pattern, allowed in checks:
                    if pattern.search(head) and not allowed.fullmatch(cell.strip()):
                        bad.append((head, cell))
                if re.search(r"[TZ][-\s]?value", head, re.I) and "±" in cell:
                    bad.append((head, cell))
    assert checked > 10000, checked
    assert not bad, bad[:6]


def test_a_rowspan_only_ever_carries_the_label_down():
    """`cells[1:]` trims the FIRST cell of a continuation row, so a rowspan is
    only safe where the first column holds the label. A table that opens on
    the triple would span the x column and trim x from every row beneath it:
    a coordinate lost per row, and a target asserting numbers no longer in
    the table."""
    for t in TABLES:
        layout = t.notes.get("layout") or []
        if not layout or layout[0] == "region":
            continue
        rendered = t.grid.render()
        assert "^2:" not in rendered.split("\n")[1] if len(
            rendered.split("\n")) > 1 else True
        for row in t.grid.rows:
            if row and row[0].rowspan > 1:
                raise AssertionError((t.notes.get("layout"), row[0].text))


def test_the_shapes_the_reader_reads_nothing_out_of_are_generated():
    """Read off 98 residual-route tables judged by hand: 41 held coordinates
    and the reader found none of them. That is the residual gate's whole
    positive class, and it had 34 examples."""
    for key, lo, hi in (("voxel_indices", 0.01, 0.08),
                        ("signs_spaced", 0.02, 0.10),
                        ("coords_first", 0.01, 0.07),
                        ("coords_in_name", 0.01, 0.07),
                        ("unmarked_span", 0.005, 0.06)):
        rate = _rate(lambda t, k=key: bool(t.notes.get(k)))
        assert lo < rate < hi, (key, rate)


def test_a_voxel_index_is_reported_as_the_table_prints_it():
    """`92 | 132 | 96` under a header reading `Peak MNI`. The target states
    what the table states; converting to millimetres would assert a
    normalisation the document never mentions."""
    for t in TABLES:
        if not t.notes.get("voxel_indices"):
            continue
        body = t.grid.render()
        for a in t.truth.analyses:
            for p in a.points:
                assert p.x > 0 and p.y > 0 and p.z > 0, p
                assert str(int(p.x)) in body, (p, body[:150])
        return
    raise AssertionError("no voxel-index table in the sample")
