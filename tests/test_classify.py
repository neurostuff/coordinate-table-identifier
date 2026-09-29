"""The gate's contract: it is a recall device, and its features are honest."""

import math

import pytest

from nspond_tables import serialize
from nspond_tables.classify import dataset, features, model

COORDS = """<table>
<tr><th>Region</th><th>x</th><th>y</th><th>z</th><th>Z</th><th>k</th></tr>
<tr><td>L fusiform gyrus</td><td>-42</td><td>-55</td><td>-18</td><td>5.01</td><td>218</td></tr>
<tr><td>R inferior frontal gyrus</td><td>44</td><td>16</td><td>2</td><td>3.90</td><td>1447</td></tr>
<tr><td>Vermis</td><td>0</td><td>-57</td><td>-36</td><td>4.10</td><td>88</td></tr>
</table>"""

DEMOGRAPHICS = """<table>
<tr><th>Group</th><th>N</th><th>Age</th><th>Sex (F)</th></tr>
<tr><td>Patients</td><td>24</td><td>34.2</td><td>12</td></tr>
<tr><td>Controls</td><td>26</td><td>33.8</td><td>14</td></tr>
</table>"""


def _vec(html, caption=""):
    return features.vector(serialize.serialize(html), caption)


def test_every_named_feature_is_produced_and_finite():
    v = _vec(COORDS, "Activation peaks in MNI space")
    assert set(v) == set(features.NAMES)
    assert all(isinstance(x, float) and math.isfinite(x) for x in v.values())


def test_no_single_structural_feature_separates_the_two_tables():
    """`Patients | 24 | 34.2 | 12` is three numbers in coordinate range, so a
    demographics row trips the triple detector exactly as an activation row
    does. That is why the gate needs a vector rather than a rule."""
    a = _vec(COORDS, "Peak activations, MNI coordinates")
    b = _vec(DEMOGRAPHICS, "Participant demographics")
    assert a["frac_rows_with_triple"] == b["frac_rows_with_triple"] == 1.0


def test_a_coordinate_table_separates_from_a_demographics_table():
    a = _vec(COORDS, "Peak activations, MNI coordinates")
    b = _vec(DEMOGRAPHICS, "Participant demographics")
    assert a["has_axis_header"] == 1.0 and b["has_axis_header"] == 0.0
    assert a["frac_rows_with_region"] > b["frac_rows_with_region"]
    assert a["frac_negative"] > b["frac_negative"]          # ages are not signed
    assert a["coord_words_context"] > 0
    assert b["other_topic_context"] > 0


def test_an_empty_table_yields_zeros_rather_than_an_error():
    v = features.vector("", "", "")
    assert set(v) == set(features.NAMES)
    assert v["n_cells"] == 0.0


def test_features_are_computed_from_the_grid_not_the_raw_text():
    """Both orders of the same table give the same structural features."""
    grid = serialize.from_source(COORDS)
    assert features.vector(grid) == features.vector(grid.render())


# -- the gate -------------------------------------------------------------

def _toy():
    """A tiny separable set: coordinate tables against everything else."""
    rows, labels = [], []
    for i in range(12):
        html = COORDS.replace("-42", str(-40 - i))
        rows.append([_vec(html, "MNI coordinates")[n] for n in features.NAMES])
        labels.append(1)
        other = DEMOGRAPHICS.replace("24", str(20 + i))
        rows.append([_vec(other, "Participant demographics")[n] for n in features.NAMES])
        labels.append(0)
    return rows, labels


def test_the_gate_learns_the_separation():
    rows, labels = _toy()
    gate = model.fit(rows, labels, epochs=200)
    got = model.evaluate(gate, rows, labels)
    assert got["recall"] == 1.0
    assert got["precision"] >= 0.9


def test_the_threshold_is_the_lowest_that_clears_the_precision_floor():
    """Lowest, not best: every step down recovers a table that would otherwise
    be dropped for good, and precision is the only thing bounding the descent."""
    scores = [0.9, 0.8, 0.7, 0.6, 0.4, 0.2]
    labels = [1, 1, 0, 1, 0, 0]
    t_high, m_high = model.choose_threshold(scores, labels, precision_floor=0.95)
    t_low, m_low = model.choose_threshold(scores, labels, precision_floor=0.70)
    assert t_low <= t_high
    assert m_low["recall"] >= m_high["recall"]


def test_recall_is_never_traded_below_the_floor():
    scores = [0.9, 0.1, 0.85, 0.2]
    labels = [1, 0, 1, 0]
    _, m = model.choose_threshold(scores, labels, precision_floor=0.99)
    assert m["precision"] >= 0.99


def test_a_gate_survives_a_round_trip_to_disk(tmp_path):
    rows, labels = _toy()
    gate = model.fit(rows, labels, epochs=50)
    path = tmp_path / "gate.json"
    gate.save(path)
    again = model.Gate.load(path)
    assert again.threshold == gate.threshold
    assert again.names == gate.names
    assert again.score_row(rows[0]) == pytest.approx(gate.score_row(rows[0]))


def test_explain_names_the_features_driving_a_decision():
    rows, labels = _toy()
    gate = model.fit(rows, labels, epochs=100)
    why = gate.explain(serialize.serialize(COORDS), "MNI coordinates", top=3)
    assert len(why) == 3
    assert all(name in features.NAMES for name, _ in why)


# -- labelling ------------------------------------------------------------

def test_the_uncertain_band_is_excluded_from_training():
    """It is the population the gate exists to sort; training on it would be
    circular and scoring against it would flatter the result."""
    assert dataset.label_of({"label": "positive"}) == 1
    assert dataset.label_of({"label": "negative"}) == 0
    assert dataset.label_of({"label": "uncertain"}) is None
    assert dataset.label_of({"label": "uncertain", "extractor_found": True}) == 1


def test_a_split_keeps_every_table_of_an_article_on_one_side():
    """A paper's tables share a publisher and a house style; splitting by table
    leaks that across the boundary."""
    x = [[float(i)] for i in range(12)]
    y = [i % 2 for i in range(12)]
    groups = ["art%d" % (i // 3) for i in range(12)]
    (xtr, _), (xte, _) = dataset.split_by_article(x, y, groups, holdout=0.5, seed=0)
    tr_vals = {v[0] for v in xtr}
    te_vals = {v[0] for v in xte}
    assert not (tr_vals & te_vals)
    for start in range(0, 12, 3):
        block = {float(start), float(start + 1), float(start + 2)}
        assert block <= tr_vals or block <= te_vals


# -- which tables the gate is fitted on ------------------------------------

def test_the_gate_is_fitted_on_what_the_reader_leaves_in_question():
    """The residual, and the triples the reader reads out of tables that hold
    none. A table the reader reads correctly settles itself."""
    assert dataset.is_hard({"reader_points": 0.0}, 1)      # residual, real
    assert dataset.is_hard({"reader_points": 0.0}, 0)      # residual, not
    assert dataset.is_hard({"reader_points": 2.0}, 0)      # a false positive
    assert not dataset.is_hard({"reader_points": 2.0}, 1)  # settled


def test_building_keeps_the_easy_positives_out_unless_asked():
    rows = [{"label": "positive", "table_serialised": serialize.serialize(COORDS),
             "caption": "MNI"},
            {"label": "negative", "table_serialised": serialize.serialize(DEMOGRAPHICS),
             "caption": "Demographics"}]
    hard, _, _ = dataset.build(rows)
    everything, _, _ = dataset.build(rows, hard_only=False)
    assert len(hard) == 1 and len(everything) == 2
