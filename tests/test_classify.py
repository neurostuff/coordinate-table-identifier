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


def test_the_threshold_is_the_highest_that_keeps_the_positives():
    """Not the lowest that clears a precision floor, which reads as the
    generous choice and is not: precision is measured where the gate was
    fitted and the threshold is used somewhere else. On a population that is
    83.5% positive, a threshold below the negatives' own median still scores
    91% precision -- and lets nearly everything through in a corpus where far
    fewer tables hold coordinates.

    The scores are bimodal, so the threshold belongs at the top of the gap."""
    scores = [0.99, 0.97, 0.96, 0.03, 0.02, 0.001]
    labels = [1, 1, 1, 0, 0, 0]
    t, m = model.choose_threshold(scores, labels, precision_floor=0.90)
    assert m["recall"] == 1.0
    assert m["precision"] == 1.0
    assert t > 0.03                       # above every negative, not below them


def test_a_recall_floor_below_one_may_give_up_a_positive_for_a_much_higher_bar():
    scores = [0.99, 0.97, 0.05, 0.04, 0.02]
    labels = [1, 1, 1, 0, 0]
    t, m = model.choose_threshold(scores, labels, precision_floor=0.5, recall_floor=0.6)
    assert t > 0.05 and m["recall"] < 1.0 and m["precision"] == 1.0

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
    corroborated = {"label": "uncertain", "extractor_found": True,
                    "table_serialised": serialize.serialize(COORDS),
                    "caption": "MNI coordinates"}
    assert dataset.label_of(corroborated) == 1


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


# -- which gate decides a table --------------------------------------------

def test_a_table_is_routed_by_what_the_reader_made_of_it():
    """The route needs no label, which is what makes two gates possible at
    inference and not only in training."""
    assert dataset.population_of(_vec(COORDS, "MNI")) == dataset.CANDIDATES
    assert dataset.population_of(_vec(DEMOGRAPHICS, "Demographics")) == dataset.RESIDUAL


def test_each_gate_is_built_from_its_own_population():
    rows = [{"label": "positive", "table_serialised": serialize.serialize(COORDS),
             "caption": "MNI"},
            {"label": "negative", "table_serialised": serialize.serialize(DEMOGRAPHICS),
             "caption": "Demographics"}]
    assert len(dataset.build(rows, population=dataset.CANDIDATES)[0]) == 1
    assert len(dataset.build(rows, population=dataset.RESIDUAL)[0]) == 1
    assert len(dataset.build(rows)[0]) == 2


def test_a_routed_gate_sends_a_table_to_the_gate_that_was_fitted_on_it(tmp_path):
    rows, labels = _toy()
    pair = model.RoutedGate(candidates=model.fit(rows, labels, epochs=50),
                            residual=model.fit(rows, labels, epochs=50))
    assert pair.gate_for(serialize.serialize(COORDS), "MNI") is pair.candidates
    assert pair.gate_for(serialize.serialize(DEMOGRAPHICS), "Demographics") is pair.residual
    path = tmp_path / "pair.json"
    pair.save(path)
    again = model.RoutedGate.load(path)
    assert again.candidates.threshold == pair.candidates.threshold
    assert again.residual.names == pair.residual.names


def test_the_forest_has_the_same_surface_as_the_gate():
    """The residual side needs a model that can express combinations; the
    candidate side does not. Both must be usable the same way."""
    pytest.importorskip("sklearn")
    rows, labels = _toy()
    forest = model.fit_forest(rows, labels, n_estimators=20)
    assert 0.0 <= forest.score(serialize.serialize(COORDS), "MNI coordinates") <= 1.0
    assert forest.predict(serialize.serialize(COORDS), "MNI coordinates") in (True, False)
    assert all(n in features.NAMES for n, _ in forest.explain(
        serialize.serialize(COORDS), "MNI", top=3))


def test_a_rejected_candidate_is_not_put_to_the_other_gate():
    """The route is a partition. A table the candidate gate rejects stays
    rejected; sending it on to the residual gate would undo the decision, and
    that gate has never seen a table the reader read anything out of."""
    rows, labels = _toy()
    strict = model.fit(rows, labels, epochs=100)
    strict.threshold = 1.1                      # rejects everything
    lenient = model.fit(rows, labels, epochs=100)
    lenient.threshold = -0.1                    # accepts everything
    pair = model.RoutedGate(candidates=strict, residual=lenient)
    table = serialize.serialize(COORDS)
    assert pair.gate_for(table, "MNI coordinates") is strict
    assert pair.predict(table, "MNI coordinates") is False


def test_a_verdict_records_the_route_that_produced_it():
    """The reader changes, so the route changes, so a stored verdict has to say
    which gate made it and on what evidence."""
    rows, labels = _toy()
    pair = model.RoutedGate(candidates=model.fit(rows, labels, epochs=50),
                            residual=model.fit(rows, labels, epochs=50))
    got = pair.decide(serialize.serialize(COORDS), "MNI coordinates")
    assert got["route"] == "candidates" and got["reader_points"] > 0
    assert got["passes"] == (got["score"] >= got["threshold"])
    assert pair.decide(serialize.serialize(DEMOGRAPHICS), "Demographics")["route"] == "residual"


def test_a_pair_holding_a_forest_survives_a_round_trip(tmp_path):
    """A forest cannot be written as JSON, and the residual side is one."""
    pytest.importorskip("sklearn")
    pytest.importorskip("joblib")
    rows, labels = _toy()
    pair = model.RoutedGate(candidates=model.fit(rows, labels, epochs=50),
                            residual=model.fit_forest(rows, labels, n_estimators=20))
    path = tmp_path / "gate.joblib"
    pair.save(path)
    again = model.RoutedGate.load(path)
    assert isinstance(again.residual, model.Forest)
    assert again.residual.threshold == pair.residual.threshold
    assert again.candidates.threshold == pair.candidates.threshold
    table = serialize.serialize(DEMOGRAPHICS)
    assert again.decide(table, "Demographics")["route"] == "residual"
    assert again.residual.score_row([0.0] * len(again.residual.names)) >= 0.0


def test_a_big_table_is_not_rejected_for_being_big():
    """A count has no ceiling, and a linear model fitted where `max_rowspan`
    is a median of 1 and a maximum of 40 extrapolates on one of 112. A 492-row
    ALE source table headed `MNI/Talairach` and `Coordinates` scored 0.0007
    and was dropped; its first 120 rows scored 0.984, and the only difference
    was its rowspan."""
    small = "#Study | #<3:Coordinates\n" + "\n".join(
        "S%d | %d | %d | %d" % (i, i - 30, -i, i) for i in range(1, 12))
    big = "#Study | #<3:Coordinates\n" + "\n".join(
        "S%d | %d | %d | %d" % (i, (i % 60) - 30, -(i % 80), i % 50)
        for i in range(1, 400))
    a, b = features.vector(small), features.vector(big)
    # log1p keeps the ordering without letting the difference dominate
    assert a["n_rows"] < b["n_rows"]
    assert b["n_rows"] - a["n_rows"] < 4.0
    assert b["n_cells"] - a["n_cells"] < 4.0


# -- the threshold has to describe a table the model has not seen ---------

def test_the_forest_threshold_is_chosen_out_of_fold():
    """A forest scores its own training rows at almost 0 or almost 1, so a
    threshold read off them is met by any cut in the gap and means nothing.
    Read that way, 0.694 looked like 99% recall and delivered 56%."""
    import random

    from nspond_tables.classify import model

    rng = random.Random(0)
    rows, labels = [], []
    for i in range(200):
        y = int(i % 4 == 0)
        # One weak signal and four columns of noise: separable in sample,
        # not out of it.
        rows.append([y * 0.6 + rng.gauss(0, 1)] + [rng.gauss(0, 1) for _ in range(4)])
        labels.append(y)
    f = model.fit_forest(rows, labels, n_estimators=60, precision_floor=0.0,
                         recall_floor=0.97)
    assert f.metrics["out_of_fold"] is True
    in_sample = [f.score_row(r) for r in rows]
    kept = sum(1 for s, y in zip(in_sample, labels) if s >= f.threshold and y)
    # In sample it keeps every positive with room to spare; the point is that
    # the threshold was not chosen to make that true.
    assert kept == sum(labels)
    assert f.metrics["recall"] >= 0.97


def test_the_residual_gate_runs_for_recall_with_no_precision_floor():
    """3.0% of the tables it sees hold coordinates. Asking for 97% recall and
    90% precision at once asks for a point that does not exist, and a floor
    that cannot be met hands back whatever the search settled on."""
    import inspect

    from nspond_tables.classify import model

    sig = inspect.signature(model.fit_routed)
    assert sig.parameters["residual_floor"].default == 0.0
    assert sig.parameters["residual_recall"].default == 0.97


def test_too_few_of_one_class_to_hold_any_out_says_so(monkeypatch):
    """Rather than pretending the in-sample threshold is out of fold."""
    from nspond_tables.classify import model

    rows = [[float(i), 0.0] for i in range(12)]
    labels = [1] + [0] * 11
    f = model.fit_forest(rows, labels, n_estimators=10, precision_floor=0.0)
    assert f.metrics["out_of_fold"] is False
