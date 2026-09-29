"""What a reader can determine without a model, and what it must refuse to guess."""

import pytest

from nspond_tables import read, serialize

TABLE = """<table>
<tr><th rowspan="2">Region</th><th colspan="3">MNI coordinates</th>
    <th rowspan="2">Cluster size (voxels)</th><th rowspan="2">Z value</th></tr>
<tr><th>x</th><th>y</th><th>z</th></tr>
<tr><td colspan="6">Faces &gt; Houses</td></tr>
<tr><td rowspan="2">L fusiform</td><td>-42</td><td>-55</td><td>-18</td><td>218</td><td>5.01</td></tr>
<tr><td>-40</td><td>-52</td><td>-20</td><td>37</td><td>4.22</td></tr>
<tr><td>R IFG</td><td>44</td><td>16</td><td>2</td><td>1447</td><td>3.90</td></tr>
</table>"""


@pytest.fixture
def result():
    return read.extract(serialize.serialize(TABLE),
                        caption="Table 1. Activations in MNI space.")


def test_the_axis_header_locates_the_coordinate_columns(result):
    assert result.axis_columns == {"x": 1, "y": 2, "z": 3}
    assert result.located_by == "header"


def test_every_point_is_read_from_its_own_row(result):
    assert [(p.x, p.y, p.z) for p in result.points] == [
        (-42.0, -55.0, -18.0), (-40.0, -52.0, -20.0), (44.0, 16.0, 2.0)]


def test_statistic_type_and_extent_come_from_their_headers(result):
    assert [p.statistic_type for p in result.points] == ["Z", "Z", "Z"]
    assert [p.statistic_value for p in result.points] == [5.01, 4.22, 3.90]
    assert [p.extent for p in result.points] == [218.0, 37.0, 1447.0]
    assert result.measure == "voxels"


def test_a_label_is_inherited_through_a_rowspan(result):
    assert [p.label for p in result.points] == ["L fusiform", "L fusiform", "R IFG"]


def test_banner_rows_are_reported_not_treated_as_points(result):
    assert result.sections == [(2, "Faces > Houses")]
    assert len(result.points) == 3


def test_space_is_read_from_the_caption_when_the_table_is_silent():
    silent = """<table><tr><th>Region</th><th>x</th><th>y</th><th>z</th></tr>
    <tr><td>L IFG</td><td>-42</td><td>18</td><td>4</td></tr></table>"""
    text = serialize.serialize(silent)
    assert read.extract(text, caption="reported in MNI space").space == "MNI"
    assert read.extract(text, caption="Talairach coordinates").space == "TAL"


def test_a_table_that_names_its_own_space_is_read_from_the_table(result):
    # TABLE's header says "MNI coordinates", so no caption is needed.
    assert result.space == "MNI"
    # ...and a caption naming the other space makes it genuinely ambiguous.
    both = read.extract(serialize.serialize(TABLE), caption="Talairach coordinates")
    assert both.space is None


def test_space_is_none_when_nothing_states_it_and_when_both_do():
    bare = "<table><tr><th>x</th><th>y</th><th>z</th></tr>" \
           "<tr><td>2</td><td>4</td><td>6</td></tr></table>"
    assert read.extract(serialize.serialize(bare)).space is None
    assert read.extract(serialize.serialize(bare),
                        caption="MNI converted from Talairach").space is None


def test_a_table_with_no_axis_header_yields_no_points_rather_than_guesses():
    blind = "<table><tr><td>L IFG</td><td>-42</td><td>18</td><td>4</td></tr></table>"
    got = read.extract(serialize.serialize(blind))
    assert got.located_by == "none"
    assert got.points == []


def test_a_packed_triple_column_is_found_when_no_axis_header_exists():
    packed = """<table>
    <tr><th>Region</th><th>Coordinates</th><th>Z</th></tr>
    <tr><td>L IFG</td><td>-42 -55 -18</td><td>5.01</td></tr>
    <tr><td>R IFG</td><td>44, 16, 2</td><td>3.90</td></tr>
    <tr><td>Vermis</td><td>0 -57 -36</td><td>4.10</td></tr></table>"""
    got = read.extract(serialize.serialize(packed))
    assert got.located_by == "packed cell"
    assert (-42.0, -55.0, -18.0) in [(p.x, p.y, p.z) for p in got.points]


def test_a_left_label_with_a_positive_x_is_counted_as_a_disagreement():
    wrong = """<table><tr><th>Region</th><th>x</th><th>y</th><th>z</th></tr>
    <tr><td>L fusiform</td><td>42</td><td>-55</td><td>-18</td></tr></table>"""
    assert read.extract(serialize.serialize(wrong)).sign_disagreements == 1


def test_statistic_type_is_none_when_the_header_names_two_or_none():
    from nspond_tables.fields import statistic_type
    assert statistic_type("Z value") == "Z"
    assert statistic_type("t(79)") == "T"
    assert statistic_type("p(FWE cor.)") == "P"
    assert statistic_type("value") is None
    assert statistic_type("T or Z") is None


def test_numbers_in_reads_a_composite_cell():
    assert read.numbers_in("t(79)=6.24") == [79.0, 6.24]
    assert read.numbers_in("-4.0 (18, -26, 62)") == [-4.0, 18.0, -26.0, 62.0]
    assert read.numbers_in("16 317") == [16317.0]
