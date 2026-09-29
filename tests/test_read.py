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


# -- what the uncertain band taught the reader ----------------------------

SPANNED = """<table>
<tr><th>Region</th><th colspan="3">MNI coordinates (X Y Z)</th><th>mm3</th></tr>
<tr><td>ACC</td><td>-8</td><td>52</td><td>2</td><td>7816</td></tr>
<tr><td>r-SFG</td><td>20</td><td>-12</td><td>30</td><td>744</td></tr>
<tr><td>l-IFG</td><td>-52</td><td>34</td><td>2</td><td>544</td></tr>
</table>"""


def test_a_span_names_the_axes_when_nothing_else_does():
    """Six of the seventeen head their coordinates once and never write x."""
    got = read.extract(serialize.serialize(SPANNED))
    assert got.axis_columns == {"x": 1, "y": 2, "z": 3}
    assert [p.as_tuple()[:3] for p in got.points][0] == (-8.0, 52.0, 2.0)
    assert len(got.points) == 3


def test_a_span_that_holds_no_coordinates_is_not_taken_for_axes():
    html = SPANNED.replace("-8", "980").replace("52", "961").replace(
        "<td>2</td>", "<td>706</td>")
    assert read.extract(serialize.serialize(html)).axis_columns != {"x": 1, "y": 2, "z": 3}


def test_a_header_written_in_td_still_names_the_axes():
    """Three tables mark no header at all; the axis row is plain <td>."""
    html = """<table>
    <tr><td>Volume (mm3)</td><td>x</td><td>y</td><td>z</td><td>Location</td></tr>
    <tr><td>1064</td><td>-3</td><td>31</td><td>27</td><td>Left cingulate</td></tr>
    <tr><td>776</td><td>-5</td><td>-19</td><td>49</td><td>Left medial frontal</td></tr>
    </table>"""
    got = read.extract(serialize.serialize(html))
    assert got.axis_columns == {"x": 1, "y": 2, "z": 3}
    assert len(got.points) == 2


def test_the_body_settles_a_disagreement_about_which_columns_are_the_axes():
    """A sub-header row holding only the spanned cells lands at the left edge,
    so `x | y | z` registers over the region name. The numbers say otherwise."""
    html = """<table>
    <tr><td>Brain areas</td><td colspan="3">Coordinates</td><td>t value</td></tr>
    <tr><td>x</td><td>y</td><td>z</td></tr>
    <tr><td>Left Rectal Gyrus</td><td>-6</td><td>24</td><td>-21</td><td>3.70</td></tr>
    <tr><td>Left Midbrain</td><td>-15</td><td>-12</td><td>-9</td><td>3.37</td></tr>
    </table>"""
    got = read.extract(serialize.serialize(html))
    assert got.axis_columns == {"x": 1, "y": 2, "z": 3}
    assert [p.as_tuple()[:3] for p in got.points] == [(-6.0, 24.0, -21.0), (-15.0, -12.0, -9.0)]


def test_a_packed_triple_may_print_its_plus_and_detach_its_sign():
    html = """<table>
    <tr><th>Region</th><th>MNI x,y,z</th><th>t</th></tr>
    <tr><td>Hippocampus</td><td>-44 -90 + 12</td><td>25.43</td></tr>
    <tr><td>Insula</td><td>+68 -16 34</td><td>21.78</td></tr>
    </table>"""
    got = read.extract(serialize.serialize(html))
    assert [p.as_tuple()[:3] for p in got.points] == [(-44.0, -90.0, 12.0), (68.0, -16.0, 34.0)]


def test_a_row_may_state_more_than_one_peak():
    """A cluster with three peaks writes them all into the three axis cells."""
    html = """<table>
    <tr><th>Cluster</th><th>X</th><th>Y</th><th>Z</th></tr>
    <tr><td>frontal</td><td>-8/12//-6</td><td>56/48/52</td><td>38/48/30</td></tr>
    </table>"""
    got = read.extract(serialize.serialize(html))
    assert [p.as_tuple()[:3] for p in got.points] == [
        (-8.0, 56.0, 38.0), (12.0, 48.0, 48.0), (-6.0, 52.0, 30.0)]


def test_a_row_with_uneven_sub_values_is_not_guessed_at():
    html = """<table>
    <tr><th>Cluster</th><th>X</th><th>Y</th><th>Z</th></tr>
    <tr><td>frontal</td><td>-8/12</td><td>56/48/52</td><td>38</td></tr>
    </table>"""
    assert read.extract(serialize.serialize(html)).points == []


def test_the_reader_offers_candidates_and_does_not_judge_them():
    """`1.5 (0.3-7.8)` is an odds ratio with its interval and reads as a
    triple. The reader says so; the gate is what rejects it, because a reader
    strict enough to refuse it also refused real coordinates."""
    assert read.triples_in("1.5 (0.3-7.8)") == [(1.5, 0.3, -7.8)]
    assert read.triples_in("(-51, 20, 24)") == [(-51.0, 20.0, 24.0)]
    assert read.triples_in("-10-42 16") == [(-10.0, -42.0, 16.0)]
    assert read.triples_in("10.2; 20.5; 19.5") == [(10.2, 20.5, 19.5)]


def test_a_semicolon_separates_peaks_only_when_the_cell_is_not_one():
    assert read.triples_in("-16, -54, 46; -22, -54, 52") == [
        (-16.0, -54.0, 46.0), (-22.0, -54.0, 52.0)]


def test_a_statistic_reported_to_five_places_is_not_a_millimetre():
    """`(33, 33, 0.051118)` is a Mann-Whitney U with its p value, and all three
    sit inside a head. One table was labelled a coordinate table because of it."""
    assert read.triples_in("(33, 33, 0.051118)") == []


def test_an_axis_name_wearing_a_footnote_marker_is_still_an_axis():
    """`X (mm)d` heads a coordinate column in a real pdf table, and the marker
    made it invisible -- the table was dropped with its coordinates.

    `Z-maxc` keeps its `c`, because the letter follows a letter and so
    continues a word. It is a Z statistic, not the z axis, and must go on
    failing to be one."""
    html = ("<table><tr><th>Region (Brodmann)a</th><th>Hemb</th><th>Z-maxc</th>"
            "<th>X (mm)d</th><th>Y (mm)</th><th>Z (mm)</th></tr>"
            "<tr><td>DLPFC</td><td>R</td><td>3.37</td><td>38</td><td>20</td>"
            "<td>50</td></tr><tr><td>MFG</td><td>L</td><td>3.17</td><td>-52</td>"
            "<td>10</td><td>34</td></tr></table>")
    got = read.extract(serialize.serialize(html))
    assert got.axis_columns == {"x": 3, "y": 4, "z": 5}
    assert [p.as_tuple()[:3] for p in got.points] == [(38.0, 20.0, 50.0),
                                                      (-52.0, 10.0, 34.0)]


def test_a_bracket_holding_three_numbers_is_a_peak_and_two_is_a_range():
    """`9.91 [3, 15, 51]` is a statistic and then its peak: four numbers, so
    the cell as a whole is not a triple and the bracket says which three are
    the coordinate. The same rule separates a peak from an interval."""
    assert read.triples_in("9.91 [3, 15, 51]") == [(3.0, 15.0, 51.0)]
    assert read.triples_in("6.34 [-3, 12, 54]") == [(-3.0, 12.0, 54.0)]
    assert read.triples_in("F(2,38) = 4.1 (12, -44, 8)") == [(12.0, -44.0, 8.0)]


def test_every_column_of_triples_is_read_not_just_the_leftmost():
    """A table may give each region or contrast a column and put a whole
    triple in every cell, so one row holds a peak per column. Taking only the
    leftmost lost every other column of 157 tables."""
    html = ("<table><tr><th>Contrast</th><th>Striate L</th><th>Striate R</th>"
            "<th>Parietal L</th></tr>"
            "<tr><td>Target</td><td>-16 -100 -8</td><td>18 -96 2</td>"
            "<td>-30 -76 24</td></tr>"
            "<tr><td>Non-target</td><td>-16 -100 -6</td><td>16 -96 0</td>"
            "<td>-30 -78 24</td></tr></table>")
    got = read.extract(serialize.serialize(html))
    assert got.packed_columns == [1, 2, 3]
    assert len(got.points) == 6
