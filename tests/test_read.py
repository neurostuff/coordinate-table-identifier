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


def test_statistic_type_reads_a_header_and_resolves_by_priority():
    """A header naming several is resolved rather than declined. It used to
    return None, which threw the answer away on the commonest multi-statistic
    shape there is -- a test statistic printed beside its significance level."""
    from nspond_tables.fields import statistic_type

    assert statistic_type("Z value") == "Z"
    assert statistic_type("t(79)") == "T"
    assert statistic_type("p(FWE cor.)") == "P"
    assert statistic_type("Cohen's d") == "D"
    assert statistic_type("Hedges' g") == "G"
    assert statistic_type("value") is None
    assert statistic_type("T or Z") == "T"          # priority, not a decline
    assert statistic_type("Cohen's d (p<0.05)") == "D"


def test_the_priority_order_is_the_one_the_curation_uses():
    from nspond_tables.fields import STATISTIC_PRIORITY, best_of

    assert STATISTIC_PRIORITY == ("T", "Z", "D", "G", "F", "R", "B", "P")
    assert best_of({"P", "T"}) == "T"
    assert best_of({"P", "G"}) == "G"
    assert best_of({"F", "D"}) == "D"
    assert best_of({"B", "R"}) == "R"
    assert best_of(set()) is None


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


def test_an_axis_may_say_what_kind_of_coordinate_it_is():
    """`X coor`, `y coordinate`, `Z (Talairach)` all head coordinate columns.
    A statistic must not: `Z score`, `Z-max` and `z value` are not the z axis,
    which is the collision the adjacency rule exists for."""
    ok = ["X coor", "y coordinate", "Z (Talairach)", "x (mm)", "Y (MNI)", "z coords"]
    no = ["Z score", "Z-max", "Zmax", "z value", "k", "T"]
    for h in ok:
        assert any(read._AXIS[a].match(h) for a in read.AXES), h
    for h in no:
        assert not any(read._AXIS[a].match(h) for a in read.AXES), h


def test_a_note_after_a_triple_is_not_part_of_it():
    """`-56 -30 28 (OP1)` names the cytoarchitectonic area, `12 -44 8 (BA 40)`
    the Brodmann area. Neither is a coordinate."""
    assert read.triples_in("-56 -30 28 (OP1)") == [(-56.0, -30.0, 28.0)]
    assert read.triples_in("12 -44 8 (BA 40)") == [(12.0, -44.0, 8.0)]


def test_a_header_row_the_publisher_left_unmarked_is_still_a_header():
    """A publisher may mark the first header row and not the second, which is
    where the axis names usually live."""
    html = ("<table><tr><th></th><th colspan=3>Individual ROIs</th>"
            "<th colspan=3>Whole-brain</th></tr>"
            "<tr><td>ROI</td><td>x</td><td>y</td><td>z</td>"
            "<td>x</td><td>y</td><td>z</td></tr>"
            "<tr><td>RTPJ</td><td>58</td><td>-55</td><td>22</td>"
            "<td>56</td><td>-54</td><td>24</td></tr></table>")
    got = read.extract(serialize.serialize(html))
    assert got.axis_columns == {"x": 1, "y": 2, "z": 3}
    assert [p.as_tuple()[:3] for p in got.points] == [(58.0, -55.0, 22.0)]


def _blocks(markup):
    return read.coordinate_blocks(serialize.from_source(markup))


SIDE_BY_SIDE = """<table>
<tr><th>Region</th><th>x</th><th>y</th><th>z</th><th>t</th><th>x</th><th>y</th><th>z</th><th>t</th></tr>
<tr><td>Insula</td><td>30</td><td>20</td><td>4</td><td>5.1</td><td>-32</td><td>18</td><td>2</td><td>4.4</td></tr>
<tr><td>ACC</td><td>4</td><td>30</td><td>22</td><td>4.8</td><td>-6</td><td>28</td><td>24</td><td>4.0</td></tr>
</table>"""


TABLE_WITHOUT_COORDINATES = """<table><tr><th>Group</th><th>n</th><th>Age</th><th>IQ</th></tr>
<tr><td>Patients</td><td>20</td><td>34.1</td><td>102</td></tr>
<tr><td>Controls</td><td>22</td><td>33.8</td><td>104</td></tr></table>"""


@pytest.mark.parametrize(
    "markup, blocks",
    [
        (SIDE_BY_SIDE, [(1, 3), (5, 7)]),
        # Headed once over each block.
        ("""<table><tr><th>Region</th><th colspan="3">Task A MNI coordinates</th>
         <th colspan="3">Task B MNI coordinates</th></tr>
         <tr><td>Insula</td><td>30</td><td>20</td><td>4</td><td>-32</td><td>18</td><td>2</td></tr>
         <tr><td>ACC</td><td>4</td><td>30</td><td>22</td><td>-6</td><td>28</td><td>24</td></tr>
         </table>""", [(1, 3), (4, 6)]),
        # A column of whole triples per block.
        ("""<table><tr><th>Region</th><th>Task A</th><th>t</th><th>Task B</th><th>t</th></tr>
         <tr><td>Insula</td><td>30, 20, 4</td><td>5.1</td><td>-32, 18, 2</td><td>4.4</td></tr>
         <tr><td>ACC</td><td>4, 30, 22</td><td>4.8</td><td>-6, 28, 24</td><td>4.0</td></tr>
         </table>""", [(1, 1), (3, 3)]),
        # An x header spanning two columns widens its block.
        ("""<table><tr><th>Region</th><th colspan="2">x</th><th>y</th><th>z</th></tr>
         <tr><td>Insula</td><td colspan="2">-37</td><td>5</td><td>-14</td></tr>
         <tr><td>Insula</td><td></td><td>36</td><td>8</td><td>-14</td></tr>
         </table>""", [(1, 4)]),
        # F(df) cells parse as triples; beside named axes they are no block.
        ("""<table><tr><th>Region</th><th>F(df)</th><th>x</th><th>y</th><th>z</th></tr>
         <tr><td>IPL</td><td>19(1,30)</td><td>50</td><td>-41</td><td>60</td></tr>
         <tr><td>ACC</td><td>12(1,30)</td><td>4</td><td>30</td><td>22</td></tr>
         </table>""", [(2, 4)]),
        # "Voxel level" is a coordinate word over Z and p columns.
        ("""<table>
         <tr><th>Region</th><th colspan="3">Voxel level</th><th>x</th><th>y</th><th>z</th></tr>
         <tr><th></th><th>Z</th><th>p unc</th><th>p FWE</th><th></th><th></th><th></th></tr>
         <tr><td>PDI</td><td>4.80</td><td>&lt;0.001</td><td>0.03</td>
         <td>27</td><td>14</td><td>-32</td></tr>
         <tr><td>VAS</td><td>4.60</td><td>&lt;0.001</td><td>0.04</td>
         <td>18</td><td>2</td><td>-32</td></tr>
         </table>""", [(4, 6)]),
        (TABLE_WITHOUT_COORDINATES, []),
    ],
    ids=["named", "spanned", "packed", "wide-x", "statistic-triples", "spanned-statistics", "none"],
)
def test_every_block_of_coordinate_columns_is_found_left_to_right(markup, blocks):
    assert _blocks(markup) == blocks


# Real tables, cut to their header and first rows, with the blocks the review of
# the ns-pond column-group fix found misread.
#: A Brodmann column prints `39/40/7`, which reads as a triple.
BRODMANN_TRIPLE = """<table><thead><tr><th>Anatomical region</th><th>H</th><th>BA</th>
<th>MNI coordinates</th><th>Peak intensity</th><th>Nb of voxels</th></tr></thead><tbody>
<tr><td colspan="6">Auditory task</td></tr>
<tr><td>Angular gyrus</td><td>L</td><td>39/40/7</td><td>-44/-60/40</td><td>11.39</td><td>516</td></tr>
<tr><td>Angular gyrus</td><td>R</td><td>39</td><td>48/-74/32</td><td>7.47</td><td>36</td></tr>
<tr><td>Supramarginal gyrus</td><td>R</td><td>40</td><td>58/-42/34</td><td>13.45</td><td>179</td></tr>
<tr><td>Supramarginal gyrus</td><td>L</td><td>40/38/5</td><td>-62/-32/34</td><td>15.02</td><td>727</td></tr>
<tr><td colspan="6">Visual task</td></tr>
<tr><td>Angular gyrus</td><td>L</td><td>39/40/42</td><td>-44/-62/30</td><td>18.82</td><td>788</td></tr>
<tr><td>Supramarginal gyrus</td><td>R</td><td>40/42/21</td><td>60/-32/28</td><td>7.75</td><td>127</td></tr>
</tbody></table>"""
#: A region label quotes coordinates, which read as a triple (Elsevier CALS).
LABEL_TRIPLE = """<ce:table><tgroup cols="5"><thead><row><entry>Area</entry>
<entry>MNI coordinate (mm) of main peak</entry><entry>Cluster extent (voxels)</entry>
<entry>P-value</entry><entry>Z-value</entry></row></thead><tbody>
<row><entry namest="col1" nameend="col5">Response shifting &gt; object alternation</entry></row>
<row><entry>Right frontal cluster (38, 22, 30)</entry><entry>38, 22, 30</entry><entry>2245</entry>
<entry>&lt;0.001</entry><entry>4.86</entry></row>
<row><entry>Left frontal cluster (-40, 50, 6)</entry><entry>-40, 50, 6</entry><entry>2382</entry>
<entry>&lt;0.001</entry><entry>4.84</entry></row></tbody></tgroup>"""
#: The coordinate sub-header under "Left" sits one column left, over BA, x and y.
SHIFTED_SPAN = """<table><thead><tr><th rowspan="2">Lobe</th><th rowspan="2">Region</th><th>BA</th>
<th colspan="3">Left</th><th rowspan="2">t-score</th><th colspan="3">Right</th>
<th rowspan="2">t-score</th></tr>
<tr><th colspan="3">Talairach coordinate (x, y, z)</th><th>t-score</th>
<th colspan="3">Talairach coordinate (x, y, z)</th></tr></thead><tbody>
<tr><td colspan="11">major(-1)minor</td></tr>
<tr><td rowspan="5">Frontal</td><td>Superior Frontal Gyrus</td><td>6</td><td></td><td></td><td></td>
<td></td><td>6</td><td>16</td><td>59</td><td>4.51</td></tr>
<tr><td>Middle Frontal Gyrus</td><td>6</td><td>-26</td><td>4</td><td>48</td><td>5.32</td><td>28</td>
<td>-1</td><td>45</td><td>5.55</td></tr>
<tr><td>Middle Frontal gyrus</td><td>8</td><td></td><td></td><td></td><td></td><td>38</td><td>24</td>
<td>44</td><td>4.78</td></tr></tbody></table>"""
#: A column of triples with `-` where a session has no peak (ST40).
DASHED_TRIPLES = """<table><thead><tr><th rowspan="2">Structure</th><th rowspan="2">Hemisphere</th>
<th colspan="2">LV3</th><th colspan="2">LV2</th><th colspan="2">ST40</th><th colspan="2">Sham</th></tr>
<tr><th>MNI (mm) xyz</th><th>Peak T value</th><th>MNI (mm) xyz</th><th>Peak T value</th>
<th>MNI (mm) xyz</th><th>Peak T value</th><th>MNI (mm) xyz</th><th>Peak T value</th></tr></thead><tbody>
<tr><td rowspan="2">FP(BA10)</td><td>L</td><td>-3,60,0</td><td>-5,03</td><td>-6,54,-9</td><td>-5.23</td>
<td>-9,54,0</td><td>-4.77</td><td>-6,45,-6</td><td>-5.35</td></tr>
<tr><td>R</td><td>6,57,-3</td><td>-6.51</td><td>6,51,-12</td><td>-4.94</td><td>6,54,-12</td><td>-5.28</td>
<td>15,57,-3</td><td>-5.16</td></tr>
<tr><td rowspan="2">pregenual CC(BA32)</td><td>L</td><td>-</td><td>-</td><td>-6,36,0</td><td>-4.68</td>
<td>-</td><td>-</td><td>-3,39,0</td><td>-3.9</td></tr>
<tr><td>R</td><td>-</td><td>-</td><td>9,36,-3</td><td>-4.34</td><td>-</td><td>-</td><td>6,36,0</td>
<td>-4.03</td></tr></tbody></table>"""

#: Max; mean (SD) of a t map per region: each cell reads as a triple
#: (acdye54n2jq7, pone-0197715-t002).
MAX_MEAN_SD = """<table><thead>
<tr><th>Anatomical region</th><th>Left</th><th></th><th>Right</th><th></th></tr>
<tr><th></th><th>Extent</th><th>T</th><th>Extent</th><th>T</th></tr>
<tr><th></th><th>K(%clust;%reg)</th><th>Max; mean (SD)</th><th>K(%clust;%reg)</th>
<th>Max; mean (SD)</th></tr></thead><tbody>
<tr><td colspan="5">Cluster 1: kc= 42041; PcFWE&lt; 0.001; Voxel maximum x,y,z [mm]: -50,-76,23</td></tr>
<tr><td>Calcarine fissure</td><td>5521 (13.13; 30.56)</td><td>4.59; 2.96 (0.42)</td>
<td>4041 (9.61; 27.14)</td><td>5.04; 3.01 (0.51)</td></tr>
<tr><td>Middle occipital gyrus</td><td>5411 (12.87; 20.68)</td><td>5.18; 2.88 (0.41)</td>
<td></td><td></td></tr>
</tbody></table>"""
#: Odds ratios and their intervals, beside rows that quote MNI peaks
#: (tkghd73lhxrd, tbl0008).
ODDS_RATIOS = """<table><thead><tr><th></th><th>Univariate OR (95 % CI)</th><th>p-value</th>
<th>Multivariable OR (95 % CI)</th><th>p-value</th><th>Multivariable (BW-WALD) OR (95 % CI)</th>
<th>p-value</th></tr></thead><tbody>
<tr><td>Age</td><td>1.02 (0.94-1.11)</td><td>0.58</td><td>1.02 (0.87-1.19)</td><td>0.75</td>
<td>-</td><td></td></tr>
<tr><td>Education</td><td>0.70 (0.55-0.89)</td><td>0.004</td><td>0.06 (0.35-1.04)</td><td>0.07</td>
<td>-</td><td></td></tr>
<tr><td>Cluster 1: (MNI coordinate: -64 -10 -20)</td><td>0.08 (0.19-0.36)</td><td>0.001***</td>
<td>0,14 (0.13-1.54)</td><td>0.10</td><td>0.14 (0.02-0.84)</td><td>0.03*</td></tr>
<tr><td>Cluster 2: (MNI coordinate: 48 6 -32)</td><td>0.05 (0.09-0.35)</td><td>0.002**</td>
<td>0.073 (0.02-2.41)</td><td>0.14</td><td>0.11 (0.13-1.00)</td><td>0.05*</td></tr>
</tbody></table>"""


@pytest.mark.parametrize(
    "markup, blocks",
    [
        (BRODMANN_TRIPLE, [(3, 3)]),
        (LABEL_TRIPLE, [(1, 1)]),
        (SHIFTED_SPAN, [(3, 5), (7, 9)]),
        (DASHED_TRIPLES, [(2, 2), (4, 4), (6, 6), (8, 8)]),
        (MAX_MEAN_SD, []),
        (ODDS_RATIOS, []),
    ],
    ids=["brodmann-triple", "label-triple", "shifted-span", "dashed-triples", "max-mean-sd",
         "odds-ratios"],
)
def test_real_tables_count_every_block_and_only_blocks(markup, blocks):
    assert _blocks(markup) == blocks


def _table(header, *rows):
    """A table of one header row (a list of cells, or raw <tr> markup) and body rows."""
    head = header if isinstance(header, str) else "<tr>" + "".join(
        f"<th>{h}</th>" for h in header) + "</tr>"
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table>{head}{body}</table>"


@pytest.mark.parametrize(
    "markup, blocks",
    [
        # Two banners over three columns each; the x, y, z row has no cell under Region.
        (_table('<tr><th></th><th colspan="3">Task A</th><th colspan="3">Task B</th></tr>'
                "<tr><th>x</th><th>y</th><th>z</th><th>x</th><th>y</th><th>z</th></tr>",
                ["Insula", 30, 20, 4, -32, 18, 2], ["ACC", 4, 30, 22, -6, 28, 24]),
         [(1, 3), (4, 6)]),
        # "Peak voxel" spans Z, x and y; the named x, y, z wins.
        (_table('<tr><th rowspan="2">Region</th><th colspan="3">Peak voxel</th><th></th></tr>'
                "<tr><th>Z</th><th>x</th><th>y</th><th>z</th></tr>",
                ["Insula", 4.5, 30, 20, 4], ["ACC", 3.9, -4, 30, 22]),
         [(2, 4)]),
        # No header text at all: the body's two blocks.
        (_table(["", "", "", "", "", "", ""],
                ["Insula", -30, 20, 4, -32, 18, 2], ["ACC", 4, 30, 22, -6, 28, 24]),
         [(1, 3), (4, 6)]),
        # Statistics are never negative where an x would be.
        (_table(["Region", "BA", "k", "Z"], ["Insula", 13, 40, 4.5], ["ACC", 32, 25, 3.9]), []),
        # Beside named axes, a column of unit vectors is no block.
        (_table(["Region", "x", "y", "z", "Direction (x, y, z)"],
                ["Insula", 30, 20, 4, "0.32, -0.71, -0.47"],
                ["ACC", -4, 30, 22, "0.36, -0.69, -0.45"]),
         [(1, 3)]),
        # Axis names with a prefix or a footnote mark, in each of two blocks.
        (_table(["Region", "A x", "A y", "A z", "B x", "B y", "B z"],
                ["Insula", 30, 20, 4, 32, 18, 2], ["ACC", 4, 30, 22, 6, 28, 24]),
         [(1, 3), (4, 6)]),
        (_table(["Region", "x*", "y*", "z*", "x*", "y*", "z*"],
                ["Insula", 30, 20, 4, 32, 18, 2], ["ACC", 4, 30, 22, 6, 28, 24]),
         [(1, 3), (4, 6)]),
        # One coordinate header over two blocks, and one placed a column left of its block.
        (_table('<tr><th>Region</th><th colspan="6">MNI coordinates</th></tr>',
                ["Insula", 30, 20, 4, 32, 18, 2], ["ACC", 4, 30, 22, 6, 28, 24]),
         [(1, 3), (4, 6)]),
        (_table('<tr><th>Region</th><th colspan="3">MNI coordinates</th><th>t</th></tr>',
                ["Insula", "47/13", 30, 20, 4, 5.1], ["ACC", "8/9", 4, 30, 22, 4.8]),
         [(2, 4)]),
        # Unnamed columns: whole numbers with some x negative, not the BA and k
        # columns left of them, nor a t column that is negative too.
        (_table(["", "", "", "", "", ""], ["Insula", 13, 40, -30, 20, 4], ["ACC", 32, 25, 4, 30, 22]),
         [(3, 5)]),
        (_table(["", "", "", "", ""], ["Insula", -4.5, -30, 20, 4], ["ACC", 3.9, 4, 30, 22]),
         [(2, 4)]),
        # With no block named, a column of triples headed `x, y, z` rules out a
        # label column that quotes them.
        (_table(["Area", "x, y, z", "k"], ["Insula (-42, 18, 6)", "-42, 18, 6", 120],
                ["ACC (4, 30, 22)", "4, 30, 22", 80]),
         [(1, 1)]),
        # Triples of decimal millimetres count; triples no head holds do not.
        (_table(["Region", "Location", "k"], ["Insula", "-42.5, 18.3, 6.1", 120],
                ["ACC", "4.2, 30.6, 22.4", 80]),
         [(1, 1)]),
        (_table(["Model", "Sizes", "k"], ["A", "22, 961, 706", 120], ["B", "31, 112, 641", 80]), []),
        # A column of triples headed `Peak` rules out a label column headed `x, y, z`.
        (_table(["Region (x, y, z)", "Peak", "k"], ["Insula (-42, 18, 6)", "-42, 18, 6", 120],
                ["ACC (4, 30, 22)", "4, 30, 22", 80]),
         [(1, 1)]),
        # x and z with no y between them are not a block.
        (_table(["Region", "x", "t", "z"], ["Insula", 30, 4.5, 4], ["ACC", 4, 3.9, 22]), []),
        # Named axes over values no head holds, or over p values in most rows.
        (_table(["Region", "x", "y", "z"], ["A", 22, 961, 706], ["B", 31, 112, 641]), []),
        (_table(["Region", "x", "y", "z"], ["A", 30, 20, 4], ["B", 0.5, 0.001, 0.02],
                ["C", 0.4, 0.003, 0.01]),
         []),
    ],
    ids=["sub-header-shifted", "overlap", "body-only", "statistics", "direction-vectors",
         "axis-prefix", "axis-marker", "six-wide-span", "span-shifted", "integer-statistics",
         "negative-t", "label-beside-xyz", "decimal-millimetres", "outside-head-triples",
         "label-beside-peak", "no-y", "outside-head", "mostly-p-values"],
)
def test_blocks_come_from_the_table_shape_alone(markup, blocks):
    assert _blocks(markup) == blocks
