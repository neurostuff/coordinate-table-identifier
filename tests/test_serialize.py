"""Markers a reader never sees are not part of a cell."""

from nspond_tables import serialize


def test_a_marker_a_reader_never_sees_is_not_part_of_the_cell():
    """Oxford Academic hides a sort marker in every <th>. It read as part of
    the axis name -- `x .` -- and cost three real coordinate tables."""
    html = ('<table><tr><th><em>x</em><span aria-hidden="true" '
            'style="display: none;">\n   . </span></th><th>y</th></tr>'
            '<tr><td>-42</td><td>16</td></tr></table>')
    assert serialize.serialize(html).splitlines()[0] == "#x | #y"


def test_a_table_inside_a_collapsed_container_survives():
    """Dropping every hidden element would drop the table with it."""
    html = ('<div style="display: none"><table><tr><th>x</th></tr>'
            '<tr><td>-42</td></tr></table></div>')
    assert serialize.serialize(html) == "#x\n-42"


def test_a_cell_holding_two_lines_does_not_fuse_them():
    """A journal writes two logical rows as two block elements in one cell:
    `<td><p>33, 39, 15</p><p>27, 51, 3</p></td>`. Stripping the tags to
    nothing fused them into `33, 39, 1527, 51, 3`, and `<p>7.26</p><p>4.17</p>`
    into `7.264.17` -- two decimal points, so a model echoing it emitted JSON
    that would not parse and the table's 57 coordinates were lost."""
    from nspond_tables.serialize import clean

    assert clean("<p>33, 39, 15</p><p>27, 51, 3</p>") == "33, 39, 15 27, 51, 3"
    assert clean("<p>7.26</p><p>4.17</p>") == "7.26 4.17"
    assert "7.264.17" not in clean("<p>7.26</p><p>4.17</p>")


def test_an_inline_tag_still_vanishes_without_a_trace():
    """`-<em>45</em>` has to stay `-45`. A separator there would make it
    `- 45`, which is not a number."""
    from nspond_tables.serialize import clean

    assert clean("-<em>45</em>") == "-45"
    assert clean("<span>x</span>y") == "xy"
    assert clean("<sub>2</sub>3") == "23"


def test_a_line_break_separates():
    from nspond_tables.serialize import clean

    assert clean("a<br/>b") == "a b"
    assert clean("a<br>b") == "a b"
    assert clean("−14<break/>−20<break/>−4") == "-14 -20 -4"


def test_a_sign_parted_from_its_digits_by_a_thin_space():
    """A journal writes `<td>- 45</td>`. `&#x02009;` is the fourth
    commonest entity in the corpus, and the hand-written whitespace class did
    not include it, so the cell read `- 45` and a number parser took `45` --
    the sign gone, a left-hemisphere focus on the right."""
    from nspond_tables.serialize import clean

    assert clean("<td>- 45</td>") == "-45"
    assert clean("<td>- 45</td>") == "-45"
    assert clean("<td>- 45</td>") == "-45"
    assert clean("<td>- 45</td>") == "-45"


def test_a_range_is_not_turned_into_two_numbers():
    """Only a sign starting the cell is rejoined. A dash between two numbers
    is a range or a subtraction."""
    from nspond_tables.serialize import clean

    assert clean("<td>10 - 20</td>") == "10 - 20"
    assert clean("<td>Region - name</td>") == "Region - name"
    assert clean("<td>5 -3</td>") == "5 -3"


def test_a_plus_sign_is_rejoined_too():
    from nspond_tables.serialize import clean

    assert clean("<td>+ 12</td>") == "+12"
    assert clean("<td>- 0.5</td>") == "-0.5"


def test_a_self_closing_cell_keeps_its_column():
    """`<td colspan="1" rowspan="1"/>` is how several publishers write an empty
    cell. Requiring a closing tag did not merely skip it: the engine ran on to
    the next `</td>`, consuming the empty cell and the one after it in a single
    match, so every value in the row moved one column left."""
    from nspond_tables import serialize

    raw = '<tr><td>Fusiform</td><td rowspan="1" colspan="1"/>' \
          '<td>37.2</td><td>-41.9</td></tr>'
    assert serialize.serialize(raw) == "Fusiform |  | 37.2 | -41.9"


def test_a_self_closing_cals_entry_keeps_its_column():
    from nspond_tables import serialize

    raw = "<row><entry>Fusiform</entry><entry/><entry>37.2</entry></row>"
    assert serialize.serialize(raw) == "Fusiform |  | 37.2"


def test_an_elsevier_space_between_two_numbers_separates_them():
    """Elsevier writes the space in `−24 54 −6` as `<hsp sp="0.25"/>`, and
    stripping it like any inline tag read the triplet as `-2454-6`."""
    from nspond_tables.serialize import clean

    assert clean('−24<hsp sp="0.25"/>54<hsp sp="0.25"/>−6') == "-24 54 -6"
    assert clean('8<ce:hsp sp="0.25"/>44') == "8 44"
    assert clean('18.7<hsp sp="0.12"/>+<hsp sp="0.12"/>10.2') == "18.7 +10.2"


def test_an_elsevier_space_elsewhere_still_vanishes():
    """A sign and its digits, and the groups of a count, stay one number."""
    from nspond_tables.serialize import clean

    assert clean('−<hsp sp="0.10"/>52') == "-52"
    assert clean('R +<hsp sp="0.10"/>80') == "R +80"
    assert clean('25<hsp sp="0.25"/>000') == "25000"
    assert clean('62<ce:hsp sp="0.25"/>077 voxels') == "62077 voxels"
    assert clean('&lt;<hsp sp="0.10"/>0.001') == "<0.001"
    assert clean('3200, 25<hsp sp="0.25"/>000, 20000') == "3200, 25000, 20000"


def test_a_gap_misplaced_inside_a_two_digit_coordinate_does_not_split_it():
    """A cell of one digit, a gap and one digit sits in a column of two-digit
    coordinates: `−3<hsp/>5` is -35. Two real numbers keep their space."""
    from nspond_tables.serialize import clean

    assert clean('−3<hsp sp="0.10"/>5') == "-35"
    assert clean('<hsp sp="0.10"/>2<hsp sp="0.10"/>0') == "20"
    assert clean('0.62<hsp sp="0.25"/>0.004') == "0.62 0.004"


def test_a_namespaced_cals_cell_is_a_cell():
    """Elsevier writes some cells `<ce:entry>` inside a plain `<row>`. Looking
    for `<entry` alone sent the table to the CSV reader, which fused each row
    into one run of numbers."""
    cals = ('<ce:table><tgroup cols="4"><thead><row><ce:entry>Region</ce:entry>'
            '<ce:entry>x</ce:entry><ce:entry>y</ce:entry><ce:entry>z</ce:entry></row></thead>'
            '<tbody><row><ce:entry>Left precentral gyrus</ce:entry><ce:entry>−26</ce:entry>'
            '<ce:entry>−18</ce:entry><ce:entry>65</ce:entry></row></tbody></tgroup></ce:table>')
    assert serialize.serialize(cals).splitlines() == [
        "#Region | #x | #y | #z", "Left precentral gyrus | -26 | -18 | 65"]


def test_a_row_stacking_several_foci_becomes_one_row_per_focus():
    """JATS breaks every cell of a row the same number of times to stack
    foci. Spaced out, the x column read `8 24 8`, which looks like a triplet."""
    html = ('<table><tr><th>Trait</th><th>Region</th><th>x</th><th>y</th><th>z</th></tr>'
            '<tr><td>Novelty Seeking</td><td>Hypothalamus<break/>Amygdala<break/>Caudate</td>'
            '<td>8<break/>24<break/>8</td><td>2<break/>−6<break/>18</td>'
            '<td>−14<break/>−20<break/>−4</td></tr>'
            '<tr><td>Reward</td><td>Thalamus</td><td>−18</td><td>−26</td><td>6</td></tr></table>')
    assert serialize.serialize(html).splitlines() == [
        "#Trait | #Region | #x | #y | #z",
        "^3:Novelty Seeking | Hypothalamus | 8 | 2 | -14",
        "~ | Amygdala | 24 | -6 | -20",
        "~ | Caudate | 8 | 18 | -4",
        "Reward | Thalamus | -18 | -26 | 6",
    ]


def test_a_wrapped_header_is_not_a_stack():
    html = ('<table><tr><td>Cluster<break/>size</td><td>Z<break/>score</td></tr>'
            '<tr><td>120</td><td>4.1</td></tr></table>')
    assert serialize.serialize(html).splitlines()[0] == "Cluster size | Z score"


def test_a_single_broken_cell_is_spaced_not_split():
    html = '<table><tr><td>L</td><td>33, 2, −3<break/>−35, −4, 0</td></tr></table>'
    assert serialize.serialize(html) == "L | 33, 2, -3 -35, -4, 0"


def test_a_value_and_its_annotation_stay_one_row():
    html = ('<table><tr><td>CT</td><td>0.54<break/>(3.35x10<sup>-21</sup>)</td>'
            '<td>0.41<break/>(5.03x10<sup>-12</sup>)</td></tr></table>')
    assert len(serialize.serialize(html).splitlines()) == 1


def test_a_table_with_rowspans_is_not_restacked():
    html = ('<table><tr><td rowspan="2">Dual</td><td>1<break/>2</td><td>3<break/>4</td></tr>'
            '<tr><td>5</td><td>6</td></tr></table>')
    assert serialize.serialize(html).splitlines() == ["^2:Dual | 1 2 | 3 4", "~ | 5 | 6"]
