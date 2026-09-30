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
