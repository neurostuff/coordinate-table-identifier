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
