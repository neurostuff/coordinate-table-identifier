"""The rendering contract: what a mark means and that it survives a round trip."""

import pytest

from nspond_tables.grid import COVERED, Cell, Grid, as_number, parse, render_cell


def test_span_digits_are_separated_from_the_text():
    """`^2` before `37` must not render as `^237`, which has three readings."""
    assert render_cell(Cell("37", rowspan=2)) == "^2:37"
    assert render_cell(Cell("1447", rowspan=3, colspan=2)) == "<2:^3:1447"
    assert render_cell(Cell("MNI", header=True, colspan=3)) == "#<3:MNI"


def test_a_plain_cell_carries_no_mark():
    assert render_cell(Cell("-42")) == "-42"
    assert render_cell(Cell("Region", header=True)) == "#Region"


def test_pipes_and_leading_hashes_are_escaped():
    assert render_cell(Cell("R|IFG")) == r"R\|IFG"
    assert render_cell(Cell("#4")) == r"\#4"
    assert parse(r"R\|IFG | \#4").rows[0][0].text == "R|IFG"
    assert parse(r"R\|IFG | \#4").rows[0][1].text == "#4"


def test_covered_cells_become_fillers_so_columns_line_up():
    g = Grid()
    g.add([Cell("Region", header=True, rowspan=2), Cell("MNI", header=True, colspan=3)])
    g.add([Cell("x", header=True), Cell("y", header=True), Cell("z", header=True)])
    g.add([Cell("L IFG", rowspan=2), Cell("-42"), Cell("18"), Cell("4")])
    g.add([Cell("-40"), Cell("20"), Cell("6")])
    lines = g.render().split("\n")
    assert lines[1] == "~ | #x | #y | #z"
    assert lines[3] == "~ | -40 | 20 | 6"
    # every row now occupies the same number of columns
    assert {sum(p.cell.colspan for p in row) for row in g.resolve()} == {4}


def test_a_filler_remembers_what_covers_it():
    g = Grid()
    g.add([Cell("L IFG", rowspan=2), Cell("-42")])
    g.add([Cell("-40")])
    second = g.resolve()[1]
    assert second[0].cell.is_filler
    assert second[0].cell.covers == "L IFG"


def test_round_trip_through_parse_is_stable():
    g = Grid()
    g.add([Cell("Region", header=True, rowspan=2), Cell("MNI", header=True, colspan=3)])
    g.add([Cell("x", header=True), Cell("y", header=True), Cell("z", header=True)])
    g.add([Cell("L IFG"), Cell("-42"), Cell("18"), Cell("4")])
    once = g.render()
    assert parse(once).render() == once


def test_resolving_an_already_filled_grid_does_not_grow_it():
    g = Grid()
    g.add([Cell("A", rowspan=2), Cell("1")])
    g.add([Cell("2")])
    filled = g.filled()
    assert filled.render() == g.render()
    assert filled.filled().render() == g.render()


@pytest.mark.parametrize("text,expected", [
    ("-42", -42.0),
    ("−42", -42.0),          # unicode minus
    ("< 0.001", 0.001),
    ("4.66†", 4.66),
    ("16,317", 16317.0),
    ("16 317", 16317.0),
    ("1,2", None),                # not a thousands separator
    ("L IFG", None),
    ("", None),
])
def test_as_number_reads_the_decorations_papers_use(text, expected):
    assert as_number(text) == expected


def test_rows_of_pure_markup_are_dropped():
    g = Grid()
    g.add([Cell(""), Cell("")])
    g.add([Cell("L IFG"), Cell("-42")])
    assert g.render() == "L IFG | -42"
